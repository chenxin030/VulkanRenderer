# SSR（Screen Space Reflection）实现说明（`RENDERING_LEVEL == 7`）

[返回目录](../../README.md)

概述：**基于深度重建 + 屏幕空间步进**：先复制主场景颜色，再通过全屏着色器计算命中并混合反射。

---

## 1. 整体渲染流程

单帧渲染的大致顺序是：

1. **Shadow Pass（深度）**：渲染阴影图（`shadowMap`）
2. **Main Lit Pass（场景着色）**：将带阴影的场景直接渲染到 `swapchain` 颜色附件，并写入主深度
3. **SSR Composite Pass（全屏三角形）**：
   - 先把当前 `swapchain` 颜色拷贝到 `ssrColorData`（作为“反射采样源”）
   - 再执行全屏 SSR 光线步进，并将反射结果混合回 `swapchain`
4. **UI Pass**：叠加 ImGui
5. **Present**

对应：
- 主命令录制：`src/Core/Renderer_rendering.cpp` 的 `recordCommandBuffer()`
- SSR 录制：`src/Core/Renderer_SSR.cpp` 的 `recordSSR()`

---

## 2. C++ 侧：SSR用到的对象及作用

### 2.1 主要成员对象

- `MeshBuffer ssrSceneUboResources`: 传 `projection/view/invProjection/camera` 等参数给 SSR shader
- `MeshBuffer ssrParamsUboResources`: 传 SSR 可调参数（步长、厚度、最大步数、强度等）
- `TextureData ssrColorData`: 保存主场景颜色拷贝（用于反射采样）

- 参数：
  - `int ssrMaxSteps = 64`
  - `float ssrMaxRayDistance = 16.0f`
  - `float ssrThickness = 0.12f`
  - `float ssrStride = 0.25f`
  - `float ssrIntensity = 0.5f`
  - `bool ssrEnabled = true`

### 2.2 UBO 结构（`Renderer_SSR.cpp`）

- `SSRSceneUBO`
  - `projection`
  - `view`
  - `invProjection`
  - `cameraPosNear`
    - `xyz`：相机世界坐标（当前 `ssr.slang` 主流程里未直接使用，但可用于扩展：例如按相机距离衰减反射、世界空间调试可视化、混合其他反射技术时做权重计算）
    - `w`：近裁剪面 `near`（用于与深度重建/线性深度相关的扩展，当前版本主要保留为统一场景参数）
  - `cameraFarPadding`
    - `x`：远裁剪面 `far`（与 `near` 配套，便于在 shader 中进行深度区间归一化、距离相关衰减或调试输出）
    - `yzw`：padding，对齐用（保证 UBO 在 std140/std430 对齐下布局稳定，避免跨编译器/平台出现错读）

- `SSRParams`
  - `maxRayDistance`
  - `thickness`
    - 深度命中厚度阈值（容差）。
    - 越大：更容易“命中”；但过大容易穿帮、粘连、出现错误反射。
  - `stride`
    - 每次步进长度（步长）。
    - 越小：命中更细致、边缘更平滑；但循环次数实际更高、性能更差。
  - `maxSteps`
  - `intensity`
    - 反射叠加总强度系数。
    - 直接影响“反射有多明显”。
  - `invResolution`
    - 屏幕分辨率倒数（`1/width, 1/height`）。
    - 用于邻域采样偏移（如法线估计、反射稳化采样）时的“1 像素单位”。
  - `debugMode` ： 调试显示模式开关
    - `0`：正常输出
    - `1`：命中掩码（HitMask）
      - 含义：显示当前像素的 SSR 射线是否命中几何。
      - 颜色语义：命中为红色（`1`），未命中为黑色（`0`）。
      - 用途：快速判断“为什么某区域没有反射”——如果大面积是黑色，说明射线没打到有效屏幕内几何（可能是步长过大、最大步数不足、厚度阈值不合适或目标在屏幕外）。
    - `2`：步进进度可视化（Steps）
      - 含义：显示命中发生在第几步（归一化到 `0~1`，越亮通常代表命中越晚/射线走得更远）。
      - 颜色语义：灰度值 = `hitSteps`。
      - 用途：观察参数是否合理：
        - 普遍偏亮：通常说明步长太小或最大距离太远，很多命中发生在后段；
        - 普遍偏暗但命中少：可能步长太大导致跨过细节；
        - 局部突变：可能存在深度不连续引发的不稳定命中。
    - `3`：深度图可视化（Depth）

### 2.3 描述符绑定关系

在 `createSSRDescriptorSetLayout()` 中：

- `binding 0`：`UniformBuffer`（`SSRSceneUBO`）
- `binding 1`：`CombinedImageSampler`（主深度 `depthData`）
- `binding 2`：`CombinedImageSampler`（场景颜色拷贝 `ssrColorData`）
- `binding 3`：`UniformBuffer`（`SSRParams`）

对应着色器 `shaders/ssr.slang`。

---

## 3. C++ 侧：SSR生命周期与执行过程

### 3.1 每帧更新

`updateSSRBuffers(currentFrame)` 会更新两块 UBO：

- 场景矩阵与相机参数（用于重建 view-space 位置）
- SSR 参数（射线步进参数、强度、调试模式等）

### 3.2 执行 SSR Pass（`recordSSR`）

`recordSSR(commandBuffer, imageIndex)` 的关键步骤：

1. 若 `ssrEnabled == false`，直接跳过
2. 将 `swapchain` 图像转为 `TransferSrc`
3. 将 `ssrColorData` 转为 `TransferDst`
4. `blitImage`：把当前 `swapchain` 颜色拷贝到 `ssrColorData`
5. `ssrColorData` 转为 `ShaderReadOnly`
6. `swapchain` 转回 `ColorAttachment`
7. 主深度图 `depthData` 转为 `DepthReadOnly`（供片段着色器采样）
8. 开始动态渲染（`loadOp = Load`），绑定 SSR 管线并 `draw(3,1,0,0)` 绘制全屏三角形
9. 结束后将深度布局恢复为 `DepthAttachmentOptimal`

> 注意：SSR pass 是“在已完成主场景颜色基础上做后处理混合”，并非单独 GBuffer 管线。

---

## 4. 着色器侧：对象及作用（`shaders/ssr.slang`）

### 4.3 FS 关键函数

- `reconstructViewPos(uv, depth)`
  - 用 `invProjection` 从屏幕坐标重建 View 空间位置
- `estimateNormal(uv)`
  - 通过邻域深度（右/上）差分估计View 空间法线
- `projectToUV(viewPos)`
  - View 空间点投影回屏幕 UV
- `raymarchSSR(originVS, dirVS, out hitUV, out hitSteps)`
  - 在屏幕空间进行射线步进，检测与深度重建几何是否相交
  - 命中后进行简短二分 refine

### 4.4 FS 主流程 `fragMain`

1. 读取当前像素深度，过滤背景
2. 重建 `posVS`、估算法线 `N`
3. 由视线向量 `V` 与法线求反射方向 `R`
4. 调 `raymarchSSR` 获得命中位置 `hitUV`
5. `baseColor = colorTex.Sample(uv)`
6. `refl = hit ? colorTex.Sample(hitUV) : 0`
7. 边缘衰减 `fade`
8. 按掩码控制反射贡献并叠加到 `baseColor`

输出最终颜色到 `swapchain`。

---

## 5. “只有白色地板（白色 cube）拥有反射”

采用**屏幕空间掩码**限制反射：

1. **白色判定 `whiteMask`**：
   - 高亮度（`luminance` 高）
   - 低饱和（`saturation` 低）
2. **地板朝向判定 `floorMask`**：
   - 法线与“世界上方向在视空间下的方向”点积较大（朝上）
3. 最终：
   - `reflectMask = saturate(whiteMask * floorMask)`
   - 反射贡献：`refl * ssrIntensity * fade * reflectMask`

这样可使反射主要落在“亮且近白、且表面朝上”的区域（即白色地板）。

---

## 6. SSR 原理总结

SSR 的核心思想：

- 在屏幕空间中，用当前像素法线构造反射光线
- 沿反射方向步进，持续投影到屏幕 UV
- 与深度重建的几何位置比较，判断是否命中
- 命中则用命中点屏幕颜色作为反射颜色

优点：

- 无需额外离屏反射相机
- 实现成本低，适合实时后处理

局限：

- 只能反射屏幕内可见内容（屏外信息缺失）
- 对深度精度与法线估计质量敏感
- 边缘与薄物体容易出现断裂/闪烁

本项目已包含基础缓解：

- 命中后二分 refine
- 边缘衰减（`fade`）
- `thickness/stride/maxSteps` 可调

---

## 7. 参数建议（调优方向）

- `ssrMaxSteps`
  - 大：命中率更高，性能更差
- `ssrStride`
  - 小：更精细，性能更差
- `ssrThickness`
  - 大：更易命中，但容易“穿帮/粘连”
- `ssrMaxRayDistance`
  - 大：可见反射更远，误命中和开销上升
- `ssrIntensity`
  - 控制反射叠加强度

可在 `updateSSRUI()` 中实时调整这些参数观察效果。

---

## 8. SSR 数据流（输入 → 计算 → 输出）

### 8.1 输入

- 来自全屏三角形的当前像素 `uv`
- `colorTex`（主场景颜色拷贝）
- `depthTex`（主场景深度拷贝）
- `SceneUBO`
  - `projection / view / invProjection`
  - `cameraPosNear.w`（near）
  - `cameraFarPadding.x`（far）
- `SSRParams`
  - `maxSteps / stride / thickness / maxRayDistance / intensity / debugMode / invResolution`

### 8.2 计算过程

1. **重建位置与法线**
   -  `depthTex * invProjection` 得到当前像素 View 空间位置 `posVS`
   - 采样法线贴图得到世界空间法线，再乘以 view矩阵得到 View空间下的法线 `N`

2. **构造反射射线**
   - 视线方向：`V = normalize(-posVS)`
   - 反射方向：`R = reflect(-V, N)`
   - 起点偏移：`rayOriginVS = posVS + N * bias`（减少自相交）

3. **屏幕空间步进（Ray March）**
   - 沿反射方向 `R` 逐步前进并投影回屏幕 UV，采样 `depthTex` 得到深度
   - 深度和 `scenePos`比较 `dz = scenePos.z - p.z`
   - 若`0 < dz < thickness`，则算作命中（说明靠近摄像机，摄像机看得见这个反射的目的地；`thickness`表示接受在表面后方一点点的命中
   - 命中后做二分 refine 得到更稳定的 `hitUV`：一旦发现当前步满足命中条件（dz 落进阈值），说明真实交点大概率就在上一步和当前步区间内，取中间位置 `mid`，预测是中间位置，计算对应的 View空间的位置`mp`，再用 `mp` 计算 `muv` 采样深度图得到实际表面位置 `mpos`，比较两者 z 差值 mdz = mpos.z - mp.z，来判断交点在区间哪一边，并收缩区间，反复 5 次，收敛到最终 `hitUV`

4. **反射采样与稳定化**
   - 命中且有效（屏幕安全区内、非背景）才采样
   - 在 `hitUV` 做邻域加权采样，得到 `refl`

5. **反射权重计算**
   - 白色/朝上掩码：`reflectMask = saturate(whiteMask * floorMask)`
   - 再乘以：边缘衰减 `fade`、距离衰减 `distanceFade`、角度增强 `fresnel`、全局强度 `intensity`
   - 得到最终反射强度 `reflectStrength`

6. **最终合成**
   - `finalColor = baseColor + refl * reflectStrength`（含可见性增强项）

### 8.3 输出
输出到 `swapchain` 的最终颜色：主场景颜色 + SSR 反射结果

---

## 9. 补充：如何区分“哪些物体参与 SSR”

正常来说不会用“颜色接近白色”这种办法来区分哪些物体会有SSR功能，而是用材质和其他反射方法共同决定。

### 9.1 材质（主流做法）

常见材质参数：

- `roughness`（粗糙度）
- `metallic`（金属度）
- `specular / F0`（镜面反射基线）
- `clearcoat`（清漆层，可选）

SSR 权重一般是连续值而不是开关。比如：

- 低 roughness（光滑） → SSR 权重大
- 高 roughness（粗糙） → SSR 权重小（甚至接近 0）

### 9.2 GBuffer / 材质标记

延迟管线里把反射相关量写进 GBuffer（或单独 RT）：

- 法线 `normal`
- 粗糙度 `roughness`
- 金属度 `metallic`
- 反射掩码 `reflectMask`（可选）

SSR pass 读取这些参数后按像素计算反射贡献，实现“同一物体不同区域反射强度不同”。

### 9.3 按材质类型走不同反射路径

- **镜子 / 高质量水面**：常用 Planar Reflection 或 Ray Tracing，SSR 主要补细节
- **普通光滑材质**：SSR + Reflection Probe（立方体探针）混合
- **粗糙材质**：SSR 占比较低，主要靠 Probe/IBL

### 9.4 不要只用 SSR 而是混合

- SSR 只能反射屏幕内可见内容，屏外物体无法反射
- 屏幕边缘容易缺失
- 遮挡/薄物体区域容易不稳定

所以游戏里一般会做：

- `FinalReflection = SSR * hitConfidence + Probe/Planar/RT * (1 - hitConfidence)`

再结合材质权重做最终能量分配。

---



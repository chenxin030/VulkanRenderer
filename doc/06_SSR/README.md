# SSR（Screen Space Reflection）实现说明

[返回目录](../../README.md)

概述：**基于深度重建 + 屏幕空间步进**：先复制主场景颜色，再通过全屏着色器计算命中并混合反射。

---

## 1. 整体渲染流程

单帧渲染的大致顺序是：

1. **Shadow Pass（深度）**：渲染阴影图（`shadowMap`）
2. **Shadow Lit Pass（场景着色）**：渲染带阴影的场景，同时输出到三个附件：
   - `swapchain`：主场景颜色
   - `ssrNormalData`：世界空间法线（格式 `R16G16B16A16Sfloat`，alpha 通道编码 roughness）
   - `ssrDepthData`：主场景深度（用于 SSR 步进命中检测）
3. **SSR Composite Pass（全屏三角形）**：
   - 先把当前 `swapchain` 颜色拷贝到 `ssrColorData`（作为"反射采样源"）
   - 再执行全屏 SSR 光线步进，并将反射结果混合回 `swapchain`
4. **UI Pass**：叠加 ImGui
5. **Present**

对应文件：
- 主命令录制：`src/ssr/SSRRenderer.cpp` 的 `recordCommandBuffer()`
- SSR 录制：`src/ssr/SSRRenderer.cpp` 的 `recordSSR()`

---

## 2. C++ 侧：SSR用到的对象及作用

### 2.1 主要成员对象

- `MeshBuffer ssrSceneUboResources`: 传 `projection/view/invProjection/camera` 等参数给 SSR shader
- `MeshBuffer ssrParamsUboResources`: 传 SSR 可调参数（步长、厚度、最大步数、强度等）
- `TextureData ssrColorData`: 保存主场景颜色拷贝（用于反射采样）
- `TextureData ssrNormalData`: 世界空间法线（alpha 通道编码 roughness）
- `TextureData ssrDepthData`: 保存主场景深度（用于 SSR 步进命中检测）

- 参数：
  - `int ssrMaxSteps = 85`
  - `float ssrMaxRayDistance = 16.0f`
  - `float ssrThickness = 0.12f`
  - `float ssrStride = 0.05f`
  - `float ssrIntensity = 1.0f`
  - `float defaultRoughness = 0.05f`：SSR 启用阈值（UI 可调）

### 2.2 UBO 结构（`SSRRenderer.cpp`）

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
    - 越大：更容易"命中"；但过大容易穿帮、粘连、出现错误反射。
  - `stride`
    - 每次步进长度（步长）。
    - 越小：命中更细致、边缘更平滑；但循环次数实际更高、性能更差。
  - `maxSteps`
  - `intensity`
    - 反射叠加总强度系数。
    - 直接影响"反射有多明显"。
  - `invResolution`
    - 屏幕分辨率倒数（`1/width, 1/height`）。
    - 用于邻域采样偏移（如法线估计、反射稳化采样）时的"1 像素单位"。
  - `debugMode` ： 调试显示模式开关
    - `0`：正常输出
    - `1`：命中掩码（HitMask）
      - 含义：显示当前像素的 SSR 射线是否命中几何。
      - 颜色语义：命中为红色（`1`），未命中为黑色（`0`）。
      - 用途：快速判断"为什么某区域没有反射"——如果大面积是黑色，说明射线没打到有效屏幕内几何（可能是步长过大、最大步数不足、厚度阈值不合适或目标在屏幕外）。
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
- `binding 1`：`CombinedImageSampler`（深度 `ssrDepthData`）
- `binding 2`：`CombinedImageSampler`（场景颜色拷贝 `ssrColorData`）
- `binding 3`：`CombinedImageSampler`（世界空间法线 `ssrNormalData`，alpha = roughness）
- `binding 4`：`UniformBuffer`（`SSRParams`）

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

> 注意：SSR pass 是"在已完成主场景颜色基础上做后处理混合"，并非单独 GBuffer 管线。

---

## 4. 着色器侧：对象及作用（`shaders/ssr.slang`）

### 4.1 VS 入口 `vertMain`

全屏三角形顶点着色器，直接输出裁剪空间坐标（无模型/视图/投影变换）。

### 4.2 FS 关键函数

- `reconstructViewPos(uv, depth)`
  - 用 `invProjection` 从屏幕坐标重建 View 空间位置
- `sampleNormalVS(uv, outRoughness)`
  - 从法线贴图读取世界空间法线（解码 `xyz`），转 View 空间法线
  - `roughness` 从法线 RT 的 alpha 通道读取
- `projectToUV(viewPos)`
  - View 空间点投影回屏幕 UV
- `raymarchSSR(originVS, dirVS, jitter01, out hitUV, out hitSteps)`
  - 在屏幕空间进行射线步进，检测与深度重建几何是否相交
  - 命中后进行简短二分 refine

### 4.3 FS 主流程 `fragMain`

1. 读取当前像素深度，过滤背景
2. 重建 `posVS`、采样法线 `N`，同时读取 `roughness`
3. 由视线向量 `V` 与法线求反射方向 `R`
4. 调 `raymarchSSR` 获得命中位置 `hitUV`
5. `baseColor = colorTex.Sample(uv)`
6. `refl = hit ? colorTex.Sample(hitUV) : 0`
7. 边缘衰减 `fade`
8. 基于 `roughness` 计算 `roughnessMask`（低粗糙度 → 高反射权重）
9. 按掩码控制反射贡献并叠加到 `baseColor`

输出最终颜色到 `swapchain`。

---

## 5. 反射掩码：基于 Roughness 的工程化方案

采用**材质粗糙度驱动**的方式决定哪些表面参与 SSR 反射：

### 5.1 数据链路

```
ShadowInstanceData (C++)
    ├── model: glm::mat4
    ├── color: glm::vec4
    └── roughness: float     ← 每个实例独立的粗糙度
         ↓
shadow_lit_ssr.slang: InstanceData
    ├── vertex 输出 roughness 到片段
         ↓
shadow_lit_ssr.slang: fragMain
    输出 normalRT = float4(encode(normal), roughness)
         ↓
ssr.slang: sampleNormalVS(uv, outRoughness)
    读取 normalRT.a → roughness
         ↓
ssr.slang: fragMain
    roughnessMask = 1.0 - smoothstep(0.35, 0.50, roughness)
```

### 5.2 粗糙度掩码计算

```slang
// roughness ∈ [0,1]：0=完美镜面（高反射），1=完全粗糙（无反射）
const float roughnessThreshold = 0.35f;
const float smoothing = 0.15f;
float roughnessMask = 1.0f - smoothstep(roughnessThreshold, roughnessThreshold + smoothing, roughness);
```

- `roughness < 0.35` → `roughnessMask ≈ 1`（完全反射）
- `roughness ∈ [0.35, 0.50]` → 平滑过渡
- `roughness > 0.50` → `roughnessMask ≈ 0`（无反射）

### 5.3 场景中各物体的粗糙度设置

| 实例       | 颜色 | Roughness | 反射效果     |
| ---------- | ---- | --------- | ------------ |
| 地板       | 白色 | `0.05`    | 强镜面反射   |
| 奇数立方体 | 红色 | `0.10`    | 较强镜面反射 |
| 偶数立方体 | 蓝色 | `0.50`    | 无/弱反射    |

可通过 UI 的 `DefaultRoughness` 滑块统一调整基准粗糙度。

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
- 5-tap 邻域采样降低闪烁

---

## 7. 参数建议（调优方向）

- `ssrMaxSteps`
  - 大：命中率更高，性能更差
- `ssrStride`
  - 小：更精细，性能更差
- `ssrThickness`
  - 大：更易命中，但容易"穿帮/粘连"
- `ssrMaxRayDistance`
  - 大：可见反射更远，误命中和开销上升
- `ssrIntensity`
  - 控制反射叠加强度
- `defaultRoughness`
  - 调高：更多物体参与反射（低粗糙度阈值降低）
  - 调低：只有光滑表面反射

可在 `updateUIPanel()` 中实时调整这些参数观察效果。

---

## 8. SSR 数据流（输入 → 计算 → 输出）

### 8.1 输入

- 来自全屏三角形的当前像素 `uv`
- `colorTex`（主场景颜色拷贝 `ssrColorData`）
- `depthTex`（主场景深度拷贝 `ssrDepthData`）
- `normalTex`（世界空间法线 `ssrNormalData`，alpha = roughness）
- `SSRSceneUBO`
  - `projection / view / invProjection`
  - `cameraPosNear.w`（near）
  - `cameraFarPadding.x`（far）
- `SSRParams`
  - `maxSteps / stride / thickness / maxRayDistance / intensity / debugMode / invResolution`

### 8.2 计算过程

1. **重建位置与法线**
   -  `depthTex * invProjection` 得到当前像素 View 空间位置 `posVS`
   - 采样法线贴图得到世界空间法线（解码 xyz），转 View 空间得到 `N`
   - 从法线贴图 alpha 通道读取 `roughness`

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
   - 粗糙度掩码：`roughnessMask = 1 - smoothstep(0.35, 0.50, roughness)`
   - 再乘以：边缘衰减 `fade`、距离衰减 `distanceFade`、Fresnel 增强、全局强度 `intensity`
   - 得到最终反射强度 `reflectStrength`

6. **最终合成**
   - `finalColor = baseColor + reflBoosted * reflectStrength + minVisible`（含可见性增强项）

### 8.3 输出
输出到 `swapchain` 的最终颜色：主场景颜色 + SSR 反射结果

---

## 9. 补充：反射方案的工程考量

### 9.1 粗糙度（Roughness）驱动反射掩码

本实现采用 PBR 体系中的 `roughness` 参数作为 SSR 掩码的核心依据：

- 低 `roughness`（光滑）→ SSR 权重大
- 高 `roughness`（粗糙）→ SSR 权重小（接近 0）

这是业界最常见的基础方案，直接利用材质系统已有的参数，无需额外标记位。

### 9.2 为什么不只用白色/法线掩码

demo 级方案（颜色亮度+表面朝上）的问题：

- 无法表达"同一物体不同区域的反射差异"
- 不能与材质系统联动
- 只能区分"是否反射"，不能做平滑过渡

### 9.3 按材质类型走不同反射路径

- **镜子 / 高质量水面**：常用 Planar Reflection 或 Ray Tracing，SSR 主要补细节
- **普通光滑材质**：SSR + Reflection Probe（立方体探针）混合
- **粗糙材质**：SSR 占比较低，主要靠 Probe/IBL

### 9.4 不要只用 SSR 而是混合

- SSR 只能反射屏幕内可见内容，屏外物体无法反射
- 屏幕边缘容易缺失
- 遮挡/薄物体区域容易不稳定

所以游戏里一般会做：

```
FinalReflection = SSR * hitConfidence + Probe/Planar/RT * (1 - hitConfidence)
```

再结合材质权重做最终能量分配。

## SSR 噪点成因与解决方案

SSR 反闪烁噪点主要有以下几个层面的原因：

### 1. 抖动策略不对（最重要）

当前实现对每个像素使用固定的 hash jitter 偏移 ray 起点，只能缓解马赫带（stepping artifacts），但无法实现时间稳定性。噪点会随时间闪烁。

**原因**：同一像素每帧用相同的 jitter seed，导致每次 ray march 跳到相同的错误位置，时间上不平均。

**解决方案**：让抖动的 seed 包含帧号或时间，实现每帧不同的 jitter：
```hlsl
// 当前写法（闪烁）
float jitter01 = hash12(pixelCoord);

// 修复：用帧号做 jitter seed
float jitter01 = hash12(pixelCoord + sceneUbo.frameCount * 17.3);
```

### 2. stride 步长过大

默认 stride = 0.05，在视空间里步长可能过大。ray 在命中表面附近时可能跳过去，在命中边界处产生噪点。

**解决方案**：减小 stride 或在 binary search 阶段使用更小的步长。

### 3. 缺少时序抗锯齿（TAA）

每帧独立计算反射，噪点无法被时间平均平滑。这是 SSR 噪点的根本原因。

**解决方案**：将 SSR 输出到单独的 RT（alpha 通道存 hit confidence），用 TAA 与历史帧混合：
```hlsl
// 1. 每帧用不同 jitter 采样
float jitter01 = hash12(pixelCoord + frameCount * goldenRatio);

// 2. 输出 alpha = hitConfidence（命中置信度）
float4 ssrOut = float4(reflColor, hitConfidence);

// 3. 与上一帧混合
float3 history = texture(ssrHistoryTex, uv).rgb;
float blend = clamp(hitConfidence * 4.0, 0.0, 1.0); // 置信度越高混合越快
float3 result = lerp(history, ssrColor, blend);
```

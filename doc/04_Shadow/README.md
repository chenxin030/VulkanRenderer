# Shadow（4_shadow）

[返回目录](../../README.md)

运行时通过 UI 切换： Hard / PCF / PCSS，以及独立开关三种光源。

## 渲染架构

```
┌──────────────────────────────────────────────────────────┐
│                 ShadowRenderer::render()                  │
│                                                          │
│  1. swapChain.acquireNextImage()                         │
│  2. updateUIFrame() — ImGui 收集本帧输入                │
│  3. updateShadowBuffers(currentFrame)                   │
│     ├─ SceneUBO（相机 VP）                              │
│     ├─ ShadowUBO（光源参数 + 3 个 enable 标志）         │
│     ├─ ShadowParamsUBO（过滤参数）                      │
│     └─ ShadowInstanceData（1 地板 + 8 方块）            │
│  4. recordCommandBuffer(imageIndex)                     │
│     ├─ [Shadow Depth Pass] → shadowDepthPipeline         │
│     │   输出：shadowMap（2048×2048，DepthOnly）         │
│     │   Pipeline: shadowDepthPipeline                   │
│     │   Layout: shadowPipelineLayout                     │
│     │   Descriptor: shadowInstanceBufferResources        │
│     │   Draw: drawIndexed(cubeMesh.indices × 9 instances) │
│     ├─ transition shadowMap → ShaderReadOnlyOptimal     │
│     ├─ [Lit Pass] → shadowLitPipeline                   │
│     │   采样 shadowMap 做阴影判断                        │
│     │   Pipeline: shadowLitPipeline                      │
│     │   Layout: shadowPipelineLayout                     │
│     │   Descriptor: shadowInstanceBufferResources        │
│     │   Draw: drawIndexed(cubeMesh.indices × 9 instances) │
│     └─ [UI Pass] → imgui pipeline                        │
│        recordUICmdBuffer(commandBuffer, currentFrame)     │
│  5. graphicsQueue.submit() + presentQueue.presentKHR()   │
└──────────────────────────────────────────────────────────┘
```

### 同步与 Layout 转换

| 步骤                 | 操作                      | Layout 变化                                      | 屏障                                            |
| -------------------- | ------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| Shadow Depth Pass 前 | `transition_image_layout` | `Undefined → DepthAttachmentOptimal`             | `ALL_COMMANDS → DEPTH_STENCIL_ATTACHMENT_WRITE` |
| Lit Pass 前          | `transition_image_layout` | `DepthAttachmentOptimal → ShaderReadOnlyOptimal` | `DEPTH_STENCIL_ATTACHMENT_WRITE → ShaderRead`   |
| Present 前           | `transition_image_layout` | `ColorAttachmentOptimal → PresentSrcKHR`         | `COLOR_ATTACHMENT_WRITE → BottomOfPipe`         |

## 资源与 Descriptor Set

### Descriptor Set Layout（`shadowDescriptorSetLayout`）

| binding | 类型                   | 用途                               | Stage       |
| ------- | ---------------------- | ---------------------------------- | ----------- |
| 0       | `UniformBuffer`        | `SceneUBO`（相机 VP）              | Vert + Frag |
| 1       | `StorageBuffer`        | `ShadowInstanceData[]`（实例列表） | Vert + Frag |
| 2       | `UniformBuffer`        | `ShadowUBO`（光源参数）            | Vert + Frag |
| 3       | `CombinedImageSampler` | `shadowMap`（2048? 深度图）        | Frag        |
| 4       | `UniformBuffer`        | `ShadowParamsUBO`（过滤参数）      | Frag        |

### UBO / SSBO 规格

| 缓冲区                          | 尺寸                             | 双缓冲 |
| ------------------------------- | -------------------------------- | ------ |
| `sceneUboResources`             | `sizeof(SceneUBO)`               | ?      |
| `shadowUboResources`            | `sizeof(ShadowUBO)`              | ?      |
| `shadowParamsUboResources`      | `sizeof(ShadowParamsUBO)`        | ?      |
| `shadowInstanceBufferResources` | `9 × sizeof(ShadowInstanceData)` | ?      |

### ShadowUBO（`VulkanTypes.h`）

```cpp
struct ShadowUBO {
    glm::mat4 lightViewProj;         // 光源 VP 矩阵（正交）
    glm::mat4 prevViewProj;          // 上一帧 VP（未使用，保留接口）
    glm::vec4 dirLightDirIntensity;   // xyz=方向, w=强度
    glm::vec4 dirLightColor;
    glm::vec4 pointLightPosIntensity; // xyz=位置, w=强度
    glm::vec4 pointLightColor;
    glm::vec4 areaLightPosIntensity;  // xyz=中心, w=强度
    glm::vec4 areaLightColor;
    glm::vec4 areaLightU;             
    glm::vec4 areaLightV;            
    uint32_t dirLightEnabled;        
    uint32_t pointLightEnabled;      
    uint32_t areaLightEnabled;      
};
```

### ShadowParamsUBO

```cpp
struct ShadowParamsUBO {
    int shadowFilterMode;             // 0=Hard, 1=PCF, 2=PCSS
    float pcfRadiusTexels;            // PCF 核半径（texel 单位）
    float pcssLightSizeTexels;        // PCSS 光源尺寸（texel 单位）
    float shadowBiasMin;              // 最小 bias
    glm::vec2 invShadowMapSize;       // 1/2048
    glm::vec2 padding0;
};
```

## 测试场景

场景包含 9 个实例：
- **1 个地板**：位于 y=-2，缩放 `(8.0, 0.25, 8.0)`，灰色
- **8 个方块**：围绕原点均匀分布（半径 2.8），随时间绕 Y 轴旋转，红/蓝交替

光源（方向光）绕场景缓慢旋转，用户可在 UI 中独立开关三种光源来观察各自产生的阴影效果。

## 光照模型（3 种光源）

### Directional light（**投射阴影**）

- 方向与强度：`dirL = normalize(-dirLightDirIntensity.xyz)`
- 漫反射：`baseColor * max(dot(N, L), 0) * dirIntensity * dirColor`
- 高光：`pow(max(dot(N, H), 0), 64) * 0.25 * dirIntensity * dirColor`
- 阴影：`dirShadow = computeShadow(shadowPos, N, dirL)`，最终贡献乘以 `dirShadow`

### Point light（不投射阴影，含距离衰减）

- 位置与强度：`pointPos = pointLightPosIntensity.xyz`，`pointIntensity = pointLightPosIntensity.w`
- 距离衰减：`pointAtt = 1 / (1 + dist^2)`
- 漫反射：`baseColor * max(dot(N, L), 0) * pointIntensity * pointAtt * pointColor`
- 高光：`pow(max(dot(N, H), 0), 64) * 0.20 * pointIntensity * pointAtt * pointColor`

### Area light（矩形面积光，4 点采样近似）

- 光源由中心 `areaLightPosIntensity.xyz` 和两条边向量 `areaLightU/V` 定义
- 以 4 个角点采样（`[-1,-1]`, `[-1,1]`, `[1,-1]`, `[1,1]`）：
  1. 对每个采样点计算 `L`、衰减、漫反射与高光
  2. 将 4 次采样求和并取平均（乘 `0.25`）
- 最终：`areaLight = areaSum * (areaIntensity * 0.25) * areaColor`

## Shadow Map Pass

`shadow_depth.spv`（深度预通道）：
- 输入：所有 9 个实例的 model 矩阵 + 立方体顶点
- 输出：`shadowMapData.textureImage`（2048×2048，D32/D16）
- 变换链：`model * position → worldPos → shadowUbo.lightViewProj → NDC`
- 管线配置：开启 `depthBias`（constant=1.25, slope=1.75），关闭背面剔除（`cullMode=None`）
- 片段着色器为空 pass-through，仅写入深度

## 阴影采样（`computeShadow`，`shadow_lit.fragMain`）

1. **光源 NDC 转 UV**：`shadowPos.xyz / shadowPos.w`，映射到 `uv = proj.xy * 0.5 + 0.5`
2. **深度合法性检查**：超出 `[0,1]` 返回可见度 1（UV 超范围表示不在该光源 Shadow Frustum 内；深度超范围表示超出光源裁剪面）
3. **自适应偏移**：`bias = max(0.0018 * (1 - ndotl), shadowBiasMin)`
4. **滤波模式**：
   - `Hard`（shadowFilterMode=0）：单采样深度比较
   - `PCF`（shadowFilterMode=1）：16 点 Poisson 采样，求平均可见度
   - `PCSS`（shadowFilterMode=2）：
     - 以当前像素 UV 为中心、pcssLightSizeTexels 为半径的范围内，用 16 个 Poisson 点采样 shadow map。采样深度小于当前深度的点被认为是"遮挡体"，求其平均深度 avgBlockerDepth。遮挡体离接收点越近，avgBlockerDepth 越接近 currentDepth，半影应该越窄；反之越远，半影越宽。
     - Penumbra（半影系数） 估计：`float penumbra = (currentDepth - avgBlockerDepth) / max(avgBlockerDepth, 1e-4);`接收点比遮挡体远多少（归一化度量）。这个比值越大，说明接收点和遮挡体之间的间隙越大，半影应该越宽。然后用这个系数去缩放 PCF 的滤波半径 filterRadiusTexels，再用它做最终的 16 点 Poisson 采样得到柔和阴影。 
     - PCF 过滤

## PCF

多点采样求平均，采样多个 offset，对每个 sample 比较 `currentDepth - bias` 与 `closestDepth`，取平均得到可见度

### 公式

- 采样半径（纹理坐标空间）：$radius = pcfRadiusTexels \times invShadowMapSize$ 也就是"多少个 texel 的半径"，换算成 UV 偏移量
- 每个点的判断：$\text{lit}_i = ((currentDepth - bias) \le closestDepth_i)\ ?\ 1:0$ 被挡住记 0，没挡住记 1
  - `currentDepth - bias` ：从阴影接受点往回退一点，如果接受点确实在遮挡体后面，接受点的 depth 明显大于 shadow map depth，减掉 bias 后仍然大，也不会漏判。用一个宽松的判定条件来容忍浮点误差，"假阴影"（acne）就被放过了。
  - 若 `receiverDepth > shadowMapDepth` → 被挡住；但数值误差会让本该相等的深度也被判">"，产生 acne，所以用 `receiverDepth - bias` 去比较，放宽判定，减少误判阴影
  - 如果改成 `currentDepth + bias`，会更更容易判成阴影（更黑），acne 会更严重
  - 而减太多又会导致 peter-panning（阴影与物体脱离）
  - 所以 bias 是在 acne 和 peter-panning 之间折中调参：`bias = max(0.0018 * (1 - ndotl), shadowBiasMin)`，斜面越陡 bias 越大，。
- 平均可见度：$visibility = \frac{1}{16}\sum_i \text{lit}_i$
- 最终返回：$shadow = lerp(0.25, 1.0, visibility)$

### 参数

- `pcfRadiusTexels`
  - PCF 的核半径，单位是"阴影贴图 texel"
  - 影响采样范围，采样半径越大，阴影边缘越软、越模糊
  - 值越大：阴影边缘越软，但更糊、漏光风险更高
  - 值越小：边缘更硬，容易锯齿
- `texelSize = invShadowMapSize`（float2）
  - 阴影贴图尺寸倒数 $(1/width,\ 1/height)$
  - 用于把"texel 单位"转换成"UV 单位"
- `shadowBiasMin`
  - 最小深度偏移（bias）的下限，防止自阴影粉刺（shadow acne）
  - 代码里实际 bias 是：$bias = \max(0.0018 \cdot (1-ndotl),\ shadowBiasMin)$ 即"斜率相关 bias + 最小 bias"组合
- `poisson[16]`
  - 16 个固定采样方向（分布比较均匀）
  - 比规则 4×4 核更不容易出现条纹感

## PCSS

PCF 半径固定 `pcfRadiusTexels`，所以软硬基本固定。PCSS 会先估计"接收点和遮挡体之间的相对距离"，再动态放大滤波半径 `pcssLightSizeTexels`：

- 接收点离遮挡体越远 → 半影更宽 → 阴影更软
- 接收点靠近遮挡体 → 半影更窄 → 阴影更硬

两阶段：

1. **Blocker Search**：在搜索半径 `pcssLightSizeTexels` 内多次采样，把周围深度求和取平均，得到平均遮挡深度 `avgBlockerDepth`
2. **Penumbra 估计 + PCF**：用 `penumbra = (currentDepth - avgBlockerDepth) / max(avgBlockerDepth, 1e-4)` 动态缩放 PCF 半径后做过滤

### 一句话

做了 2 次深度采样：第一次的深度用来计算第二次 PCF 的采样半径要缩放多少。

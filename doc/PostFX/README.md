# PostFX（Level 9：后处理）

[返回目录](../../README.md)

PostFX 基于延迟渲染架构，增加 Bloom（泛光）和通用后处理效果（色调映射、色差、暗角）。

## 1. 整体架构

```
GBuffer Pass ──→ Deferred Lighting Pass ──→ Bloom Extract Pass
                                              │
                                              ├─→ Gaussian Blur Pass (H+V × N)
                                              │
                                              └─→ Bloom Composite Pass ──→ UI Pass
```

6 个 Pass，3 个 Shader：

| Shader 文件 | 用途 |
|---|---|
| `deferred_gbuffer.slang` | GBuffer Pass：几何体 → MRT albedo/normal/material |
| `deferred_lighting.slang` | Deferred Lighting Pass：GBuffer → HDR scene + PBR 光照 |
| `bloom_extract.slang` | Bloom Extract Pass：HDR scene → quarter-res 亮度图 |
| `blur.slang` | Gaussian Blur Pass（H+V 共用同一 shader） |
| `bloom_composite.slang` | Bloom Composite Pass：scene + bloom → LDR + tone mapping + lens FX |

---

## 2. 资源总览

### 2.1 GBuffer（屏幕分辨率，固定）

| Texture | Format | 用途 |
|---|---|---|
| `gbuffer.albedo` | `R16G16B16A16Sfloat` | BaseColor (RGB) |
| `gbuffer.normal` | `R16G16B16A16Sfloat` | View-space 法线 |
| `gbuffer.material` | `R16G16B16A16Sfloat` | metallic/roughness/ao |
| `gbuffer.depth` | `D32Sfloat` | 深度缓冲 |

### 2.2 Post Buffers（ping-pong per frame）

| Texture | Resolution | Format | 用途 |
|---|---|---|---|
| `postBuffers[i].color` | 屏幕分辨率 | `R16G16B16A16Sfloat` | HDR 场景（Deferred 输出，Composite 输入） |
| `postBuffers[i].bloom` | quarter (÷4) | `R16G16B16A16Sfloat` | 亮度提取 → 模糊源 → 模糊结果 |
| `blurBuffers[i].horizontal` | quarter (÷4) | `R16G16B16A16Sfloat` | 水平模糊中间结果 |
| `blurBuffers[i].vertical` | quarter (÷4) | `R16G16B16A16Sfloat` | 垂直模糊最终结果 |

### 2.3 UBO / SSBO

| Buffer | Type | 说明 |
|---|---|---|
| `sceneUboResources` | Uniform | camera VP + invProjection + invView + camPos |
| `lightUboResources` | Uniform | 4 个点光源（位置 + 颜色强度） |
| `deferredSettingsUboResources` | Uniform | Deferred Lighting 专用参数（ambient/exposure/gamma/lightScale + debugView） |
| `postFxSettingsUboResources` | Uniform | PostFX 专用参数（exposure/gamma/bloom/threshold/intensity + tone mapping + lens effects） |
| `instanceBufferResources` | Storage | 49 个球体的 model 矩阵 + PBR 材质（7×7 metallic/roughness 渐变网格） |

---

## 3. 初始化流程

```
prepareResource()
 ├─ generateSphere(sphereMesh) + createVertexBuffer/createIndexBuffer
 ├─ createDeferredBuffers()         创建 4 个 UBO/SSBO
 │
 ├─ createGBufferDescriptorSetLayout / Pool / Resources / Pipeline / DescriptorSets
 │  └─ GBuffer: albedo/normal/material (R16G16B16A16Sfloat MRT) + depth (D32Sfloat)
 │
 ├─ createDeferredDescriptorSetLayout / Pool / Pipeline / DescriptorSets
 │  └─ Deferred Lighting: 7 个 binding（4×gbuffer sampler + 3×UBO）
 │
 ├─ createPostDescriptorSetLayout / Pool / Pipelines
 │  ├─ postPipeline        → postfx.slang（未使用）
 │  ├─ bloomExtractPipeline → bloom_extract.slang
 │  ├─ blurHPipeline/blurVPipeline → blur.slang（PushConst 区分 H/V）
 │  └─ bloomCompositePipeline → bloom_composite.slang
 │
 ├─ createPostBuffers()             创建 postBuffers + blurBuffers（quarter-res）
 ├─ createPostDescriptorSets()       分配后处理 descriptor sets
 └─ initUI()                        ImGui 字体纹理 + uiPipeline
```

### Quarter-res 尺寸计算

```cpp
uint32_t bw = (swapChainExtent.width + 3) / 4;  // 向下取整 + 1
uint32_t bh = (swapChainExtent.height + 3) / 4;
```

所有 Post Buffer（含 blurBuffers）均为 `bw × bh`，这样 Bloom Extract Pass 通过 viewport 缩放直接实现下采样。

---

## 4. 渲染循环

每帧 CPU 数据准备 → Descriptor 更新 → 命令录制（6 Pass） → submit + present。

### 4.1 CPU 数据准备

**`updateDeferredBuffers(frameIndex)`** — 4 类 CPU→GPU 写入：

**SceneUBO**：
```cpp
sceneUbo.projection = perspective(Y flip=-1)
sceneUbo.view = camera.GetViewMatrix()
sceneUbo.invProjection = inverse(projection)  // Deferred 重建世界坐标用
sceneUbo.invView = inverse(view)
sceneUbo.camPos = camera.Position
```

**Instance Data**（7×7 metallic/roughness 渐变）：
```cpp
// x ∈ [-3,3], y ∈ [-3,3]
metallic = (x + 3) / 6.0f       // 0.0 → 1.0
roughness = clamp((y + 3) / 6.0f, 0.04, 1.0f)
baseColor = gold (1.0, 0.86, 0.57)
```

**LightUBO**（4 个点光源，UI 控制动画）：
```cpp
lights[0]: 静止白色高强度 (400×scale)
lights[1]: 静止暖白低强度 (80×scale)
lights[2]: sin运动 (180×scale)
lights[3]: cos运动 (180×scale)
```

### 4.2 Descriptor 更新（`render()` 中，commandBuffer.begin() 之前）

所有 Descriptor 更新必须在 `begin()` 之前完成：

```
① Deferred Lighting DescriptorSet
   updateDescriptorSets: lightUboResources.descriptorSets[frame]
   b0→albedo, b1→normal, b2→material, b3→depth, b4→sceneUbo, b5→lightUbo, b6→deferredSettingsUbo

② Bloom Extract DescriptorSet
   memcpy PostFXSettingsUBO → postFxSettingsUboMapped
   updateDescriptorSets: bloomExtractDescriptorSets[frame]
   b0→postBuffers[frame].color, b1→postFxSettingsUbo

③ Blur H DescriptorSet
   updateDescriptorSets: blurHDescriptorSets[frame]
   b0→postBuffers[frame].bloom

④ Blur V DescriptorSet
   updateDescriptorSets: blurDescriptorSets[frame]
   b0→blurBuffers[frame].horizontal

⑤ Bloom Composite DescriptorSet
   memcpy PostFXSettingsUBO → postFxSettingsUboMapped（覆盖 Deferred settings）
   updateDescriptorSets: bloomCompositeDescriptorSets[frame]
   b0→postBuffers[frame].color, b1→postBuffers[frame].bloom, b2→postFxSettingsUbo
```

---

## 5. 各 Pass 详解

### 5.1 GBuffer Pass — `recordGBufferPass`

**对应 Shader**：`deferred_gbuffer.slang` — `vertMain` + `fragMain`

**Layout 转换**：
```
gbuffer.albedo/normal/material  Undefined → ColorAttachmentOptimal
gbuffer.depth                  Undefined → DepthAttachmentOptimal
postBuffers.color             Undefined → ColorAttachmentOptimal
```

**绘制**：
```
Pipeline: gbufferPipeline
DescriptorSet: instanceBufferResources.descriptorSets[frame]
  b0 → sceneUbo (camera VP)
  b1 → instanceBuffer (49× DeferredInstanceData SSBO)
drawIndexed(sphereMesh.indexCount, 49)  // instanced
```

**MRT 输出**：

| Attachment | 内容 |
|---|---|
| albedo (RGB16F) | 各球体 baseColor（金色） |
| normal (RGB16F) | view-space 法线 × 0.5 + 0.5 |
| material (RGB16F) | metallic/roughness/ao |
| depth (D32F) | 深度值 |

---

### 5.2 Deferred Lighting Pass — `recordDeferredPass`

**对应 Shader**：`deferred_lighting.slang` — `vertMain`（全屏三角）+ `fragMain`（PBR 光照）

**Layout 转换**：
```
gbuffer.*/depth   ColorAttachmentOptimal → ShaderReadOnlyOptimal
```

**绘制**：
```
Pipeline: deferredPipeline
DescriptorSet: lightUboResources.descriptorSets[frame]
  b0→albedo, b1→normal, b2→material, b3→depth,
  b4→sceneUbo, b5→lightUbo, b6→settingsUbo(DeferredSettings)
draw(3, 1)  // 全屏三角形
```

**Shader 逻辑**：

1. 采样 GBuffer：albedo、normal×2-1、material (metallic/roughness/ao)
2. 采样 depth，通过 `invProjection + invView` 重建世界坐标
3. PBR 光照循环（4 光源）：GGX 法线分布 + Schlick-Smith 几何遮蔽 + Fresnel
4. 硬编码 Reinhard tone mapping + gamma 矫正（不经过 UI 参数）
5. `debugView` 可选显示 albedo/normal/material/depth

**输出**：`postBuffers[frame].color`（HDR 场景，未做 UI 控制的 tone mapping）

---

### 5.3 Bloom Extract Pass — `recordBloomExtractPass`

**对应 Shader**：`bloom_extract.slang` — `vertMain` + `fragMain`

**Layout 转换**：
```
postBuffers.color  ColorAttachmentOptimal → ShaderReadOnlyOptimal
postBuffers.bloom Undefined → ColorAttachmentOptimal
```

**绘制**：
```
Pipeline: bloomExtractPipeline
Viewport: (0,0) → (bw, bh)   bw=width/4, bh=height/4
DescriptorSet: bloomExtractDescriptorSets[frame]
  b0→postBuffers[frame].color, b1→settingsUbo(PostFX)
draw(3, 1)
```

**Shader 逻辑**：

1. 采样 HDR scene × exposure
2. 计算 luminance：`dot(c, [0.2126, 0.7152, 0.0722])`
3. soft knee 过渡：`t = saturate((lum - (threshold - knee)) / (2×knee))`
4. 权重 = `max(lum - threshold, 0) × t²`
5. 输出 `color × (weight/lum) × bloomIntensity`

Viewport 下采样直接实现 quarter-res，减少模糊计算量。

**输出**：`postBuffers[frame].bloom`（1/4 分辨率亮度图）

---

### 5.4 Gaussian Blur Pass — `recordBlurPasses`

**对应 Shader**：`blur.slang` — `vertMain` + `fragMain`（blurHPipeline 和 blurVPipeline 共用）

**参数**：`blurIterations = round(bloomRadius)`（UI 控制 1~8 次）

**Ping-pong 循环**：

```
srcBloom = postBuffers[frame].bloom
for iter in [0, blurIterations):
    # H-blur: src → blurH
    Bind: blurHDescriptorSets[frame] → src (bloom 或上一轮 blurV)
    PushConst: texelSize = (4/bw, 0)
    draw(3,1) → blurBuffers[frame].horizontal

    # V-blur: blurH → blurV
    Bind: blurDescriptorSets[frame] → blurH
    PushConst: texelSize = (0, 4/bh)
    draw(3,1) → blurBuffers[frame].vertical

    # Copy blurV → bloom (下一轮输入或最终 bloom)
    copyImage(blurV → bloom)
```

同一 shader 两个 pipeline（H/V），通过 PushConstant `texelSize` 控制方向：
- H：`texelSize = (4.0/bw, 0.0)` — 水平步进
- V：`texelSize = (0.0, 4.0/bh)` — 垂直步进

9-tap Gaussian weights（sigma=2.0）：
```
[0.0162, 0.054, 0.122, 0.195, 0.227, 0.195, 0.122, 0.054, 0.016]
```

**输出**：`postBuffers[frame].bloom`（模糊后的 bloom，quarter-res）

---

### 5.5 Bloom Composite Pass — `recordBloomCompositePass`

**对应 Shader**：`bloom_composite.slang` — `vertMain` + `fragMain`

**Layout 转换**：
```
postBuffers.color  ColorAttachmentOptimal → ShaderReadOnlyOptimal
postBuffers.bloom TransferSrcOptimal → ShaderReadOnlyOptimal
swapChainImages   PresentSrcKHR → ColorAttachmentOptimal
```

**绘制**：
```
Pipeline: bloomCompositePipeline
Viewport: (0,0) → (width, height)
DescriptorSet: bloomCompositeDescriptorSets[frame]
  b0→postBuffers[frame].color, b1→postBuffers[frame].bloom, b2→settingsUbo(PostFX)
draw(3, 1)
```

**Shader 逻辑**：

1. 采样 HDR scene + bloom
2. 如果 `bloomEnabled`：scene + bloom × bloomIntensity
3. **Chromatic aberration**（可选）：对 bloom 纹理 R/G/B 通道做不同 UV 偏移
4. **Tone mapping**：
   - Exposure × color
   - Reinhard 或 ACES Filmic
   - Gamma 矫正
5. **Vignette**（可选）：`vig = 1 - smoothstep(0.3, 0.9, dist)`, `color × lerp(1-intensity, 1, vig)`

所有 UI 控制的后处理效果（tone mapping / chromatic aberration / vignette / bloom 叠加）都在此 Pass 完成。

**输出**：`swapChainImages[frame]`（LDR 8-bit，准备显示）

---

### 5.6 UI Pass — `recordUIPass`

**对应 Shader**：`imgui.slang` — `vertMain` + `fragMain`

**Layout 转换**：
```
swapChainImages   ColorAttachmentOptimal → ColorAttachmentOptimal (loadOp=Load)
swapChainImages   ColorAttachmentOptimal → PresentSrcKHR
```

**绘制**：
```
Pipeline: uiPipeline
VertexBuffer: uiFrameBuffers[currentFrame].vertexBuffer (ImDrawVert)
IndexBuffer:  uiFrameBuffers[currentFrame].indexBuffer
DescriptorSet: uiDescriptorSets[0] → uiFontTexture (RGBA8)
PushConst: scale=(2/displayW, 2/displayH), translate=(-1,-1)
drawIndexed(ElemCount)
```

`loadOp=Load` 保留 composite 内容，ImGui 叠加其上。

**输出**：`swapChainImages[frame]`（LDR + UI → Present）

---

## 6. UI 控制参数

**Tone Mapping**：`toneMappingMode`（0=Reinhard, 1=ACES Filmic）、`exposure`、`gamma`

**Bloom**：`bloomEnabled`、`bloomThreshold`（亮度提取阈值）、`bloomIntensity`（叠加强度）、`bloomRadius`（控制模糊迭代次数）

**Lens Effects**：`chromaticAberrationEnabled`、`chromaticAberrationStrength`、`vignetteEnabled`、`vignetteIntensity`

**Scene**：`animateLights`、`lightIntensityScale`、`debugView`（0=正常 / 1=albedo / 2=normal / 3=material / 4=depth）

---

## 7. 提交与呈现

```cpp
submitInfo {
    waitSemaphore: presentCompleteSemaphores[frame]   // ImageAvailable
    waitStage: ColorAttachmentOutput
    signalSemaphore: renderFinishedSemaphores[imageIndex]
}
presentInfo {
    waitSemaphore: renderFinishedSemaphores[imageIndex]
    swapchain: swapChain
    imageIndex
}
```

---

## 8. 数据流速查

```
GBuffer Pass
  Input:  sphereMesh + sceneUbo + instanceBuffer(49×)
  Output: gbuffer.{albedo/normal/material/depth}

Deferred Pass
  Input:  gbuffer.{albedo/normal/material/depth} + sceneUbo + lightUbo + deferredSettingsUbo
  Output: postBuffers[frame].color (HDR scene)

Bloom Extract Pass
  Input:  postBuffers[frame].color + postFxSettingsUbo
  Output: postBuffers[frame].bloom (1/4-res bright-pass)

Gaussian Blur (N iterations × H+V)
  Input:  postBuffers[frame].bloom
  Output: postBuffers[frame].bloom (blurred)

Bloom Composite Pass
  Input:  postBuffers[frame].color + postBuffers[frame].bloom + postFxSettingsUbo
  Output: swapChainImages[frame] (LDR)

UI Pass
  Input:  uiFontTexture + ImDrawVert/Idx
  Output: swapChainImages[frame] (LDR + UI)
```

---

## 9. 已知问题

### 9.1 `postfx.slang` 已移除（已修复）

`postfx.slang`（tone mapping + chromatic aberration + vignette）对应的 `postPipeline` 和 `postDescriptorSets` 已在代码中移除。所有后处理效果统一由 `bloom_composite.slang` 实现。

### 9.2 Settings UBO 类型已分离（已修复）

`settingsUboResources` 已拆分为：
- `deferredSettingsUboResources`：仅供 Deferred Lighting Pass 使用
- `postFxSettingsUboResources`：仅供 Bloom Extract / Composite Pass 使用

两者各自独立，无覆盖语义混淆。

### 9.3 Blur Layout Tracking Bug 已修复

Blur 循环末尾 `copyImage` 后，`postBuffers[frameIndex].bloomLayout` 现在正确设为 `TransferDstOptimal`（bloom 是 copy 目标），与 `recordBloomCompositePass` 中的 transition 从 `TransferDstOptimal` → `ShaderReadOnlyOptimal` 语义一致。

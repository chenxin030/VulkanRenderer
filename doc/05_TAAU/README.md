# TAAU（Temporal Anti-Aliasing Upsampling，`5_taau`）完整流程说明

[返回目录](../../README.md)

低分辨率渲染 + 运动矢量重投影 + 历史帧自适应融合。

---

## 1. 实现的功能

### 1.1 方向光照明

场景包含一个带 PCSS 阴影的方向光

| 光源   | 强度 | 特点                            |
| ------ | ---- | ------------------------------- |
| 方向光 | 0.5  | 带 PCSS 阴影（Poisson-16 采样） |

### 1.2 动态阴影（PCSS）

`shadow_depth.slang` 渲染 2048×2048 深度贴图；`shadow_lit.slang` 以 PCSS 模式采样（Poisson-16，搜索 Blocker → 估算半影 → 自适应滤波半径）：

### 1.3 低分辨率主渲染 + MRT

主渲染以 `taauRenderScale=0.85`（可调）在低分辨率下进行，MRT 同时输出：

| Attachment | 内容                                                           |
| ---------- | -------------------------------------------------------------- |
| Color0     | 光照 + 阴影颜色                                                |
| Color1     | 速度矢量（当前帧→上一帧 NDC 差，由 `prevViewProj` 重投影计算） |
| Depth      | 深度缓冲                                                       |

**为什么低分辨率渲染**：GPU 渲染开销最大的阶段是片元着色器。以 1920×1080、85% 缩放为例，片元数从 207 万降至约 150 万（**减少约 28%**），节省的算力用于后续 TAAU 上采样 pass。低分辨率的细节损失由时域融合弥补：历史帧信息填补单帧上采样的模糊，Halton jitter 提升等效采样率。

### 1.4 TAAU 时域融合

`taau_resolve.slang` 将低分辨率结果上采样至全分辨率，核心算法：

- **Halton Jitter**：CPU 每帧注入 1/4~1/2 像素偏移，提升边缘等效采样率
- **3×3 邻域 Clamp**：扩展包络抑制鬼影
- **亮度差拒绝**：颜色不一致时降低历史权重
- **自适应权重**：速度稳定性 × 颜色稳定性 × 边缘稳定性
- **运动自适应锐化**：高速/边缘/不稳定区域自动减弱锐化
- **History Ping-Pong**：双缓冲轮换，持续累积历史信息

---

## 2. 各个 Shader 的作用

  - `shadow_depth.slang` — Shadow Pass（仅深度）
  - `shadow_lit.slang` — 主渲染（光照 + 阴影，MRT：颜色 + 速度）
  - `taau_resolve.slang` — TAAU 时域融合 + 上采样（全屏 pass）
  - `imgui.slang` — ImGui 文字绘制

---

## 3. 初始化阶段（一次性资源）

### 3.1 geometry
- `cubeMesh`：立方体顶点/索引缓冲（同时用于地面和立方体）
- `instanceCount = 9`（1 地面 + 8 立方体）

### 3.2 Shadow Pass 资源

| 资源                  | 类型                        | 说明                               |
| --------------------- | --------------------------- | ---------------------------------- |
| `shadowMapData`       | Image + ImageView + Sampler | 2048×2048，线性采样，ClampToBorder |
| `shadowDepthPipeline` | Graphics Pipeline           | 只有深度，渲染到 shadowMap         |
| `shadowLitPipeline`   | Graphics Pipeline           | MRT：2 个 color attachment + depth |

**`shadowLitPipeline` MRT 格式**（对应 `shadow_lit.slang` 当前只输出 `SV_TARGET0`）：

| Attachment | 格式                   | 用途             |
| ---------- | ---------------------- | ---------------- |
| Color0     | `swapChainImageFormat` | 光照颜色         |
| Color1     | `R16G16Sfloat`         | 速度（velocity） |
| Depth      | `findDepthFormat()`    | 深度             |

**`shadowDescriptorSetLayout`**（binding 0~4）：

| Binding | 类型                 | 资源                                             |
| ------- | -------------------- | ------------------------------------------------ |
| 0       | UniformBuffer        | `sceneUboResources`（VP 矩阵）                   |
| 1       | StorageBuffer        | `shadowInstanceBufferResources`（instance list） |
| 2       | UniformBuffer        | `shadowUboResources`（光源参数）                 |
| 3       | CombinedImageSampler | `shadowMapData`（阴影贴图）                      |
| 4       | UniformBuffer        | `shadowParamsUboResources`（滤波参数）           |

### 3.3 TAAU 资源

| 资源                      | 尺寸                                | 格式                   | 用途                 |
| ------------------------- | ----------------------------------- | ---------------------- | -------------------- |
| `taauInputColorData`      | `swapChainExtent × taauRenderScale` | `swapChainImageFormat` | 当前帧低分辨率颜色   |
| `taauVelocityData`        | 同上                                | `R16G16Sfloat`         | 当前帧低分辨率速度   |
| `taauDepthData`           | 同上                                | depth format           | 当前帧低分辨率深度   |
| `taauHistoryColorData[2]` | `swapChainExtent`（全分辨率）       | `swapChainImageFormat` | 历史帧颜色 ping-pong |

**`taauDescriptorSetLayout`**（binding 0~4）：

| Binding | 类型                 | 资源                                                    |
| ------- | -------------------- | ------------------------------------------------------- |
| 0       | CombinedImageSampler | `taauInputColorData`（当前低分辨率颜色）                |
| 1       | CombinedImageSampler | `taauHistoryColorData[historyRead]`（历史全分辨率颜色） |
| 2       | CombinedImageSampler | `taauVelocityData`（当前速度）                          |
| 3       | CombinedImageSampler | `taauDepthData`（当前深度）                             |
| 4       | UniformBuffer        | `taauParamsUboResources`（TAAU 参数）                   |

---

## 4. 每帧 CPU 数据准备

### 4.1 `updateTAAUBuffers(currentFrame)` — Jitter 与参数

**Halton 序列抖动注入**（周期 8，base 2 和 3）：

```cpp
sampleIndex = (taauFrameCounter % 8) + 1
hx = halton(sampleIndex, 2) - 0.5   // [0,1) → [-0.5, 0.5)
hy = halton(sampleIndex, 3) - 0.5
taauJitterCurrent = vec2(hx/fullWidth, hy/fullHeight) * 0.45
taauFrameCounter++
```

结果每帧偏移量约为 1/4~1/2 像素，用于提升边缘等效采样率。

### 4.2 `updateShadowBuffers(frameIndex)` — 场景 UBO + 实例数据

**SceneUBO（CPU→GPU）**：

```cpp
sceneUbo.projection = glm::perspective(fov=45°, aspect, near=0.1, far=100)
sceneUbo.view = camera.GetViewMatrix()
sceneUbo.camPos = camera.Position

// Jitter 注入到投影矩阵
sceneUbo.projection[2][0] += taauJitterCurrent.x * 2.0
sceneUbo.projection[2][1] += taauJitterCurrent.y * 2.0
sceneUbo.projection[1][1] *= -1
```

**ShadowUBO（CPU→GPU）**：

- `lightViewProj`：`ortho(-7.0f, 7.0f, -7.0f, 7.0f, 0.1f, 30.0f) × lookAt(lightPos, target)`（平行光阴影矩阵）
- `prevViewProj`：`taauPrevViewProj`（上一帧 VP，用于 velocity 计算）
- 光源参数：方向光（`dirLightIntensity=0.5`）

**ShadowParamsUBO（CPU→GPU）**：

- `shadowFilterMode = 2`（PCSS）
- `pcssLightSizeTexels = 25.0`

---

## 5. 每帧命令录制主流程

`render()` → `recordCommandBuffer(imageIndex)` → 录制到 `commandBuffers[currentFrame]`：

```
CPU: waitFence → acquireNextImage → updateTAAUBuffers → updateShadowBuffers
     ↓
recordCommandBuffer(imageIndex):
 1. transition swapChain/depth/shadowMap → ColorAttachmentOptimal/DepthAttachmentOptimal
 2. Shadow Pass（仅深度）→ shadowMap
 3. 主渲染 Pass（低分辨率 MRT）→ taauInputColor + taauVelocity + taauDepth
 4. recordTAAU(imageIndex)
 5. UI Pass → swapChain
 6. transition swapChain → PresentSrcKHR
 7. submit(wait=presentSem, signal=renderSem) → present()
```

---

## 6. Pass 1：Shadow Pass（阴影深度）

- **Pipeline**：`shadowDepthPipeline`
- **目标**：`shadowMapData`（2048×2048）
- **内容**：9 个实例（地面 + 8 立方体）渲染到深度缓冲
- **Layout 变化**：`Undefined → DepthAttachmentOptimal → ShaderReadOnlyOptimal`

---

## 7. Pass 2：主渲染 Color Pass（低分辨率 MRT）

### 6.1 渲染目标（低分辨率）

```
taauExtent = swapChainExtent × taauRenderScale
           // 例如 1920×1080 × 0.85 → 1632×918
```

MRT 附件（同时渲染）：

1. `taauInputColorData`（Color0）：光照 + 阴影颜色
2. `taauVelocityData`（Color1）：速度矢量（`shadow_lit.slang` 输出当前帧→上一帧 NDC 差）
3. `taauDepthData`（Depth）：当前帧低分辨率深度

### 6.2 `shadow_lit.slang` 光照模型

每个像素计算：

- **环境光**：`baseColor × 0.06`
- **方向光**（带 PCSS 阴影）：`dirShadow × (diffuse + specular) × 0.5`

**Shadow Computation**（PCSS）：
- Poisson-16 PCSS（搜索遮挡物 → 估算半影 → 自适应滤波半径）

---

## 8. Pass 3：`recordTAAU` — 时域融合 + 上采样

### 7.1 分支：`taauEnabled == false`

跳过 TAAU，直接线性放大：

```
blitImage(taauInputColorData → swapChain[imageIndex])
// 低分辨率 → 全分辨率，使用 vk::Filter::eLinear
```

### 7.2 分支：`taauEnabled == true`（主流程）

**Ping-Pong 索引**：

```cpp
historyRead  = taauHistoryReadIndex
historyWrite = (historyRead + 1) % 2
```

**历史初始化**（`taauHistoryValid == false`，仅第一帧）：

```
blitImage(taauInputColorData → taauHistoryColorData[historyRead])
// 低分辨率 → 全分辨率，填充历史缓冲
```

**主 Resolve Pass（全屏，输出到 swapchain）**：

- Pipeline：`taauPipeline`（单 color attachment，全分辨率）
- Descriptor sets：`taauDescriptorSets[currentFrame]`（含历史读索引）
- `draw(3, 1, 0, 0)` — 全屏三角形（`SV_VertexID` 生成顶点）
- 结束后 `taauHistoryReadIndex = historyWrite`，`taauHistoryValid = true`

**History 更新**：

```
blitImage(swapChain[imageIndex] → taauHistoryColorData[historyWrite])
// 全分辨率 → 历史写缓冲，供下帧使用
```

---

## 9. Fragment Shader 细节：`taau_resolve.slang`

### 8.1 全屏三角形顶点着色器

使用 `SV_VertexID` 生成三个覆盖全屏的顶点，无需顶点缓冲：

| vertexId | pos      | uv     |
| -------- | -------- | ------ |
| 0        | (-1, 1)  | (0, 1) |
| 1        | (-1, -3) | (0, 0) |
| 2        | (3, 1)   | (1, 1) |

### 8.2 时域融合算法

0. **Jitter 前提**：CPU 每帧向投影矩阵注入 Halton 序列抖动，使低分辨率渲染的采样相位跨帧不同。

1. **历史重投影**：
   ```cpp
   historyUV = clamp(uv - velocity * velocityScale, 0, 1)
   historyColor = historyColorTex.Sample(historyUV)
   ```

2. **3×3 邻域包络**：
   - 采样中心 + 8 邻域低分辨率颜色
   - 计算 `neighMin / neighMax`，扩展 `±20% × historyClampGamma`
   - 将历史颜色 clamp 到扩展包络，抑制错误历史（鬼影）

3. **亮度差拒绝（无深度历史）**：
   - `lumaDiff = |lumaCurrent - lumaHistory|`
   - `rejectThreshold` 随稳定性动态变化
   - `colorReject = step(rejectThreshold, lumaDiff)`
   - 历史权重乘 `(1 - colorReject)`

4. **自适应历史权重（三因子乘积）**：
   | 因子              | 含义       | 计算                                         |
   | ----------------- | ---------- | -------------------------------------------- |
   | `motionStability` | 速度稳定性 | `saturate(1 -                                | velocity | × 5)` |
   | `colorStability`  | 颜色稳定性 | `saturate(1 - lumaDiff × 5 × reactiveClamp)` |
   | `edgeStability`   | 边缘稳定性 | `saturate(1 - edgeStrength × 0.9)`           |
   - `stability = motionStability × colorStability × edgeStability`
   - `historyWeight = lerp(0.16, blendFactor, stability)` — 静态高权重，动态低权重
   - `staticBoost`：静止区域额外提升 `+0.14`，上限 `0.96`
   - `antiFlicker`：最终颜色再对 `currentColor` 做一次 `lerp`，减少闪烁

5. **运动自适应锐化**（三抑制门控）：
   ```cpp
   sharpenGate = (1 - motionSuppress) × (1 - edgeSuppress) × (1 - unstableSuppress)
   sharpenAmount = lerp(0.005, 0.065, sharpenGate)
   // 高速/边缘/不稳定区域自动降低锐化
   ```

---

## 10. Shader 细节

### 9.1 `shadow_lit.slang`（主渲染 MRT）

顶点阶段：
- `sceneUbo`（binding 0）：当前帧 VP（含 jitter）
- `instanceBuffer`（binding 1）：实例列表（9 个）
- `shadowUbo`（binding 2）：光源参数 + `prevViewProj`（用于 velocity 计算）

片元阶段：
- `shadowMap`（binding 3）：阴影贴图采样
- `shadowParams`（binding 4）：滤波参数

> 注意：`shadow_lit.slang` 输出 `SV_TARGET0`（颜色）+ `SV_TARGET1`（velocity：当前帧→上一帧 NDC 差），`taau_resolve.slang` 据此做历史重投影。

### 9.2 `taau_resolve.slang`（时域融合）

| Binding | 资源              | 描述                                                           |
| ------- | ----------------- | -------------------------------------------------------------- |
| 0       | `inputColorTex`   | 低分辨率当前颜色                                               |
| 1       | `historyColorTex` | 全分辨率历史颜色                                               |
| 2       | `velocityTex`     | 低分辨率速度（由 `shadow_lit.slang` 输出当前帧→上一帧 NDC 差） |
| 3       | `depthTex`        | 低分辨率深度（预留）                                           |
| 4       | `TAAUParams` UBO  | 参数集                                                         |

---

## 11. 参数说明（`TAAUParams` / `TAAUParamsUBO`）

| 参数                     | 默认值 | 范围      | 效果                                   |
| ------------------------ | ------ | --------- | -------------------------------------- |
| `blendFactor`            | 0.90   | 0.20~0.98 | 历史基础权重：高=更稳/更模糊           |
| `reactiveClamp`          | 0.55   | 0.20~1.20 | 亮度差敏感度：高=变化区降历史更激进    |
| `antiFlicker`            | 0.88   | 0.00~1.00 | 抗闪烁 lerp：高=更稳/更软              |
| `velocityScale`          | 1.00   | 0.20~2.50 | 速度幅值缩放                           |
| `historyClampGamma`      | 1.15   | 0.50~2.50 | 邻域包络扩展幅度：高=宽松/保留细节     |
| `historyRejectThreshold` | 0.22   | 0.01~0.50 | 亮度差拒绝阈值：高=更保护历史/可能鬼影 |

---

## 12. UI 调试项（`TAAU Debug`）

```
┌──────────────────────────────────┐
│ TAAU Debug                       │
│ TAA stage-3 (jitter + rejection) │
│ • Current + reprojected history  │
│ • Halton jitter enabled          │
│ • Neighborhood clamp + rejection  │
│──────────────────────────────────│
│ [x] Enable TAA Resolve           │
│ BlendFactor     [━━━━━●━━] 0.90  │
│ ReactiveClamp    [━━━●━━━━] 0.55  │
│ AntiFlicker      [━━━━━●━━] 0.88  │
│ VelocityScale    [━━━━━●━━] 1.00  │
│ ClampGamma       [━━━●━━━━] 1.15  │
│ RejectThreshold  [━━━━━●━━] 0.22  │
│ RenderScale      [━━━━●━━━] 0.85  │
│ [ ] FreezeHistory                │
│ [Reset History]                  │
└──────────────────────────────────┘
```

- **Enable TAA Resolve**：开关时域融合
- **RenderScale** 滑动：重建 TAAU 资源并使历史失效
- **FreezeHistory**：冻结立方体 #1 的动画，测试静止区域稳定性
- **Reset History**：仅使历史失效（不重建资源）

调参建议顺序：

1. 固定 `RenderScale`（例如 0.85）
2. 先调 `blendFactor` / `historyRejectThreshold`（基础稳定性）
3. 再调 `reactiveClamp` / `historyClampGamma`（拒绝策略）
4. 最后微调 `antiFlicker`

---

## 13. SwapChain 重建

`recreateSwapChain()` 覆盖 `VulkanBase::recreateSwapChain()`，额外重建所有 TAAU 尺寸相关资源：

- 清空 `taauInputColorData` / `taauVelocityData` / `taauDepthData`
- 清空 `taauHistoryColorData[2]`
- 重建 descriptor pool + descriptor sets
- `taauHistoryValid = false`（重新初始化历史）

---

## 14. 总时序

```
┌──────────────────────────────────────────────────────────┐
│ T=0: CPU — updateTAAUBuffers()                            │
│        halton jitter + taauParams → GPU                    │
│        updateShadowBuffers():                             │
│          sceneUbo (投影+jitter) → GPU                    │
│          shadowUbo (光源+prevVP) → GPU                    │
│          instanceData (9个) → GPU                         │
│        taauPrevViewProj = currentViewProj                 │
│                                                          │
│ T=1: CPU — recordCommandBuffer()                          │
│   cmdBuffer: shadowDepthPipeline → shadowMap (2048×2048)  │
│   cmdBuffer: shadowLitPipeline → MRT (1632×918)            │
│   cmdBuffer: taauPipeline → swapChain (1920×1080)          │
│   cmdBuffer: UI pipeline → swapChain                       │
│                                                          │
│ T=2: CPU — submit(presentSem) → presentQueue.present()   │
└──────────────────────────────────────────────────────────┘
```

---

## 15. 数据读写关系速查

| 方向        | 资源                            | 操作                            |
| ----------- | ------------------------------- | ------------------------------- |
| **CPU→GPU** | `sceneUboResources`             | 每帧 `memcpy`（VP+jitter）      |
| **CPU→GPU** | `shadowUboResources`            | 每帧 `memcpy`（光源+prevVP）    |
| **CPU→GPU** | `shadowInstanceBufferResources` | 每帧 `memcpy`（9 实例）         |
| **CPU→GPU** | `shadowParamsUboResources`      | 每帧 `memcpy`（滤波参数）       |
| **CPU→GPU** | `taauParamsUboResources`        | 每帧 `memcpy`（TAAU 参数）      |
| **GPU写**   | `shadowMapData`                 | Shadow Pass 深度写入            |
| **GPU写**   | `taauInputColorData`            | 主渲染 Pass MRT Color0          |
| **GPU写**   | `taauVelocityData`              | 主渲染 Pass MRT Color1          |
| **GPU写**   | `taauDepthData`                 | 主渲染 Pass 深度                |
| **GPU写**   | `swapChainImages`               | TAAU Resolve 输出               |
| **GPU写**   | `taauHistoryColorData[write]`   | Blit swapChain → history        |
| **GPU读**   | `shadowMapData`                 | `shadow_lit.slang` 阴影采样     |
| **GPU读**   | `taauInputColorData`            | `taau_resolve.slang` 当前帧采样 |
| **GPU读**   | `taauHistoryColorData[read]`    | `taau_resolve.slang` 历史帧采样 |
| **GPU读**   | `taauVelocityData`              | `taau_resolve.slang` 重投影     |

---

## 16. TAAU 的画质特性

TAAU 并非消除锯齿，而是**以时域稳定性换取锐度**。具体表现：

| 场景         | 主要表现                                                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **静止画面** | 单帧 jitter 导致每帧采样位置略有偏移，但经过几帧收敛后历史累积良好，画面非常稳定。由于历史权重可拉高，噪声被大量过滤，锯齿反而被抑制。          |
| **运动画面** | 锯齿被运动模糊掩盖，时域融合效果良好。但运动物体的历史重投影可能暴露（disocclusion）和遮挡（occlusion）问题，导致边缘出现**鬼影（ghosting）**。 |
| **全局**     | TAAU 始终在降低锐度换取时域稳定，细节保真度始终不如原生全分辨率渲染。真正的锐利抗锯齿需依靠 TSR/GSR 等空间重建技术，或 DLSS 的 AI 超分。        |
�，或 DLSS 的 AI 超分。        |

# TAAU（Temporal Anti-Aliasing Upsampling，`5_taau`）完整流程说明

[返回目录](../../README.md)

低分辨率渲染 + 运动矢量重投影 + 历史帧自适应融合。

---

## 1. 实现的功能

### 1.1 低分辨率主渲染 + MRT

主渲染以 `taauRenderScale=0.85`（可调）在低分辨率下进行，MRT 同时输出：

| Attachment | 内容 |
| ---------- | ---- |
| Color0 | 实例颜色（地面灰 + 立方体彩色） |
| Color1 | 速度矢量（当前帧→上一帧 NDC 差，由 `prevViewProj` 重投影计算） |
| Depth | 深度缓冲 |

**为什么低分辨率渲染**：GPU 渲染开销最大的阶段是片元着色器。以 1920×1080、85% 缩放为例，片元数从 207 万降至约 150 万（**减少约 28%**），节省的算力用于后续 TAAU 上采样 pass。低分辨率的细节损失由时域融合弥补：历史帧信息填补单帧上采样的模糊，Halton jitter 提升等效采样率。

### 1.2 TAAU 时域融合

`taau_resolve.slang` 将低分辨率结果上采样至全分辨率，核心算法：

- **Halton Jitter**：CPU 每帧注入 1/4~1/2 像素偏移，提升边缘等效采样率
- **3×3 邻域 Clamp**：扩展包络抑制鬼影
- **亮度差拒绝**：颜色不一致时降低历史权重
- **自适应权重**：速度稳定性 × 颜色稳定性 × 边缘稳定性
- **运动自适应锐化**：高速/边缘/不稳定区域自动减弱锐化
- **History Ping-Pong**：双缓冲轮换，持续累积历史信息

---

## 2. 各个 Shader 的作用

- `shadow_lit.slang` — 主渲染（纯色 + 速度，MRT：颜色 + 速度）
- `taau_resolve.slang` — TAAU 时域融合 + 上采样（全屏 pass）
- `imgui.slang` — ImGui 文字绘制

---

## 3. 初始化阶段（一次性资源）

### 3.1 Geometry

- `cubeMesh`：立方体顶点/索引缓冲（同时用于地面和立方体）
- `instanceCount = 9`（1 地面 + 8 立方体）

### 3.2 主渲染 Pass 资源

| 资源 | 类型 | 说明 |
| ---- | ---- | ---- |
| `sceneUboResources` | Uniform Buffers | VP 矩阵 + 相机位置 |
| `prevVpUboResources` | Uniform Buffers | 上一帧 VP（用于 velocity） |
| `instanceBufferResources` | Storage Buffers | 实例列表（9 个：1 地面 + 8 立方体） |
| `mainRenderPipeline` | Graphics Pipeline | MRT：2 个 color attachment + depth |

**`mainRenderPipeline` MRT 格式**（`shadow_lit.slang` 输出 `SV_TARGET0` + `SV_TARGET1`）：

| Attachment | 格式 | 用途 |
| ---------- | ---- | ---- |
| Color0 | `swapChainImageFormat` | 实例颜色 |
| Color1 | `R16G16Sfloat` | 速度（velocity） |
| Depth | `findDepthFormat()` | 深度 |

**`mainRenderDescriptorSetLayout`**（binding 0~2）：

| Binding | 类型 | 资源 |
| ------- | ---- | ---- |
| 0 | StorageBuffer | `instanceBufferResources`（instance list） |
| 1 | UniformBuffer | `sceneUboResources`（VP 矩阵） |
| 2 | UniformBuffer | `prevVpUboResources`（上一帧 VP） |

### 3.3 TAAU 资源

| 资源 | 尺寸 | 格式 | 用途 |
| ---- | ---- | ---- | ---- |
| `taauInputColorData` | `swapChainExtent × taauRenderScale` | `swapChainImageFormat` | 当前帧低分辨率颜色 |
| `taauVelocityData` | 同上 | `R16G16Sfloat` | 当前帧低分辨率速度 |
| `taauDepthData` | 同上 | depth format | 当前帧低分辨率深度 |
| `taauHistoryColorData[2]` | `swapChainExtent`（全分辨率） | `swapChainImageFormat` | 历史帧颜色 ping-pong |

**`taauDescriptorSetLayout`**（binding 0~4）：

| Binding | 类型 | 资源 |
| ------- | ---- | ---- |
| 0 | CombinedImageSampler | `taauInputColorData`（当前低分辨率颜色） |
| 1 | CombinedImageSampler | `taauHistoryColorData[historyRead]`（历史全分辨率颜色） |
| 2 | CombinedImageSampler | `taauVelocityData`（当前速度） |
| 3 | CombinedImageSampler | `taauDepthData`（当前深度） |
| 4 | UniformBuffer | `taauParamsUboResources`（TAAU 参数） |

---

## 4. 每帧 CPU 数据准备

### 4.1 `updateTAAUBuffers(currentFrame)` — Jitter + 参数 + 实例 + 场景 UBO

**Halton 序列抖动注入**（周期 8，base 2 和 3）：

```cpp
sampleIndex = (taauFrameCounter % 8) + 1
hx = halton(sampleIndex, 2) - 0.5   // [0,1) → [-0.5, 0.5)
hy = halton(sampleIndex, 3) - 0.5
taauJitterCurrent = vec2(hx/fullWidth, hy/fullHeight) * 0.45
taauFrameCounter++
```

**实例动画数据**（9 个实例）：

- 地面：固定位置 `(0, -2, 0)`，灰色
- 8 立方体：环绕半径 2.8，逆时针排列，每帧更新位置
  - 立方体 #1：快速 zigzag 动画（`FreezeHistory` 冻结）
  - 立方体 #5：轻微水平摆动

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

**PrevVPBuffer（CPU→GPU）**：

- `prevViewProj = taauPrevViewProj`（上一帧 VP，由 `updateTAAUHistory` 更新）

---

## 5. 每帧命令录制主流程

`render()` → `recordCommandBuffer(imageIndex)` → 录制到 `commandBuffers[currentFrame]`：

```
CPU: waitFence → acquireNextImage → updateUIFrame → updateTAAUBuffers
     ↓
recordCommandBuffer(imageIndex):
 1. transition swapChain/depth → ColorAttachmentOptimal/DepthAttachmentOptimal
 2. 主渲染 Pass（低分辨率 MRT）→ taauInputColor + taauVelocity + taauDepth
 3. recordTAAU(imageIndex)
 4. UI Pass → swapChain
 5. transition swapChain → PresentSrcKHR
 6. submit(wait=presentSem, signal=renderSem) → present()
```

---

## 6. Pass 1：主渲染 Color Pass（低分辨率 MRT）

### 6.1 渲染目标（低分辨率）

```
taauExtent = swapChainExtent × taauRenderScale
           // 例如 1920×1080 × 0.85 → 1632×918
```

MRT 附件（同时渲染）：

1. `taauInputColorData`（Color0）：实例颜色
2. `taauVelocityData`（Color1）：速度矢量（`shadow_lit.slang` 输出当前帧→上一帧 NDC 差）
3. `taauDepthData`（Depth）：当前帧低分辨率深度

### 6.2 `shadow_lit.slang` 功能

- **顶点着色器**：使用 `instanceBuffer[instanceID]` 的 model 矩阵变换顶点位置，输出 `currClipPos` 和 `prevClipPos`
- **片元着色器**：直接输出实例颜色；velocity = `currNDC - prevNDC`

---

## 7. Pass 2：`recordTAAU` — 时域融合 + 上采样

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

## 8. Fragment Shader 细节：`taau_resolve.slang`

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

3. **亮度差拒绝**：
   - `lumaDiff = |lumaCurrent - lumaHistory|`
   - `rejectThreshold` 随稳定性动态变化
   - `colorReject = step(rejectThreshold, lumaDiff)`
   - 历史权重乘 `(1 - colorReject)`

4. **自适应历史权重（三因子乘积）**：

   | 因子 | 含义 | 计算 |
   | ---- | ---- | ---- |
   | `motionStability` | 速度稳定性 | `saturate(1 - |velocity| × 5)` |
   | `colorStability` | 颜色稳定性 | `saturate(1 - lumaDiff × 5 × reactiveClamp)` |
   | `edgeStability` | 边缘稳定性 | `saturate(1 - edgeStrength × 0.9)` |

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

## 9. Shader 细节

### 9.1 `shadow_lit.slang`（主渲染 MRT）

顶点阶段：
- `instanceBuffer`（binding 0）：实例列表（9 个，model 矩阵 + 颜色）
- `sceneUbo`（binding 1）：当前帧 VP（含 jitter）
- `prevVpBuffer`（binding 2）：上一帧 VP（用于 velocity 计算）

片元阶段：
- 输出 `color = SV_TARGET0`（实例颜色）
- 输出 `velocity = SV_TARGET1`（currNDC - prevNDC）

> 注意：`shadow_lit.slang` 输出 `SV_TARGET0`（颜色）+ `SV_TARGET1`（velocity：当前帧→上一帧 NDC 差），`taau_resolve.slang` 据此做历史重投影。

### 9.2 `taau_resolve.slang`（时域融合）

| Binding | 资源 | 描述 |
| ------- | ---- | ---- |
| 0 | `inputColorTex` | 低分辨率当前颜色 |
| 1 | `historyColorTex` | 全分辨率历史颜色 |
| 2 | `velocityTex` | 低分辨率速度（由 `shadow_lit.slang` 输出当前帧→上一帧 NDC 差） |
| 3 | `depthTex` | 低分辨率深度 |
| 4 | `TAAUParams` UBO | 参数集 |

---

## 10. UI 参数详解

### 10.1 参数说明（`TAAUParams` / `TAAUParamsUBO`）

| 参数 | 默认值 | 范围 | 效果 |
| ---- | ------ | ---- | ---- |
| `blendFactor` | 0.90 | 0.20~0.98 | **历史基础权重**。控制当前帧与历史帧的融合比例：高值（0.9+）= 画面稳定但更模糊，拖影明显；低值（0.5~0.7）= 锐利但闪烁/噪点多。 |
| `reactiveClamp` | 0.55 | 0.20~1.20 | **亮度差敏感度**。控制颜色突变时的历史降权激进程度：高值 = 变化区域更积极抛弃历史帧，响应快但易闪烁；低值 = 变化区域也保持历史权重，画面稳但可能拖残影。 |
| `antiFlicker` | 0.88 | 0.00~1.00 | **抗闪烁 lerp**。最终颜色再对当前帧颜色做一次插值：高值 = 极稳/极软，几乎无闪烁；低值 = 锐利但条纹闪烁风险大。建议不低于 0.7。 |
| `velocityScale` | 1.00 | 0.20~2.50 | **速度幅值缩放**。对 velocity 做全局缩放，用于校准速度矢量精度。高值 = 历史重投影偏移更大；低值 = 重投影偏移不足，导致运动物体鬼影。 |
| `historyClampGamma` | 1.15 | 0.50~2.50 | **邻域包络扩展幅度**。控制 3×3 邻域颜色包络的扩展范围：高值 = 包络宽松，历史上颜色容忍度大，减少拒绝但可能引入鬼影；低值 = 包络严格，历史超出邻域范围就被拒绝，抑制鬼影但画面可能闪烁。 |
| `historyRejectThreshold` | 0.22 | 0.01~0.50 | **亮度差拒绝阈值**。超过此亮差的像素将大幅降低历史权重：高值 = 保护历史帧，难以被颜色变化刷新，可能导致拖影；低值 = 颜色稍有变化就拒绝历史，响应快但画面不稳。 |

### 10.2 其他 UI 参数

| 参数 | 效果 |
| ---- | ---- |
| `RenderScale` | **渲染缩放比例**（0.5~1.0）。控制低分辨率渲染的尺寸：高值（如 1.0）= 片元数接近原生，画质好但性能低；低值（如 0.5）= 片元数减半，性能高但依赖 TAAU 填补细节。修改会自动重建 TAAU 资源并重置历史。 |
| `FreezeHistory` | **冻结立方体 #1 的动画**。用于测试静止区域在 TAAU 下的稳定性：冻结时该立方体静止，历史帧完美累积；解冻后快速运动测试动态区域的鬼影控制。 |
| `Reset History` | **仅重置历史有效标志**（不重建资源）。手动刷新历史，强制所有像素从当前帧重新开始累积，用于修复积累错误（如相机突变后残留鬼影）。 |
| `Enable TAA Resolve` | **开关时域融合**。关闭时直接 blit 低分辨率颜色到全分辨率，跳过 TAAU pass，用于对比有/无 TAAU 的画质差异。 |

### 10.3 调参建议顺序

1. 固定 `RenderScale`（例如 0.85）
2. 先调 `blendFactor` / `historyRejectThreshold`（基础稳定性）
3. 再调 `reactiveClamp` / `historyClampGamma`（拒绝策略）
4. 最后微调 `antiFlicker`

---

## 11. 拖影与闪烁的成因分析

### 11.1 拖影（Ghosting）

**表现**：物体移动后，原位置残留上一帧的残影，逐渐淡出。

| 成因 | 说明 | 对应参数 |
| ---- | ---- | -------- |
| **历史降权不足** | 物体运动后，历史帧中对应位置的颜色与当前差异大，但权重仍偏高，导致旧颜色混入新画面。 | `blendFactor` 过高、`historyRejectThreshold` 过高 |
| **邻域包络过宽** | 3×3 邻域颜色范围大，历史被 clamp 后仍落在包络内，混入最终结果。 | `historyClampGamma` 过高 |
| **亮度差拒绝失效** | 动态区域的亮度差虽然大，但 `reactiveClamp` 过低时，颜色稳定性被高估，历史权重未充分降低。 | `reactiveClamp` 过低 |
| **velocity 不准确** | 运动物体的速度矢量计算有误差（旋转、非线性运动、实例动画），历史重投影到了错误位置。 | `velocityScale` 未校准 |
| **相机快速运动** | 相机突然旋转/平移时，所有像素的 historyUV 整体偏移，旧历史大量错位，产生大范围拖影。 | — |
| **disocclusion** | 物体移开后暴露了原本被遮挡的背景，历史帧中没有这部分背景信息，重投影到错误位置。 | — |

**典型场景**：立方体 #1 快速 zigzag 时，轨迹上残留颜色残影。

### 11.2 闪烁（Flickering）

**表现**：静止或缓慢移动的区域出现明暗/颜色不稳定跳动。

| 成因 | 说明 | 对应参数 |
| ---- | ---- | -------- |
| **历史权重过低** | 静止区域本应高权重累积历史，但 `blendFactor` 过低（< 0.7）导致每帧只融合少量历史，噪声未充分过滤。 | `blendFactor` 过低 |
| **邻域包络过窄** | 3×3 邻域颜色范围小时，历史颜色稍偏出包络就被 clamp 到边缘，每帧 clamp 结果不同，产生闪烁。 | `historyClampGamma` 过低 |
| **亮度差拒绝过于激进** | 静止区域的亮度本身有量化噪声（如渲染精度限制），`reactiveClamp` 过高时噪声也被视为颜色变化，拒绝历史。 | `reactiveClamp` 过高 |
| **historyRejectThreshold 过低** | 亮度差阈值设得太小，正常采样误差都被判定为颜色变化，频繁拒绝历史。 | `historyRejectThreshold` 过低 |
| **antiFlicker 不足** | 缺少最终 lerp 到当前帧的滤波，闪烁未被抑制。 | `antiFlicker` 过低 |
| **低 RenderScale** | 缩放比例过低（如 0.5）时，低分辨率信息量少，上采样误差大，每帧误差颜色不同。 | `RenderScale` 过低 |
| **halton jitter 采样不足** | jitter 幅度固定，高分辨率下每帧偏移量小，等效采样率不够，噪声过滤慢。 | — |

### 11.3 两者矛盾的根源

拖影与闪烁本质上是同一机制的两个极端：

```
historyWeight ← stability（三因子乘积）

stability 低 → historyWeight 低 → 当前帧权重高 → 锐利但闪烁
stability 高 → historyWeight 高 → 历史帧权重高 → 稳定但拖影
```

调参就是在这条连续谱上找平衡点。不同场景需要不同的平衡：

| 场景 | 主要矛盾 | 推荐策略 |
| ---- | -------- | -------- |
| 静止为主的 UI/截图 | 闪烁 | 提高 `blendFactor`、`antiFlicker`，降低 `reactiveClamp` |
| 相机固定、物体运动 | 拖影 | 提高 `reactiveClamp`，降低 `historyClampGamma` |
| 相机快速运动 | 拖影 | 降低 `blendFactor`，配合 `Reset History` |
| 低 RenderScale 高性能 | 闪烁+拖影 | 提高 `blendFactor` 补偿精度损失 |

### 11.4 根治手段

TAAU 的参数调节只能缓解症状，无法根治。要彻底解决：

| 方案 | 效果 |
| ---- | ---- |
| **提高 RenderScale** | 增加低分辨率信息量，上采样误差减小，闪烁和拖影同时改善 |
| **DLSS/FSR2/TSR** | AI/信号处理级重建，利用更多帧信息，效果远超参数调节 |
| **深度引导重投影** | 用深度替代颜色做邻域包络，遮挡判断更准确，减少 disocclusion 鬼影 |
| **相机运动检测** | 检测相机突变时强制刷新历史（`taauHistoryValid = false`） |

---

## 12. TAAU 的画质特性

TAAU 并非消除锯齿，而是**以时域稳定性换取锐度**。具体表现：

| 场景 | 主要表现 |
| ---- | ---- |
| **静止画面** | 单帧 jitter 导致每帧采样位置略有偏移，但经过几帧收敛后历史累积良好，画面非常稳定。由于历史权重可拉高，噪声被大量过滤，锯齿反而被抑制。 |
| **运动画面** | 锯齿被运动模糊掩盖，时域融合效果良好。但运动物体的历史重投影可能暴露（disocclusion）和遮挡（occlusion）问题，导致边缘出现**鬼影（ghosting）**。 |
| **全局** | TAAU 始终在降低锐度换取时域稳定，细节保真度始终不如原生全分辨率渲染。真正的锐利抗锯齿需依靠 TSR/GSR 等空间重建技术，或 DLSS 的 AI 超分。 |

---

## 13. 现代超分辨率技术简介

TAAU 解决了时域稳定性问题，但在空间细节重建上有本质局限。以下是业界主流的进阶方案：

### 13.1 TSR（Temporal Super Resolution）

**Unreal Engine 5** 的 TSR 是 TAAU 的深度改进版。核心区别：

| | TAAU（本实现） | TSR |
| -- | -- | -- |
| **重投影** | 颜色空间 velocity | 深度引导的遮挡检测 |
| **邻域采样** | 3×3 颜色包络 | 更大范围 + 深度权重 |
| **历史权重** | 亮度差 + 边缘 + 运动三因子 | 额外引入深度一致性判断 |
| **输出锐度** | 依赖自适应锐化，仍偏软 | 专用空间重建 pass，边缘锐利 |

TSR 在 UE5 的 Nanite 渲染管线中作为必经环节，使 Nanite 可以在 50% 分辨率下渲染仍保持高锐度。

### 13.2 FSR2 / FSR3（AMD FidelityFX Super Resolution）

FSR2 是**无 AI 的信号处理方案**，通过时域和空间算法实现超分：

- **时不依赖上一帧颜色**，而是利用当前帧的多尺度 Laplacian 金字塔信息
- **速度缓冲 + 屏幕空间遮挡检测**替代深度缓冲
- **锐化 pass**（CAS，Contrast Adaptive Sharpening）作为后处理，弥补空间信息损失
- 开源、无 AI 算力需求，兼容所有 GPU（不仅是 AMD）

FSR3 额外引入了 **AMD Fluid Motion Frames（AFMF）**帧生成技术，通过光流插帧进一步补帧。

### 13.3 DLSS（NVIDIA Deep Learning Super Sampling）

DLSS 是目前效果最好的商业方案，核心依赖：

| 组件 | 作用 |
| -- | -- |
| **专用 AI 推理核心（Tensor Core）** | 运行离线训练的神经网络，低延迟推理 |
| **离线训练网络** | 用大量高分辨率/低分辨率图像对训练，学习真实纹理的重建先验 |
| **游戏提供的 motion vector + depth** | 提供几何信息引导，提升遮挡重建精度 |
| **低延迟模式（Reflex）** | 消除 GPU-CPU 同步延迟，配合 DLSS 实现接近原生延迟 |

DLSS 的优势在于 AI 网络学到了**真实纹理的统计先验**（例如草地、砖墙、皮肤等高频纹理的结构规律），可以"猜"出低分辨率下丢失的高频细节，这是纯信号处理方案无法做到的。

NVIDIA 提供多档质量模式：

| 模式 | 渲染分辨率（4K 输出） | 性能收益 |
| -- | -- | -- |
| **DLAA** | 原生 4K | 无性能收益，质量最高 |
| **质量（Quality）** | ~66% 4K | ~50% 性能提升 |
| **平衡（Balanced）** | ~58% 4K | ~65% 性能提升 |
| **性能（Performance）** | ~50% 4K | ~110% 性能提升 |

### 13.4 技术路线对比

| | TAAU（本实现） | FSR2 | DLSS |
| -- | -- | -- | -- |
| **算力需求** | 极低（< 1 GFLOP） | 低（< 10 GFLOP） | 高（需 Tensor Core） |
| **硬件兼容** | 所有 GPU | 所有 GPU（开源） | 仅 NVIDIA |
| **空间锐度** | 依赖锐化，偏软 | CAS 锐化，中等 | AI 重建，最锐 |
| **鬼影控制** | 参数敏感 | 中等 | 最佳（深度引导） |
| **实现难度** | 中等 | 较低（开源） | 高（AI 模型专有） |

TAAU 是理解时域重建的最佳起点；掌握其原理后，可以进一步学习 FSR2 的开源实现或理解 DLSS 的设计思路。

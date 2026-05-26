# Culling 数据流与渲染流程

[返回目录](../../README.md)

- 每帧 CPU 如何准备数据
- Compute 命令里如何做 Depth Prepass / Hi-Z / Culling
- 统计数据如何回读
- Graphics 阶段如何使用 culling 结果并呈现

---

## 1. 初始化阶段（一次性资源）

`src/culling/CullingRenderer.cpp`

### 1.1 Buffer 创建（`createCullingBuffers`）

创建并长期复用的资源：

| 资源 | 类型 | 大小 | 用途 | CPU 访问 |
|---|---|---|---|---|
| `cullingGlobalUboResources` | UBO | `sizeof(SceneUBO)` | 相机 VP 矩阵 | 每帧 memcpy → GPU |
| `cullingInstanceBufferResources` | StorageBuffer | `MAX_INSTANCES × sizeof(CullingInstanceData)` | 所有实例 model+color | 每帧 memcpy → GPU |
| `cullingIndirectBufferResources` | Storage+Indirect | `sizeof(DrawCommand)` | 间接绘制命令（instanceCount 由 Compute 写入） | 每帧初值 memcpy → GPU |
| `cullingVisibleBufferResources` | StorageBuffer | `MAX_INSTANCES × uint32` | 可见实例索引列表 | GPU 写（Compute），GPU 读（Draw） |
| `cullingStatsBufferResources` | Storage+TransferSrc | `sizeof(CullingStats)` | 统计 {totalCount, visibleCount} | GPU 写（Compute），GPU→CPU 回读 |
| `cullingParamsBufferResources` | UBO | `sizeof(CullingParamsUBO)` | 视锥平面、Hi-Z 参数 | 每帧 memcpy → GPU |
| `cullingVisibleCountBuffer` | Storage+TransferDst+TransferSrc | `sizeof(uint32)` | 可见计数器（Compute 原子加） | Transfer 清零、GPU 原子写、GPU→CPU 回读 |
| `cullingStatsReadbackBuffer` | TransferDst | `sizeof(CullingStats)` | stats GPU→CPU 回读目标 | HostVisible+HostCoherent |
| `cullingVisibleReadbackBuffer` | TransferDst | `sizeof(uint32)` | 可见数 GPU→CPU 回读目标 | HostVisible+HostCoherent |
| `cullingTimestampQueryPool` | Timestamp Query | `MAX_FRAMES_IN_FLIGHT × 2` | 每帧记录 compute 阶段耗时 | GPU→CPU 回读 |

### 1.2 Descriptor Set Layouts（4 个 Pass）

#### Depth Pass（`cullingDepthDescriptorSetLayout`）
- `binding=0`: `SceneUBO`（UniformBuffer，Vertex Shader 读）
- `binding=1`: `instanceBuffer`（StorageBuffer，Vertex Shader 读）

#### Hi-Z Build Compute（`cullingHiZDescriptorSetLayout`）
- `binding=0`: `outHiZ`（StorageImage，Compute 写，当前 mip 输出）
- `binding=1`: `unusedOut`（StorageImage，当前实现未使用）
- `binding=2`: `srcDepth`（CombinedImageSampler，Compute 读）
  - mip=0：采样 `cullingDepthTexture`（深度 prepass 结果）
  - mip>0：采样 `cullingHiZTexture` 的上一 mip level
- `binding=3`: `srcSampler`（Sampler，采样参数，与 binding2 配套）

#### Culling Compute（`cullingDescriptorSetLayout`）
- `binding=0`: `SceneUBO`（UniformBuffer，Compute 读）
- `binding=1`: `instanceBuffer`（StorageBuffer，Compute 读）
- `binding=2`: `drawCommands`（RWStructuredBuffer，Compute 写 instanceCount）
- `binding=3`: `visibleIndices`（RWStructuredBuffer，Compute 写可见索引）
- `binding=4`: `stats`（RWStructuredBuffer，Compute 写统计）
- `binding=5`: `visibleCountBuffer`（RWStructuredBuffer，Compute 原子加）
- `binding=6`: `CullingParamsUBO`（UniformBuffer，Compute 读视锥/Hi-Z 参数）
- `binding=7`: `hiZTexture`（CombinedImageSampler，Compute 读 Hi-Z）
- `binding=8`: `hiZSampler`（Sampler，与 binding7 配套）

#### Draw Pass（`cullingDrawDescriptorSetLayout`）
- `binding=0`: `SceneUBO`（UniformBuffer，Vertex Shader 读）
- `binding=1`: `instanceBuffer`（StorageBuffer，Vertex Shader 读）
- `binding=2`: `visibleIndices`（StructuredBuffer，Vertex Shader 读 — 间接索引真实实例）

### 1.3 Pipeline 创建（`createCullingPipelines` + `createCullingHiZPipeline`）

| Pipeline | Shader | 入口 | 用途 |
|---|---|---|---|
| `cullingPipeline` | `culling_comp.spv` | `compMain` | GPU-Driven 视锥+遮挡剔除 |
| `cullingDepthPipeline` | `culling_depth.spv` | `vertMain` | Depth Prepass（全实例） |
| `cullingDrawPipeline` | `culling_draw.spv` | `vertMain/fragMain` | 可见实例最终绘制 |
| `cullingHiZPipeline` | `culling_hiz_build.spv` | `compMain` | 从深度图构建 Hi-Z 金字塔 |

### 1.4 深度与 Hi-Z 资源

- `createCullingDepthResources`：创建 `cullingDepthTexture`（深度 prepass 目标）
  - 尺寸：`swapChainExtent`（与屏幕分辨率相同）
  - 格式：`findDepthFormat()`
  - usage：`eDepthStencilAttachment | eSampled`
  - Sampler：Nearest + ClampToEdge + CompareOp=Always
- `createCullingHiZResources`：创建 `cullingHiZTexture`（R32Sfloat，多 mip）
  - 尺寸：与 `cullingDepthExtent` 相同
  - mipCount：`floor(log2(max(width, height))) + 1`
  - usage：`eStorage | eSampled`
- `createCullingHiZDescriptorSets`：为每帧 × 每 mip（共 `MAX_FRAMES_IN_FLIGHT × cullingHiZMipCount` 个）建 descriptor set

### 1.5 Compute 命令与同步对象

- `createCullingCommandPool`：从 `queueFamilyIndices.computeFamily` 创建（独占 compute 队列）
- `createCullingCommandBuffers`：为 `MAX_FRAMES_IN_FLIGHT` 帧各分配一个 compute command buffer
- `createCullingSyncObjects`：创建 `cullingCompleteSemaphores[MAX_FRAMES_IN_FLIGHT]`，用于 Compute→Graphics 同步

### 1.6 场景实例（`buildStressScene`）

生成城市网格状密集场景：
- 网格：`180×180 = 32400` 个立方体实例
- 位置：X/Z 范围约 ±216，带随机抖动
- 缩放：Y ∈ [0.6, 3.0] 随机高度
- 旋转：Y 轴随机旋转
- 颜色：HSV 蓝色调，部分街道区域偏暗

---

## 2. 每帧 CPU 数据准备

函数：`updateCullingBuffers(currentFrame)`

### 2.1 写入 `SceneUBO`

```cpp
sceneUbo.projection = glm::perspective(fov=45°, aspect, near=0.1, far=600)
sceneUbo.view = camera.GetViewMatrix()
sceneUbo.camPos = camera.Position
```
写入 `cullingGlobalUboResources.BuffersMapped[currentFrame]`

### 2.2 写入实例数据

```cpp
memcpy(instanceBufferMapped, sceneInstances.data(), sizeof(CullingInstanceData) * sceneInstances.size())
```
`totalInstanceCount` 固定为 32400（由 `buildStressScene` 在初始化时确定）。

### 2.3 写入初始 `DrawCommand`

```cpp
drawCmd.indexCount = cubeMesh.indices.size()
drawCmd.instanceCount = cullingEnabled ? 0 : totalInstanceCount
drawCmd.firstIndex = 0
drawCmd.vertexOffset = 0
drawCmd.firstInstance = 0
```
若开启剔除：初值为 0，等待 compute 原子写入；若关闭剔除：直接置总实例数（全可见路径）。

### 2.4 写入 `CullingParamsUBO`

```cpp
extractFrustumPlanes(projection * view, params.frustumPlanes)  // 6 个归一化平面
params.aabbMin = {-0.5, -0.5, -0.5}
params.aabbMax = {0.5, 0.5, 0.5}
params.hiZInfo = {width, height, mipCount, 0.0015f}  // depthBias=0.0015
params.totalInstances = totalInstanceCount
params.useCulling = cullingEnabled ? 1 : 0
```

---

## 3. 每帧 Compute 命令录制与执行

函数：`recordCullingCommandBuffer(imageIndex)`，录制到 `computeCommandBuffers[currentFrame]`。

### 3.1 计时起点

```cpp
commandBuffer.resetQueryPool(cullingTimestampQueryPool, currentFrame*2, 2)
commandBuffer.writeTimestamp(PipelineStage::TopOfPipe, queryPool, currentFrame*2)  // 帧 N 的起点
```

### 3.2 Depth Prepass（全实例绘制到独立深度图）

1. **Barrier**：`cullingDepthTexture` 从 `cullingDepthLayout` → `eDepthAttachmentOptimal`
2. **动态渲染**（仅深度附件）：
   - `loadOp = Clear`，清为 1.0
   - `storeOp = Store`
3. 绑定 `cullingDepthPipeline` + `cullingDepthDescriptorSets[currentFrame]`
4. 绑定 `cubeMesh` 的顶点/索引缓冲
5. **全实例绘制**：
   ```cpp
   cmdBuffer.drawIndexed(indices.size(), totalInstanceCount, 0, 0, 0)
   // SV_InstanceID 在 culling_depth.vertMain 中直接索引 instanceBuffer
   ```
   此处绘制全部实例，**与 cullingEnabled 开关无关**（culling 在后续 compute pass 中才生效）。
6. 结束渲染
7. **Barrier**：`cullingDepthTexture` → `eDepthReadOnlyOptimal`
   - 保证后续 Hi-Z Compute 可采样此深度图

> 结果：得到当前视角的完整深度图，供 Hi-Z 构建使用。

### 3.3 构建 Hi-Z（`recordCullingHiZ`）

逐级构建保守遮挡的 max-depth Hi-Z 金字塔：

1. **Barrier**：`cullingHiZTexture` 从 `cullingHiZLayout` → `eGeneral`
2. **循环 mip=0 到 mipCount-1**：
   - 绑定 `cullingHiZDescriptorSets[currentFrame * mipCount + mip]`
   - `dispatch((mipWidth+7)/8, (mipHeight+7)/8, 1)`（`numthreads(8,8,1)`）
   - `culling_hiz_build.compMain` 对目标像素做 2x2 采样取 maxDepth
   - mip0 源：`cullingDepthTexture`（经过 `eDepthReadOnlyOptimal` 转换）
   - mip>0 源：`cullingHiZTexture` 的上一 mip level
   - 每级间插入 `ComputeShaderWrite → ComputeShaderRead/Write` barrier
3. **最终 Barrier**：`cullingHiZTexture` → `eShaderReadOnlyOptimal`

> 结果：得到可被 culling compute 采样的 Hi-Z 金字塔，mip 级别按 `floor(log2(radiusPx))` 动态选择。

### 3.4 清零可见计数器

```cpp
commandBuffer.fillBuffer(cullingVisibleCountBuffer, 0, sizeof(uint32), 0u)
```
Transfer→Compute barrier（`TransferWrite → ShaderRead/ShaderWrite`），保证清零对新提交的 compute 可见。

### 3.5 执行 Culling Compute

```cpp
commandBuffer.dispatch((totalInstanceCount + 63) / 64, 1, 1)
```

**`culling_comp.slang` 数据流**（每线程处理一个实例，index = dispatchThreadID.x）：

```
1. if index >= totalInstances → return
2. if index == 0 → stats[0].totalCount = totalInstances
3. if useCulling == 0（全可见路径）:
   visibleIndices[index] = index
   visibleCountBuffer[0] = totalInstances
   drawCommands[0].instanceCount = totalInstances
   stats[0].visibleCount = totalInstances
   return
4. 读取 instanceBuffer[index].model → center = mul(model, {0,0,0,1})
5. 视锥剔除：sphereInsideFrustum(center, radius)
   - 对 6 个视锥平面逐一测试（[unroll]）
6. 遮挡剔除：occlusionTest(center, radius)
   - projectToClip(center) → NDC → UV
   - radiusPx = radius * hiZInfo.x / clip.w
   - mip = clamp(floor(log2(radiusPx)), 0, mipCount-1)
   - sampleHiZ(uv, mip) → maxDepth
   - depthTest = projectedDepth - radiusProj <= maxDepth + depthBias
7. 若可见：
   InterlockedAdd(drawCommands[0].instanceCount, 1) → dstIndex
   visibleIndices[dstIndex] = index
   InterlockedAdd(visibleCountBuffer[0], 1)
```

### 3.6 Compute 结果转可回读

1. **双重 barrier**（`ShaderWrite → TransferRead`）：
   - `cullingStatsBufferResources[currentFrame]`：stats buffer
   - `cullingVisibleCountBuffer`：可见计数器
2. **Copy 到 GPU staging buffer**：
   ```cpp
   copyBuffer(statsBuffer → cullingStatsReadbackBuffer, sizeof(CullingStats))
   copyBuffer(visibleCountBuffer → cullingVisibleReadbackBuffer, sizeof(uint32))
   ```

### 3.7 计时终点

```cpp
commandBuffer.writeTimestamp(PipelineStage::BottomOfPipe, queryPool, currentFrame*2+1)
commandBuffer.end()
```

---

## 4. Compute → Graphics 同步与 CPU 回读

### 4.1 同步机制

```
computeQueue.submit(signal=cullingCompleteSemaphores[currentFrame])
  ↓（GPU 产生信号）
graphicsQueue.submit(wait=cullingCompleteSemaphores[currentFrame])
```

Compute 的 `copyBuffer` 在 compute queue 执行；Graphics 的 `drawIndexedIndirect` 在 graphics queue 执行。信号量保证 Graphics 开始前，Compute 的回读数据已写入 staging buffer。

### 4.2 CPU 读回统计（`updateCullingStats`）

**时序**：在 `render()` 中，`updateCullingStats()` 发生在 `computeQueue.submit()` **之后**、`graphicsQueue.submit()` **之前**。

**等待机制**：通过 `device.waitForFences(inFlightFences[currentFrame])` — 等待的是上一帧（N-1）Graphics 完成时设置的 fence。因此读回的数据是 Frame N-1 的 stats，存在一帧延迟。

```
waitForFences(F(N-1))   // N-1 帧 graphics 完成才返回
  ↓
computeQueue.submit(Frame N)  // Frame N 的 compute（含 copyBuffer）
  ↓
updateCullingStats()          // 读 Frame N-1 的 stats（已由 F(N-1) 保证完成）
```

**读回内容**：
- 从 `QueryPool` 读取 timestamp N×2 和 N×2+1，计算 `cullingGpuMs = (ts[1]-ts[0]) * timestampPeriod * 1e-6`
  - 注：timestamp 对应 Frame N-1 的 compute 阶段执行时间
- 从 `cullingStatsReadbackMapped` 读取 `visibleCountCpu = stats->visibleCount`（来自 N-1 帧）
- 从 `cullingVisibleCountMapped` 读取 `visibleCountCpu`（覆盖，来自 N-1 帧）

> **一帧延迟说明**：UI 显示的 visibleCount 是上一帧的结果，这对 UI 展示是可接受的。如需当前帧数据，需在 Graphics 完成后才读回（再增加一次 fence 等待）。

### 4.3 UI 统计展示（`updateUIPanel`）

每 0.3s 平滑刷新一次显示值：
- `Enable Culling`：复选框，开关 GPU culling
- `Instances`：总实例数（32400）
- `Visible`：上一帧可见数
- `Frame`：CPU 帧耗时（ms）
- `FPS`：计算得出
- `Culling GPU`：上一帧 compute 阶段耗时（ms）

---

## 5. Graphics 阶段绘制可见实例

### 5.1 命令录制（`recordCommandBuffer`）

1. **Transition** swapChain image → `eColorAttachmentOptimal`
2. **Transition** 主 depth image → `eDepthAttachmentOptimal`
3. **动态渲染**（Color + Depth）：
   - `loadOp = Clear`，清为背景色 / 1.0
   - `storeOp = Color=Store / Depth=DontCare`
   - 绑定 `cullingDrawPipeline` + `cullingDrawDescriptorSets[currentFrame]`
   - 调用 `recordCullingDrawCommands(commandBuffer)`
4. **UI 渲染**：单独一个 RenderingPass，`loadOp = Load`，绘制 ImGui
5. **Transition**：swapChain → `ePresentSrcKHR`
6. `commandBuffer.end()`

### 5.2 实际绘制（`recordCullingDrawCommands`）

```cpp
cmdBuffer.drawIndexedIndirect(cullingIndirectBufferResources.Buffers[currentFrame], 0, 1, sizeof(DrawCommand))
```

**`culling_draw.vertMain` 如何消费可见性数据**：
- `SV_InstanceID` 由 Vulkan 自动从 0 递增到 `DrawCommand.instanceCount - 1`
- `visibleIndex = visibleIndices[instanceID]`
- `instanceData = instanceBuffer[visibleIndex]` — 通过间接索引取到真实实例的 model+color

**关键**：`DrawCommand.instanceCount` 由 Compute Shader 在 GPU 端原子写入，CPU 无需知道哪些实例可见，绘制命令自动反映 culling 结果。

### 5.3 提交与呈现

```cpp
// 等 present + compute 完成
vk::Semaphore waitSemaphores[] = {presentCompleteSemaphores, cullingCompleteSemaphores}
vk::PipelineStageFlags waitStages[] = {ColorAttachmentOutput, VertexInput}

graphicsQueue.submit({
    .waitSemaphoreCount = 2,
    .pWaitSemaphores = waitSemaphores,
    .pWaitDstStageMask = waitStages,
    .commandBufferCount = 1,
    .pCommandBuffers = &commandBuffers[currentFrame],
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &renderFinishedSemaphores[imageIndex]
}, *inFlightFences[currentFrame])

presentQueue.presentKHR(wait=renderFinishedSemaphores[imageIndex])
```

---

## 6. 完整执行顺序总览（Frame N）

```
┌─────────────────────────────────────────────────────────┐
│ T=0: device.waitForFences(F(N-1))  ← 等待 N-1 帧 graphics 完成      │
│                                                            │
│ T=1: swapChain.acquireNextImage()                         │
│                                                            │
│ T=2: updateCullingBuffers(N)                              │
│      ├─ memcpy(SceneUBO)                                  │
│      ├─ memcpy(instanceBuffer) → 32400 个实例             │
│      ├─ memcpy(DrawCommand 初值)                          │
│      └─ memcpy(CullingParamsUBO)                         │
│                                                            │
│ T=3: recordCullingCommandBuffer(N)                        │
│      ├─ writeTimestamp(TopOfPipe)                         │
│      ├─ Depth Prepass（全实例绘制到 cullingDepthTexture）  │
│      ├─ Hi-Z Build（逐 mip 构建金字塔）                   │
│      ├─ fillBuffer(visibleCountBuffer, 0)                 │
│      ├─ compute culling（原子写入 instanceCount + visibleIndices）│
│      ├─ barrier + copyBuffer(stats/count → readback)     │
│      ├─ writeTimestamp(BottomOfPipe)                      │
│      └─ end()                                            │
│                                                            │
│ T=4: computeQueue.submit(signal=computeSemaphore[N])      │
│                                                            │
│ T=5: updateCullingStats()  ← 读 Frame N-1 的 stats       │
│      ├─ getQueryPoolResults(timestamps) → cullingGpuMs   │
│      └─ memcpy(statsReadback/visibleReadback → CPU       │
│                                                            │
│ T=6: device.resetFences(F(N))                             │
│                                                            │
│ T=7: recordCommandBuffer(N)                                │
│      ├─ 过渡 swapChain/depth image                        │
│      ├─ 动态渲染 → recordCullingDrawCommands()            │
│      │   └─ drawIndexedIndirect()（可见实例）             │
│      ├─ UI 渲染（ImGui）                                   │
│      └─ 过渡 swapChain → PresentSrc                       │
│                                                            │
│ T=8: graphicsQueue.submit(                                │
│         wait=presentSemaphore[N] + computeSemaphore[N],   │
│         signal=renderFinishedSemaphore[imageIndex],       │
│         fence=F(N))                                       │
│                                                            │
│ T=9: presentQueue.presentKHR(wait=renderFinishedSemaphore)│
└─────────────────────────────────────────────────────────┘
```

---

## 7. 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU（每帧）** | `sceneUboResources` | memcpy → UBO |
| **CPU→GPU（每帧）** | `instanceBufferResources` | memcpy → StorageBuffer（32400 实例） |
| **CPU→GPU（每帧）** | `cullingParamsBufferResources` | memcpy → UBO（视锥+Hi-Z 参数） |
| **CPU→GPU（每帧）** | `cullingIndirectBufferResources` | memcpy → DrawCommand 初值 |
| **GPU Depth 写** | `cullingDepthTexture` | Depth Prepass（全部 32400 实例） |
| **GPU Compute 写** | `cullingHiZTexture` | Hi-Z Build（逐 mip） |
| **GPU Compute 写** | `drawCommands[0].instanceCount` | Culling Compute 原子写入 |
| **GPU Compute 写** | `visibleIndices[]` | Culling Compute 顺序写入 |
| **GPU Compute 写** | `visibleCountBuffer[0]` | Culling Compute InterlockedAdd |
| **GPU Compute 写** | `stats[0]` | Culling Compute 写入 totalCount/visibleCount |
| **GPU Compute 读** | `cullingDepthTexture` | Hi-Z Build mip=0 源 |
| **GPU Compute 读** | `cullingHiZTexture` | Hi-Z Build mip>0 源、occlusionTest |
| **GPU Compute 读** | `instanceBuffer` | culling pass 逐实例读取 model 矩阵 |
| **GPU Compute 读** | `sceneUbo` + `cullingParams` | 视锥平面、相机矩阵 |
| **GPU Graphics 读** | `cullingIndirectBuffer` | drawIndexedIndirect 参数源 |
| **GPU Graphics 读** | `instanceBuffer` | draw pass 经 visibleIndices 间接索引 |
| **GPU Graphics 读** | `visibleIndices` | SV_InstanceID → 真实实例索引 |
| **GPU→CPU 回读** | `cullingStatsReadbackBuffer` | copyBuffer → CPU memcpy（延迟一帧） |
| **GPU→CPU 回读** | `cullingVisibleReadbackBuffer` | copyBuffer → CPU memcpy（延迟一帧） |
| **GPU→CPU 回读** | `cullingTimestampQueryPool` | getQueryPoolResults → GPU 耗时 |

---

## 8. 资源类型决策汇总

### 8.1 Buffer `usage` 选择

| 资源 | 使用的 `usage` 标志 | 原因 |
|---|---|---|
| `cullingIndirectBufferResources` | `eStorageBuffer \| eIndirectBuffer` | Compute 原子写 instanceCount；Graphics `drawIndexedIndirect` 读取 |
| `cullingVisibleBufferResources` | `eStorageBuffer`（createStorageBuffers 默认） | Compute 写可见索引；Draw Vertex Shader 读 |
| `cullingStatsBufferResources` | `eStorageBuffer \| eTransferSrc` | Compute 写；Transfer copy 源 |
| `cullingVisibleCountBuffer` | `eStorageBuffer \| eTransferDst \| eTransferSrc` | `fillBuffer` 目标；Compute 原子加；Transfer copy 源 |
| `cullingStatsReadbackBuffer` | `eTransferDst` + HostVisible+HostCoherent | 仅作 copy 目标，CPU 读取 |
| `cullingVisibleReadbackBuffer` | `eTransferDst` + HostVisible+HostCoherent | 仅作 copy 目标，CPU 读取 |

### 8.2 Image `usage` 与 layout

| 资源 | `usage` | Layout 转换路径 |
|---|---|---|
| `cullingDepthTexture` | `eDepthStencilAttachment \| eSampled` | `Undefined → DepthAttachmentOptimal(prepass) → DepthReadOnlyOptimal(Hi-Z采样) → DepthAttachmentOptimal(下帧)` |
| `cullingHiZTexture` | `eStorage \| eSampled` | `Undefined → General(逐mip构建) → ShaderReadOnlyOptimal(culling采样) → General(下帧)` |

### 8.3 同步屏障一览

| # | 源操作 | 目标操作 | Stage Mask | Access Mask |
|---|---|---|---|---|
| 1 | Depth Prepass 写 | Hi-Z Build 读 | `LateFragmentTests → ComputeShader` | `DepthStencilAttachmentWrite → ShaderSampledRead` |
| 2 | Hi-Z mip N 写 | Hi-Z mip N+1 读 | `ComputeShader → ComputeShader` | `ShaderWrite → ShaderRead\|ShaderWrite` |
| 3 | Hi-Z 全部写完 | Culling Compute 读 | `ComputeShader → ComputeShader` | `ShaderWrite → ShaderSampledRead` |
| 4 | fillBuffer 写 | Culling Compute 原子加 | `Transfer → ComputeShader` | `TransferWrite → ShaderRead\|ShaderWrite` |
| 5 | Culling Compute 写 stats | Transfer copy | `ComputeShader → Transfer` | `ShaderWrite → TransferRead` |
| 6 | Culling Compute 写 count | Transfer copy | `ComputeShader → Transfer` | `ShaderWrite → TransferRead` |

---

## 9. Descriptor Set Layout 按 Pass 汇总

| Pass | Layout | Bindings |
|---|---|---|
| Depth Prepass | `cullingDepthDescriptorSetLayout` | b0: SceneUBO(Uniform), b1: instanceBuffer(Storage) |
| Hi-Z Build | `cullingHiZDescriptorSetLayout` | b0: outHiZ(StorageImage), b1: unusedOut, b2: srcDepth(Sampler), b3: srcSampler |
| Culling Compute | `cullingDescriptorSetLayout` | b0: SceneUBO, b1: instanceBuffer, b2: drawCommands, b3: visibleIndices, b4: stats, b5: visibleCount, b6: params, b7: hiZTexture, b8: hiZSampler |
| Draw | `cullingDrawDescriptorSetLayout` | b0: SceneUBO, b1: instanceBuffer, b2: visibleIndices |

# Culling（`RENDERING_LEVEL == 8`）数据流与渲染流程（按代码执行顺序）

[返回目录](../../README.md)

本文只描述 **`RENDERING_LEVEL==8`** 在当前代码中的执行顺序：
- CPU 每帧如何准备数据
- Compute 命令里如何做 Depth Prepass / Hi-Z / Culling
- Graphics 阶段如何使用 culling 的结果
- 统计数据如何回读到 UI

---

## 1. 初始化阶段（一次性）

`src/Core/Renderer_Culling.cpp` 

### 1.1 Buffer 创建（`createCullingBuffers`）

创建并长期复用以下资源：

- `cullingGlobalUboResources`：每帧 `SceneUBO`
- `cullingInstanceBufferResources`：每帧实例数据 `CullingInstanceData[]`
- `cullingIndirectBufferResources`：每帧 `DrawCommand`（compute 写，draw indirect 读）
- `cullingVisibleBufferResources`：每帧可见实例索引 `uint[]`
- `cullingStatsBufferResources`：每帧统计 `CullingStats`
- `cullingParamsBufferResources`：每帧 `CullingParamsUBO`
- `cullingVisibleCountBuffer`：全局计数器（GPU device local，compute 原子加）
- `cullingStatsReadbackBuffer`：统计，GPU->CPU 回读
- `cullingVisibleReadbackBuffer`：可见数，GPU->CPU 回读
- `cullingTimestampQueryPool`：每帧 2 个 timestamp（开头/结尾）

### 1.2 Descriptor Layout / Pool / Set

#### Depth Pass（`cullingDepthDescriptorSetLayout`）
- `binding0`: `SceneUBO`
- `binding1`: `instanceBuffer`

#### Hi-Z Build Compute（`cullingHiZDescriptorSetLayout`，对应 `shaders/culling_hiz_build.slang`）
- `binding0`: `RWTexture2D<float> outHiZ`
  - 当前 mip 的输出目标（写入 `cullingHiZTexture` 的第 `mip` 级）
- `binding1`: `RWTexture2D<float> unusedOut`
  - 目前实现中未使用（与 `binding0` 指向同一 mip view）
- `binding2`: `Texture2D<float> srcDepth`
  - `mip0` 时读取 `cullingDepthTexture`
  - `mip>0` 时读取 `cullingHiZTexture` 的上一 mip
- `binding3`: `SamplerState srcSampler`
  - 采样 `srcDepth` 使用的采样器

补充：`culling_hiz_build.slang` 的 `compMain` 以 `numthreads(8,8,1)` 运行，对每个目标像素做 2x2 采样并取 `maxDepth`，构建保守遮挡用的 max-depth Hi-Z。

#### Culling Compute（`cullingDescriptorSetLayout`）
- `binding0`: `SceneUBO`
- `binding1`: `instanceBuffer`
- `binding2`: `drawCommands`（RW）
- `binding3`: `visibleIndices`（RW）
- `binding4`: `stats`（RW）
- `binding5`: `visibleCountBuffer`（RW）
- `binding6`: `CullingParamsUBO`
- `binding7`: `hiZTexture`（采样）
- `binding8`: `hiZSampler`

#### Draw Pass（`cullingDrawDescriptorSetLayout`）
- `binding0`: `SceneUBO`
- `binding1`: `instanceBuffer`
- `binding2`: `visibleIndices`

### 1.3 Pipeline 创建（`createCullingPipelines` + `createCullingHiZPipeline`）

- Compute Culling：`culling_comp.spv`
- Depth Prepass：`culling_depth.spv`
- Draw Pass：`culling_draw.spv`
- Hi-Z 构建 Compute：`culling_hiz_build.spv`

### 1.4 深度与 Hi-Z 资源

- `createCullingDepthResources`：创建 `cullingDepthTexture`（深度 prepass 目标）
- `createCullingHiZResources`：创建 `cullingHiZTexture`（R32Sfloat，多 mip）及每级 mip view
- `createCullingHiZDescriptorSets`：为每帧、每 mip 建 descriptor set

### 1.5 Compute 命令与同步

- `createCullingCommandPool` / `createCullingCommandBuffers`
- `createCullingSyncObjects`（`cullingCompleteSemaphores`）

---

## 2. 每帧 CPU 数据准备（先执行）

函数：`updateCullingBuffers(currentImage)`

### 2.1 写入 `SceneUBO`

CPU 计算并上传：
- `projection`
- `view`
- `camPos`

### 2.2 收集实例数据model + color并上传 `instanceBuffer`

- 调用 `scene->world.collectRenderInstances(...)`
- 将 `model + color` 写入 `cullingInstanceBufferResources.BuffersMapped[currentImage]`
- 得到 `cullingTotalCountCpu`

### 2.3 初始化 `DrawCommand`

- `indexCount` 来自 mesh 索引数
- `instanceCount`：
  - 若开启剔除：先置 0，等待 compute 写入可见数量
  - 若关闭剔除：直接置总实例数

### 2.4 写入 `CullingParamsUBO`

- 从 `projection * view` 提取 6 个视锥平面
- 写入本地 AABB（当前是立方体 `[-0.5,0.5]`）
- 写入 Hi-Z 信息：`width / height / mipCount / depthBias`
- 写入 `totalInstances`
- 写入 `useCulling`

---

## 3. 每帧 Compute 命令录制（核心流程）

函数：`recordCullingCommandBuffer(imageIndex)`

按命令顺序如下。

### 3.1 计时起点

- reset 当前帧 query
- 写入 start timestamp（TopOfPipe）

### 3.2 Depth Prepass

1. Barrier：`cullingDepthTexture`
   - `oldLayout -> eDepthAttachmentOptimal`
2. 开始动态渲染（仅深度附件）
3. 绑定 `cullingDepthPipeline` + depth descriptor set
4. 绘制 **全部实例**：
   - `drawIndexed(..., instanceCount = cullingTotalCountCpu, ...)`
5. 结束渲染
6. Barrier：深度图
   - `eDepthAttachmentOptimal -> eDepthReadOnlyOptimal`

> 结果：得到当前视角深度图，供 Hi-Z 构建与后续 culling 采样。

### 3.3 构建 Hi-Z（`recordCullingHiZ`）

1. Barrier：`cullingHiZTexture`
   - `oldLayout -> eGeneral`（允许 compute 写）
2. 对每个 mip 逐级 dispatch
   - `mip0` 源是深度图
   - `mip>0` 源是上一 mip
3. 每级 mip 后插入 compute 内存屏障，保证下一级可读
4. 全部完成后 Barrier：
   - `eGeneral -> eShaderReadOnlyOptimal`

> 结果：得到可被 culling compute 采样的 Hi-Z 金字塔。

### 3.4 清零可见计数器

- `fillBuffer(cullingVisibleCountBuffer, 0)`
- transfer->compute barrier，保证清零结果对 compute 可见

### 3.5 执行 Culling Compute

1. 绑定 `cullingPipeline`
2. 绑定 `cullingDescriptorSets[currentFrame]`
3. `dispatch(ceil(totalInstances/64), 1, 1)`

`culling_comp.slang` 内数据流：
- 每线程处理一个实例 `index`
- 读取该实例的数据 `instanceBuffer[index]`
- 先视锥测试（`sphereInsideFrustum`）
- 再遮挡测试（`occlusionTest`，按包围球屏幕尺寸选 mip 采样 Hi-Z）
- 若可见：
  - 原子增加 `drawCommands[0].instanceCount`
  - 写 `visibleIndices[dstIndex] = index`
  - 原子增加 `visibleCountBuffer[0]`

`useCulling == 0` 时：
- 直接全可见路径，`instanceCount = totalInstances`

### 3.6 Compute 结果转可回读

1. compute->transfer barrier：
   - `statsBuffer`、`visibleCountBuffer` 从 shader write 变为 transfer read
2. `copyBuffer`：
   - `stats -> cullingStatsReadbackBuffer`
   - `visibleCount -> cullingVisibleReadbackBuffer`

### 3.7 计时终点

- 写入 end timestamp（BottomOfPipe）
- 结束 compute command buffer

---

## 4. Graphics 阶段使用 Culling 结果

函数：`recordCullingDrawCommands(commandBuffer)`

顺序：
1. 绑定 `cullingDrawPipeline`
2. 绑定 `cullingDrawDescriptorSets[currentFrame]`
3. 绑定 mesh 顶点/索引缓冲
4. 调用 `drawIndexedIndirect(cullingIndirectBuffer[currentFrame], ...)`

关键点：
- 本帧实际绘制实例数来自 `DrawCommand.instanceCount`
- 该值由 compute 阶段在 GPU 端写入
- CPU 不需要回传每个可见实例列表再发 draw call

---

## 5. CPU 读回统计并显示 UI

### 5.1 读回统计（`updateCullingStats`）

- 从 `QueryPool` 读取本帧前后 timestamp，计算 `cullingGpuMs`
- 从 `cullingStatsReadbackMapped` 读取 `totalCount/visibleCount`
- 从 `cullingVisibleCountMapped` 读取可见数（最终覆盖到 `cullingVisibleCountCpu`）

### 5.2 UI（`updateCullingUI`）

显示：
- 是否开启 culling
- 总实例数 `Instances`
- 可见数 `Visible`
- CPU 帧时间 `Frame`
- GPU culling 时间 `Culling GPU`

---

## 6. 一张“按执行先后”的总流程图（文字版）

1. `updateCullingBuffers`（CPU写UBO/实例/参数/初始DrawCommand）
2. `recordCullingCommandBuffer`：
   - 深度 prepass（全实例）
   - 构建 Hi-Z
   - 清零 visibleCount
   - compute culling（写 visibleIndices + indirect instanceCount + stats）
   - copy 到 readback
3. graphics 主渲染调用 `recordCullingDrawCommands`：
   - `drawIndexedIndirect` 只画可见实例
4. `updateCullingStats` 回读
5. `updateCullingUI` 显示

---

## 7. 当前实现的数据读写关系速查

- CPU -> GPU（每帧）
  - `SceneUBO`
  - `instanceBuffer`
  - `CullingParamsUBO`
  - `DrawCommand(初值)`

- GPU Compute 写
  - `drawCommands[0].instanceCount`
  - `visibleIndices[]`
  - `visibleCountBuffer[0]`
  - `stats[0]`

- GPU Graphics 读
  - `drawIndexedIndirect` 读取 `drawCommands`
  - Draw VS 通过 `visibleIndices` 间接索引真实实例

- GPU -> CPU 回读
  - `statsBuffer -> statsReadback`
  - `visibleCountBuffer -> visibleReadback`
  - `timestamp query -> cullingGpuMs`

---

## 8. 为了完成这些数据流向，`usage / DescriptorSetLayout` 应该如何选择

为了实现这些功能，创建资源时要选择什么类型。

### 8.1 Buffer `usage` 

#### 8.1.1 `cullingIndirectBufferResources`（`DrawCommand`）

数据流：
- Compute 写 `instanceCount`
- Graphics `drawIndexedIndirect` 读取命令

 `eStorageBuffer | eIndirectBuffer` （compute RW + indirect draw 参数源）。

#### 8.1.2 `cullingVisibleBufferResources`（`visibleIndices[]`）

数据流：
- Compute 写可见实例索引
- Draw 阶段读取该索引表

`eStorageBuffer`（该资源是“GPU 读写的大数组”，不是 uniform）。

#### 8.1.3 `cullingStatsBufferResources`（`CullingStats`）

数据流：
- Compute 写统计
- Transfer copy 到 readback

`eStorageBuffer | eIndirectBuffer`
#### 8.1.4 `cullingVisibleCountBuffer`（单值计数器）

数据流：
- 每帧 `fillBuffer` 清零
- Compute 原子加
- Transfer copy 回读

- `eStorageBuffer` | `eTransferDst`（接受 `fillBuffer`）| `eTransferSrc`（作为 copy 源）

#### 8.1.5 回读 Buffer（`statsReadback/visibleReadback`）

数据流：
- 仅作为 copy 目标并由 CPU 读取

应包含：
- `eTransferDst`
- 内存属性：`HostVisible | HostCoherent`

---

### 8.2 Image `usage` 与 layout 

#### 8.2.1 `cullingDepthTexture`

数据流：
- Depth Prepass 写
- Hi-Z 构建采样读

- `eDepthStencilAttachment | eSampled`

layout：
- `Undefined -> eDepthAttachmentOptimal -> eDepthReadOnlyOptimal`

#### 8.2.2 `cullingHiZTexture`

数据流：
- Hi-Z Build Compute 写（逐 mip）
- Culling Compute 采样读

- `eStorage`（compute 写 image）| `eSampled`（后续采样）

layout：
- 构建时：`eGeneral`
- 构建后供采样：`eShaderReadOnlyOptimal`

---

### 8.3 DescriptorSetLayout （按 Pass 拆分）

#### 8.3.1 Depth Pass（`cullingDepthDescriptorSetLayout`）
- `binding0`：`eUniformBuffer`（SceneUBO）
- `binding1`：`eStorageBuffer`（instanceBuffer）

用途单一：只服务 depth prepass，保持最小绑定集。

#### 8.3.2 Culling Compute（`cullingDescriptorSetLayout`）
- 输入：`SceneUBO`、`instanceBuffer`、`params`、`hiZTexture+sampler`
- 输出：`drawCommands`、`visibleIndices`、`stats`、`visibleCount`

对应类型：
- 固定小参数：`eUniformBuffer`
- 大数组/原子写回：`eStorageBuffer`
- 采样图：`eCombinedImageSampler`

#### 8.3.3 Draw Pass（`cullingDrawDescriptorSetLayout`）
- `binding0`：SceneUBO
- `binding1`：instanceBuffer
- `binding2`：visibleIndices

关键点：draw 只保留“绘制所需最小集”，不混入 stats/count。

#### 8.3.4 Hi-Z Build（`cullingHiZDescriptorSetLayout`）
结合 `shaders/culling_hiz_build.slang`：
- `binding0`：`eStorageImage`（输出 mip）
- `binding1`：`eStorageImage`（当前实现未实际使用）
- `binding2`：`eCombinedImageSampler`（源深度/上一 mip）
- `binding3`：`eCombinedImageSampler`（同源采样信息，兼容位）

说明：当前 shader 的 `binding1`/`binding3` 偏“兼容保留位”，后续可以在重构时收敛。

---

### 8.4 同步点为什么这样选（与数据流一一对应）

1. 深度写 -> 深度采样
- `DepthAttachmentWrite -> ShaderSampledRead`
- 用于 Hi-Z 构建读取 prepass 深度。

2. Hi-Z mip N 写 -> mip N+1 读
- `Compute ShaderWrite -> Compute ShaderRead`
- 每级 mip 之后插入 compute barrier。

3. 计数器清零 -> Compute 原子加
- `TransferWrite -> ShaderRead/ShaderWrite`
- 避免沿用上一帧值。

4. Compute 写 stats/count -> Transfer copy
- `ShaderWrite -> TransferRead`
- 保证回读看到的是本帧新值。

---

### 8.5 可复用的决策步骤（以后新增资源可照此做）

1. 先画数据流：谁写、谁读、在哪个 pass。
2. 反推 `usage`：
   - compute RW -> `eStorageBuffer/eStorageImage`
   - indirect draw -> `eIndirectBuffer`
   - copy 源/目标 -> `eTransferSrc/eTransferDst`
   - 采样读 -> `eSampled`
3. 再选 descriptor type：
   - 固定参数 -> `eUniformBuffer`
   - 大数组/写回 -> `eStorageBuffer`
   - 采样纹理 -> `eCombinedImageSampler`
   - compute 写纹理 -> `eStorageImage`
4. 最后补齐 barrier/layout：
   - 只要阶段 A 写、阶段 B 读，就显式声明 stage/access/layout 转换。

按这 4 步，能保证“资源声明、shader 绑定、命令顺序、同步关系”一致。


# 13_multithreaded — 多线程渲染与帧图架构

## 1. 目标渲染效果

展示 Vulkan 多线程渲染架构的核心组件：ThreadPool（CPU 多线程命令录制）、FrameGraph（渲染 Pass 编排）、RenderBatcher（批次拆分）、GpuProfiler（GPU 耗时统计），以及自动对比单线程/多线程的性能基准测试工具。

**渲染画面**：
- viking_room.glb 模型，纹理化渲染。
- 大量实例化物体（Medium preset: 2000 静态 + 500 动态），验证多线程录制收益。
- 粒子系统和点光源数据（已分配资源，但本模块重点不在渲染效果而在性能架构）。
- 实时 Benchmark 对比：自动运行单线程/多线程各 10 秒，统计 Frame/P95/P99 和 Record 耗时。

---

## 2. 核心架构组件

### 2.1 ThreadPool

基于 C++ 标准库的消费者-生产者线程池：

```cpp
class ThreadPool {
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::condition_variable cv_;
    std::mutex mutex_;
    std::atomic<bool> stopping_{false};

    // 启动 N 个工作线程
    void start(uint32_t workerCount);

    // 入队任务，返回 std::future
    template <typename Fn, typename... Args>
    auto enqueue(Fn&& fn, Args&&... args) -> std::future<std::invoke_result_t<Fn, Args...>> {
        auto task = std::make_shared<std::packaged_task<ReturnT()>>(
            std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
        auto result = task->get_future();
        {
            std::scoped_lock lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }
};
```

### 2.2 FrameGraph

定义渲染 Pass 及其开关状态：

```cpp
struct FramePass {
    std::string name;
    bool enabled = true;
};

class FrameGraph {
    std::vector<FramePass> passes_;
public:
    void reset();
    void addPass(const std::string& name, bool enabled = true);
    [[nodiscard]] const std::vector<FramePass>& passes() const;
};
```

注册的 Pass：

| Pass 名称 | 默认启用 |
|---|---|
| Compute: Particle Update | ✓ |
| Compute: Visibility Culling | ✓ |
| Graphics: Shadow | ✓ |
| Graphics: Geometry | ✓ |
| Graphics: Lighting | ✓ |
| Graphics: Particle | ✓ |
| Graphics: PostFX | ✓ |

### 2.3 RenderBatcher

将大量绘制任务均分到多个工作线程：

```cpp
struct RenderBatch {
    uint32_t begin = 0;
    uint32_t end = 0;
};

class RenderBatcher {
public:
    // 将 N 个 item 均分为 M 个 batch
    static std::vector<RenderBatch> splitEvenly(uint32_t itemCount, uint32_t batchCount);
};
```

### 2.4 GpuProfiler

简化版 GPU 耗时记录器（不依赖 Vulkan Profiling 扩展，存储手动注入的耗时数据）：

```cpp
class GpuProfiler {
    std::vector<GpuPassTiming> timings_;
public:
    void beginFrame();
    void endFrame();
    void setPassTiming(const std::string& passName, float timeMs);
    [[nodiscard]] const std::vector<GpuPassTiming>& timings() const;
};
```

---

## 3. 渲染流程总览

```
┌─────────────────────────────────────────────────────────────┐
│  CPU: updateFrameData()                                     │
│  └─ GlobalUBO: view / proj                                   │
│                                                               │
│  CPU: updateInstanceBuffer()                                │
│  ├─ 静态实例 (currentPresetConfig.staticInstanceCount)    │
│  └─ 动态实例 (currentPresetConfig.dynamicInstanceCount)  │
│                                                               │
│  CPU: dispatchWorkerRecording()                              │
│  ├─ staticWorkers 个线程录制静态批次                        │
│  └─ dynamicWorkers 个线程录制动态批次                        │
│      └─ 每线程 → recordWorkerRange() → WorkerRecordStats   │
│                                                               │
│  CPU: gpuProfiler.beginFrame()                               │
│  CPU: recordPrimaryCommandBuffer(imageIndex)                 │
│  ├─ 主 CommandBuffer 录制 (VK_COMMAND_BUFFER_LEVEL_PRIMARY)│
│  ├─ secondaryStaticCommandBuffer[currentFrame]              │
│  └─ secondaryDynamicCommandBuffer[currentFrame]            │
│                                                               │
│  GPU: primaryCommandBuffer → submit → present               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 流程需要用到什么东西，用在哪里

### 4.1 场景配置（Scene Preset）

| Preset | 静态实例 | 动态实例 | 粒子数 | 点光源 | 发射率 |
|---|---|---|---|---|---|
| **Small** | 1,000 | 200 | 50,000 | 128 | 3,000/s |
| **Medium** | 2,000 | 500 | 100,000 | 256 | 6,000/s |
| **Large** | 5,000 | 1,000 | 200,000 | 512 | 12,000/s |

### 4.2 Buffer 资源

| 资源 | 类型 | 大小 | 用途 |
|---|---|---|---|
| `globalUboResources` | UniformBuffer | sizeof(GlobalUBO) | view + proj |
| `instanceBufferResources` | StorageBuffer | N×sizeof(InstanceData) | N 个 model 矩阵 |
| `particleBufferResources` | StorageBuffer | 预留 | 粒子数据（本模块不渲染） |
| `lightBufferResources` | StorageBuffer | 预留 | 点光源数据（本模块不渲染） |

**`GlobalUBO`**：

```cpp
struct GlobalUBO {
    glm::mat4 view;
    glm::mat4 proj;
};
```

**`InstanceData`**：

```cpp
struct InstanceData {
    glm::mat4 model; // 每个实例的 model 矩阵
};
```

### 4.3 Descriptor Set Layout

单一 layout，绑定三个资源：

| Binding | Type | Stage | 内容 |
|---|---|---|---|
| b0 | UniformBuffer | Vertex | `globalUboResources` |
| b1 | StorageBuffer | Vertex | `instanceBufferResources` |
| b2 | CombinedImageSampler | Fragment | viking_room 纹理 |

### 4.4 Secondary Command Buffer

每个帧（MAX_FRAMES_IN_FLIGHT=2）分配两个 Secondary CB：

| CommandBuffer | 用途 |
|---|---|
| `secondaryStaticCommandBuffers[i]` | 录制静态实例 draw call |
| `secondaryDynamicCommandBuffers[i]` | 录制动态实例 draw call |

通过 `vk::CommandBufferUsageFlagBits::eRenderPassContinue` 在主 CB `beginRendering` 块内通过 `primary.executeCommands()` 执行。

---

## 5. 东西的初始化过程

### 5.1 调用链

```
prepareResource()
 ├─ applyPreset(currentPreset)  // 设置 Medium 配置

 ├─ loadModel("viking_room.glb", mesh)
 ├─ createVertexBuffer(mesh) / createIndexBuffer(mesh)
 ├─ LoadTextureFromFile("viking_room.png", texture)
 └─ createTextureSampler(texture.textureSampler)

 ├─ createSceneBuffers()
 │   ├─ createUniformBuffers(globalUboResources)
 │   └─ createStorageBuffers(instanceBufferResources)
 │       → 2000+500 × sizeof(InstanceData)
 │
 ├─ createDescriptors()
 │   └─ 单一 descriptorSetLayout: b0=UBO, b1=Storage, b2=ImageSampler
 │
 ├─ createPipeline()
 │   └─ instanced.spv → vertMain/fragMain
 │       → VertexInput: 2 bindings (mesh VB + InstanceData SB)
 │
 ├─ createSecondaryCommandResources()
 │   └─ 为每帧分配 2 个 secondary CB (eRenderPassContinue)
 │
 ├─ updateDescriptorSets()  // 绑定 texture + globalUbo + instanceBuffer
 │
 ├─ initFrameGraph()  // 注册 7 个 Frame Pass
 │
 ├─ initThreading()
 │   └─ threadPool.start(workerThreadCount=4)
 │       └─ workerStats.resize(4)
 │
 └─ initUI()
```

---

## 6. 渲染循环

### 6.1 `render()` 主流程

```
1. device.waitForFences(inFlightFences[currentFrame])
2. swapChain.acquireNextImage()
3. device.resetFences()

4. if (presetResourcesDirty):
     device.waitIdle()
     recreatePresetDrivenBuffers()  // 切换 preset 时重建 instanceBuffer

5. updateFrameData(currentFrame)      // GlobalUBO: view + proj
6. updateInstanceBuffer(currentFrame) // 静态+动态 instance model 矩阵
7. updateUIFrame()

8. gpuProfiler.beginFrame()
9. dispatchWorkerRecording(currentFrame)  // 多线程录制
   ├─ 按比例拆分 staticWorkers / dynamicWorkers
   ├─ threadPool.enqueue(recordWorkerRange) × N
   └─ 收集所有 future → workerStats
10. gpuProfiler.endFrame()

11. recordPrimaryCommandBuffer(imageIndex)  // 录制主 CB
12. updateBenchmarkFlow(dt, totalRecordMs)  // 自动 Benchmark 状态机

13. graphicsQueue.submit(wait=presentCompleteSem, signal=renderFinishedSem)
14. presentQueue.presentKHR()
15. currentFrame = (currentFrame+1)%2
```

### 6.2 `dispatchWorkerRecording` — 多线程分发

```
dispatchWorkerRecording(frameIndex):
  if (!enableMultiThreadRecording):
    // 单线程回退：合并所有批次
    recordWorkerRange(frameIndex, {0, staticCount}, false)
    recordWorkerRange(frameIndex, {0, dynamicCount}, true)
    return

  // 比例分配线程
  staticWorkers = max(1, workerCount × staticCount / sceneInstanceCount)
  dynamicWorkers = workerCount - staticWorkers

  staticBatches = RenderBatcher::splitEvenly(staticCount, staticWorkers)
  dynamicBatches = RenderBatcher::splitEvenly(dynamicCount, dynamicWorkers)

  // 入队所有批次
  for batch in staticBatches:
    futures.push_back(threadPool.enqueue(
      [=]() { return recordWorkerRange(frameIndex, batch, false); }))
  for batch in dynamicBatches:
    futures.push_back(threadPool.enqueue(
      [=]() { return recordWorkerRange(frameIndex, batch, true); }))

  // 等待所有结果
  for future in futures:
    workerStats.push_back(future.get())
```

### 6.3 `recordPrimaryCommandBuffer` 完整命令序列

```
primary.begin()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 swapChain → ColorAttachmentOptimal
 depth → DepthAttachmentOptimal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

beginRendering(color + depth, loadOp=CLEAR, flags=eContentsSecondaryCommandBuffers)
  primary.setViewport(0, fullExtent)
  primary.setScissor(0, fullExtent)

  // 录制 secondary CB（inheritanceRenderingInfo 继承渲染信息）
  recordSecondary(secondaryStaticCommandBuffers[i], firstInstance=0, staticCount)
    secondary.reset()
    secondary.begin(flags=eRenderPassContinue, inheritanceInfo)
    secondary.setViewport / setScissor
    secondary.bindPipeline(instancedPipeline)
    secondary.bindDescriptorSets(globalDescriptorSets[currentFrame])
    secondary.bindVertexBuffers(mesh.vertexBuffer)
    secondary.bindIndexBuffer(mesh.indexBuffer)
    secondary.drawIndexed(mesh.indices × staticCount instances)
    secondary.end()

  recordSecondary(secondaryDynamicCommandBuffers[i], firstInstance=staticCount, dynamicCount)
    [同上，每个动态实例从 staticCount 偏移]

  // 执行 secondary CB
  primary.executeCommands({*secondaryStatic, *secondaryDynamic})

endRendering()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 UI Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 beginRendering(swapChain, loadOp=LOAD)
   recordUI(primary, imageIndex)
 endRendering()

 swapChain → PresentSrcKHR

primary.end()

graphicsQueue.submit(primary)
```

### 6.4 `recordWorkerRange` — 模拟录制

**注意**：本模块的 `recordWorkerRange` 是**模拟实现**，仅记录批次尺寸和耗时，不实际操作 Vulkan Command Buffer（真实多线程命令录制需要更复杂的同步和生命周期管理）：

```cpp
WorkerRecordStats recordWorkerRange(uint32_t frameIndex, RenderBatch batch, bool dynamicBatch) {
    WorkerRecordStats stats{};
    stats.drawCalls = batch.end - batch.begin;
    if (dynamicBatch) stats.dynamicDrawCalls = stats.drawCalls;
    else stats.staticDrawCalls = stats.drawCalls;
    // 记录"模拟录制耗时"（timer 高精度计时）
    return stats;
}
```

---

## 7. 自动 Benchmark 机制

### 7.1 Benchmark 状态机

```
None → SingleThread → MultiThread → Done
```

**SingleThread 阶段**：
- `enableMultiThreadRecording = false`
- 采样 Frame Time 和 Record Time
- 达到 `benchmarkDurationSeconds` 后切换

**MultiThread 阶段**：
- `enableMultiThreadRecording = true`
- 采样多线程下 Frame Time 和 Record Time
- 达到时长后切换

**Done 阶段**：
- 恢复原始 `enableMultiThreadRecording` 状态
- 显示对比表格：Single vs Multi 的 Avg/P95/P99

### 7.2 百分位计算

```cpp
PercentileStats calcPercentiles(samples):
  avg = sum(samples) / count
  sorted = sort(samples)
  p95 = sorted[count × 0.95]
  p99 = sorted[count × 0.99]
```

---

## 8. 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU** | `globalUboResources` | 每帧 memcpy → UBO |
| **CPU→GPU** | `instanceBufferResources` | 每帧 memcpy → SSBO（静态+动态 model 矩阵） |
| **CPU→GPU** | `particleBufferResources` | 预留（本模块不写） |
| **CPU→GPU** | `lightBufferResources` | 预留（本模块不写） |
| **GPU读** | `globalUboResources` | Vertex shader (b0) |
| **GPU读** | `instanceBufferResources` | Vertex shader (b1) — instanceID 索引 |
| **GPU读** | `viking_room` texture | Fragment shader (b2) — 模型纹理 |
| **GPU写** | SwapChain | Fragment shader 输出 |

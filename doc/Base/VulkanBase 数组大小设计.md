# VulkanBase 数组大小设计：swapChainImages.size() vs MAX_FRAMES_IN_FLIGHT

[返回目录](../../README.md)

---

## 1. 问题背景

`VulkanBase` 中所有与帧相关的数组同时存在两个概念：

| 概念                     | 值               | 含义                                              |
| ------------------------ | ---------------- | ------------------------------------------------- |
| `MAX_FRAMES_IN_FLIGHT`   | 2                | CPU 可同时"在飞"的帧数（CPU 提交后不等 GPU 完成） |
| `swapChainImages.size()` | 动态（通常 2~3） | 交换链图像数量，由驱动/硬件决定                   |


---

## 2. 为什么基类用 `swapChainImages.size()`

### 2.1 一一对应

交换链中的每张图像（SwapChain Image）有且只有 **一个** renderTarget。正确的一一对应如下：

```
swapChainImages[0] ←→ presentCompleteSemaphores[0] ←→ renderFinishedSemaphores[0] ←→ inFlightFences[0] ←→ commandBuffers[0]
swapChainImages[1] ←→ presentCompleteSemaphores[1] ←→ renderFinishedSemaphores[1] ←→ inFlightFences[1] ←→ commandBuffers[1]
swapChainImages[2] ←→ presentCompleteSemaphores[2] ←→ renderFinishedSemaphores[2] ←→ inFlightFences[2] ←→ commandBuffers[2]  (如果存在)
```

这意味着 `acquireNextImage` 返回的 `imageIndex` 索引可以直接用于访问所有这些数组。

当前 `VulkanBase` 的实现（`VulkanBase_commands.cpp`）使用的是 `swapChainImages.size()`：

```22:43:src/Base/VulkanBase_commands.cpp
bool VulkanBase::createCommandBuffers()
{
    try
    {
        commandBuffers.clear();
        commandBuffers.reserve(swapChainImages.size());

        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(swapChainImages.size())
        };

        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
        return true;
    }
```

```45:76:src/Base/VulkanBase_commands.cpp
bool VulkanBase::createSyncObjects()
{
    try
    {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();

        const auto count = static_cast<uint32_t>(swapChainImages.size());
        presentCompleteSemaphores.reserve(count);
        renderFinishedSemaphores.reserve(count);
        inFlightFences.reserve(count);

        vk::SemaphoreCreateInfo semaphoreInfo{};
        for (uint32_t i = 0; i < count; i++)
        {
            presentCompleteSemaphores.emplace_back(device, semaphoreInfo);
            renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
        }

        vk::FenceCreateInfo fenceInfo{ .flags = vk::FenceCreateFlagBits::eSignaled };
        for (uint32_t i = 0; i < count; i++)
        {
            inFlightFences.emplace_back(device, fenceInfo);
        }
        return true;
    }
```

### 2.3 为什么不能只用 `MAX_FRAMES_IN_FLIGHT`

`MAX_FRAMES_IN_FLIGHT` 是一个**编译期常量**（固定为 2），而 `swapChainImages.size()` 是**运行时值**。在以下场景中两者不相等：

| 场景                                                                   | `swapChainImages.size()` | `MAX_FRAMES_IN_FLIGHT` | 后果                                              |
| ---------------------------------------------------------------------- | ------------------------ | ---------------------- | ------------------------------------------------- |
| 双缓冲（典型）                                                         | 2                        | 2                      | 相等，安全                                        |
| 三缓冲（高刷新率显示器）                                               | 3                        | 2                      | **越界访问**：用 `imageIndex=2` 访问 `[2]` 时崩溃 |
| `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` 返回的 `minImageCount` > 2 | ≥3                       | 2                      | 同上                                              |

如果用 `MAX_FRAMES_IN_FLIGHT` 分配数组，而 `imageIndex` 来自 `acquireNextImage`（范围是 `[0, swapChainImages.size())`），当 `swapChainImages.size() > MAX_FRAMES_IN_FLIGHT` 时：

```
std::vector<T> arr(MAX_FRAMES_IN_FLIGHT);  // size = 2
arr[imageIndex];                            // imageIndex = 2 → 越界！
```

这是 **未定义行为（UB）**。

---

## 3. `currentFrame` 的真正用途

`VulkanBase::currentFrame` 是用于**双缓冲轮转**的计数器，与 `MAX_FRAMES_IN_FLIGHT` 配合使用：

```cpp
currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
```

它保证 CPU 端最多同时有 `MAX_FRAMES_IN_FLIGHT` 帧在飞行。但**渲染目标的索引必须用 `imageIndex`**，不是 `currentFrame`：

```
acquireNextImage() → imageIndex    ← 用于所有渲染资源的索引
                        ↑
submit() 之后           │
currentFrame = (currentFrame+1) % MAX_FRAMES_IN_FLIGHT   ← 仅用于 CPU 端轮转
```

**常见错误**：在 `render()` 中同时使用两者：

```cpp
// ❌ 错误：混用 currentFrame 和 imageIndex
vk::SubmitInfo{
    .pCommandBuffers = &*commandBuffers[currentFrame],  // ← currentFrame
    .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]  // ← imageIndex
};
```

正确做法：**始终使用 `imageIndex`** 作为渲染资源的索引。

---

## 4. 全项目索引选择速查表

### 4.1 `swapChainImages.size()` 用于分配/索引的资源

#### VulkanBase（基类）

| 数组/容器                   | 用途                                         | 是否在 `recreateSwapChain` 中重建 |
| --------------------------- | -------------------------------------------- | --------------------------------- |
| `swapChainImages`           | 交换链 Image（源头）                         | —                                 |
| `swapChainImageViews`       | ImageView 列表                               | ✅                                 |
| `swapChainImageLayouts`     | 每张 Image 的 layout 状态                    | ✅                                 |
| `commandBuffers`            | 主渲染 CommandBuffer                         | ✅                                 |
| `presentCompleteSemaphores` | Image 获取完成信号                           | ✅                                 |
| `renderFinishedSemaphores`  | 渲染完成信号                                 | ✅                                 |
| `inFlightFences`            | CPU-GPU 同步围栏                             | ✅                                 |
| `uiFrameBuffers`            | ImGui Framebuffer（每 swapchain image 一个） | ✅                                 |

#### ClusteredRenderer（12_clustered）

| 数组/容器                        | 用途                        | 索引         | 是否在 `recreateSwapChain` 中重建 |
| -------------------------------- | --------------------------- | ------------ | --------------------------------- |
| `computeCommandBuffers`          | Compute 调度 CommandBuffer  | `imageIndex` | ✅                                 |
| `computeCompleteSemaphores`      | Compute→Graphics 同步信号   | `imageIndex` | ✅                                 |
| `computeFences`                  | Compute 完成围栏            | `imageIndex` | ✅                                 |
| `clusteredDescriptorSets`        | Graphics Pass DescriptorSet | `imageIndex` | ✅                                 |
| `computeDescriptorSets`          | Compute Pass DescriptorSet  | `imageIndex` | ✅                                 |
| `sceneUboResources.Buffers`      | CPU→GPU：相机 VP 矩阵       | `imageIndex` | ✅                                 |
| `clusterParamsResources.Buffers` | CPU→GPU：Cluster 参数       | `imageIndex` | ✅                                 |
| `groundUboResources.Buffers`     | CPU→GPU：地面 UBO           | `imageIndex` | ✅                                 |
| `lightBufferResources.Buffers`   | CPU→GPU：光源列表 SSBO      | `imageIndex` | ✅                                 |

> ClusteredRenderer 是三缓冲安全做得最完整的 Renderer —— Compute/Graphics 所有资源都按 `swapChainImages.size()` 分配，用 `imageIndex` 索引。

#### PostFXRenderer（PostFX）

| 数组/容器                              | 用途                                    | 索引         | 是否在 `recreateSwapChain` 中重建 |
| -------------------------------------- | --------------------------------------- | ------------ | --------------------------------- |
| `postBuffers`                          | HDR 场景 + Bloom 缓冲                   | `imageIndex` | ✅                                 |
| `blurBuffers`                          | Bloom 模糊 ping-pong 缓冲               | `imageIndex` | ✅                                 |
| `gbuffer.*.texture`                    | GBuffer 附件（每 swapchain image 一套） | `imageIndex` | ✅                                 |
| `instanceBufferResources.Buffers`      | CPU→GPU：场景/实例 UBO/SSBO             | `imageIndex` | ✅                                 |
| `lightUboResources.Buffers`            | CPU→GPU：光照/设置 UBO                  | `imageIndex` | ✅                                 |
| `deferredSettingsUboResources.Buffers` | CPU→GPU：后处理设置 UBO                 | `imageIndex` | ✅                                 |
| `postFxSettingsUboResources.Buffers`   | CPU→GPU：PostFX 参数 UBO                | `imageIndex` | ✅                                 |

#### 其他 Renderer（资源分配统一用 `MAX_FRAMES_IN_FLIGHT`，但正确场景下安全）

| Renderer         | 资源类型                     | 分配用 `MAX_FRAMES_IN_FLIGHT` |
| ---------------- | ---------------------------- | ----------------------------- |
| 所有 Renderer    | `MeshBuffer::descriptorSets` | ✅ `MAX_FRAMES_IN_FLIGHT`      |
| ParticleRenderer | `computeCommandBuffers`      | ✅ `MAX_FRAMES_IN_FLIGHT`      |
| ParticleRenderer | `computeCompleteSemaphores`  | ✅ `MAX_FRAMES_IN_FLIGHT`      |
| CullingRenderer  | `cullingCommandBuffers`      | ✅ `MAX_FRAMES_IN_FLIGHT`      |
| CullingRenderer  | `cullingCompleteSemaphores`  | ✅ `MAX_FRAMES_IN_FLIGHT`      |

> **重要**：`ParticleRenderer` 和 `CullingRenderer` 的 Compute 资源数组用 `MAX_FRAMES_IN_FLIGHT` 分配，但 `recordCommandBuffer` 和 `render()` 中用 `currentFrame` 索引。在双缓冲下（`swapChainImages.size() == 2 == MAX_FRAMES_IN_FLIGHT`）安全；在三缓冲下存在理论越界风险。

---

### 4.2 `MAX_FRAMES_IN_FLIGHT` 用于分配/索引的资源

#### VulkanBase（基类）

| 数组/容器                                | 用途                | 分配位置                          |
| ---------------------------------------- | ------------------- | --------------------------------- |
| `createUniformBuffers(count)` 无参数重载 | Mesh UBO（基础版）  | `VulkanBase_Resource.cpp:52`      |
| `createStorageBuffers(count)` 无参数重载 | Mesh SSBO（基础版） | `VulkanBase_Resource.cpp:88, 106` |

#### 所有 Renderer 的通用模式

| 数组/容器                                                 | 用途                 | 典型代码                                    |
| --------------------------------------------------------- | -------------------- | ------------------------------------------- |
| `MeshBuffer::Buffers` / `BuffersMemory` / `BuffersMapped` | 双缓冲 GPU 资源      | `createUniformBuffers(MeshBuffer&, size)`   |
| `MeshBuffer::descriptorSets`                              | 双缓冲 DescriptorSet | `layouts.assign(MAX_FRAMES_IN_FLIGHT, ...)` |
| `DescriptorPool::maxSets`                                 | 描述符池容量         | `.maxSets = MAX_FRAMES_IN_FLIGHT`           |

#### 各 Renderer 的 CPU→GPU 数据上传

| Renderer      | 上传函数                                                                     | 索引             |
| ------------- | ---------------------------------------------------------------------------- | ---------------- |
| IBL_pbr       | `updatePBRInstanceBuffers(currentFrame)`                                     | `currentFrame` ✅ |
| Instanced     | `updateInstancedBuffers(currentFrame)`                                       | `currentFrame` ✅ |
| PBR           | `updatePBRInstanceBuffers(currentFrame)`                                     | `currentFrame` ✅ |
| Shadow        | `updateShadowBuffers(currentFrame)`                                          | `currentFrame` ✅ |
| SSR           | `updateShadowBuffers(currentFrame)`, `updateSSRBuffers(currentFrame)`        | `currentFrame` ✅ |
| TAAU          | `updateTAAUBuffers(currentFrame)`, `updateShadowBuffers(currentFrame)`       | `currentFrame` ✅ |
| Deferred      | `updateDeferredBuffers(currentFrame)`                                        | `currentFrame` ✅ |
| Clustered     | `updateClusterBuffers(currentFrame)`                                         | `currentFrame` ✅ |
| PostFX        | `updateDeferredBuffers(imageIndex)`, `updatePostSettingsBuffers(imageIndex)` | `imageIndex` ✅   |
| Particle      | `updateParticleBuffers(currentFrame)`                                        | `currentFrame` ✅ |
| GI            | `updateBuffers(currentFrame)`                                                | `currentFrame` ✅ |
| Multithreaded | `updateFrameData(currentFrame)`, `updateInstanceBuffer(currentFrame)`        | `currentFrame` ✅ |

---

### 4.3 描述符集布局创建统一用 `MAX_FRAMES_IN_FLIGHT`

所有 Renderer 的 `DescriptorSetLayout` 创建逻辑高度一致：

```cpp
// 所有 Renderer（VulkanBase + 所有子 Renderer）
std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *someDescriptorSetLayout);
vk::DescriptorSetAllocateInfo{
    .descriptorPool = *somePool,
    .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
    .pSetLayouts = layouts.data()
};
someResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);
```

这是**安全的**，因为：
1. `DescriptorSet` 由 `DescriptorPool` 分配，池的 `maxSets` 也是 `MAX_FRAMES_IN_FLIGHT`
2. 描述符集在 `render()` 时用 `currentFrame` 索引（绑定到命令缓冲时）
3. 双缓冲下 `currentFrame ∈ {0,1}` 与 `swapChainImages.size() == 2` 恰好对齐

---

### 4.4 `currentFrame` vs `imageIndex` 使用边界

```
render() 函数内：
┌──────────────────────────────────────────────────────────────┐
│ VulkanBase 基类（所有 Renderer 通用）                          │
│  waitForFences(inFlightFences[currentFrame])   ← currentFrame│
│  acquireNextImage(presentCompleteSemaphores[currentFrame])   │
│                                                              │
│  CPU 数据准备（UBO/SSBO 写入）                                │
│  updateXxxBuffers(currentFrame)               ← currentFrame│
│                                                              │
│  recordCommandBuffer(imageIndex)                             │
│   ├─ commandBuffers[currentFrame]             ← currentFrame│
│   └─ swapChainImages[imageIndex]              ← imageIndex   │
│   └─ swapChainImageViews[imageIndex]           ← imageIndex  │
│   └─ descriptorSets[imageIndex]               ← imageIndex  │
│                                                              │
│  submit()                                                    │
│  waitSemaphores[presentCompleteSemaphores[currentFrame]]     │
│  signalSemaphores[renderFinishedSemaphores[imageIndex]]     │
│  fence = inFlightFences[currentFrame]                       │
│                                                              │
│  presentQueue.presentKHR(renderFinishedSemaphores[imageIndex])│
│                                                              │
│  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT    │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. `uiDescriptorSets` 的特殊性

`uiDescriptorSets`（`VulkanBase` 中定义）在基类 `VulkanBase_UI.cpp` 中只分配 **1 个**：

```src/Base/VulkanBase_UI.cpp
vk::DescriptorSetAllocateInfo{
    .descriptorPool = *self->uiDescriptorPool,
    .descriptorSetCount = 1,          // ← 不是 swapChainImages.size()
    .pSetLayouts = &*self->uiDescriptorSetLayout
};
self->uiDescriptorSets = vk::raii::DescriptorSets(self->device, allocInfo);
```

所有子 Renderer 调用 `recordUICmdBuffer(commandBuffer, frameIndex)` 时，基类内部始终用 `[0]` 索引：

```src/Base/VulkanBase_UI.cpp
*self->uiDescriptorSets[0]  // ← 始终为 0
```

这意味着 UI 的字体纹理 sampler 只绑定到索引 0 的描述符集。在三缓冲场景下：
- `imageIndex = 0,1,2` 均通过 `frameIndex` 参数传入
- 但 `uiDescriptorSets[0]` 对所有帧都相同
- 这是基类设计的一个已知限制：所有帧共用同一个 UI 描述符集（字体 sampler 是只读的，`imageIndex` 无关紧要）

---

## 6. 总结与设计原则

### 何时用 `swapChainImages.size()`

1. **与交换链图像有一一对应关系**的资源（CommandBuffer、Semaphore、Fence、ImageView、Framebuffer）
2. **与特定 `imageIndex` 强绑定**的资源（渲染附件、后处理中间缓冲）
3. 所有在 `recreateSwapChain()` 中需要重建的资源

### 何时用 `MAX_FRAMES_IN_FLIGHT`

1. **CPU 数据双缓冲**（UBO/SSBO 的 CPU 端写入轮转）
2. **DescriptorSet 布局分配**（因双缓冲与描述符池容量一致）
3. **描述符集绑定**（用 `currentFrame` 索引，与 CPU 写入顺序对齐）

### 设计约束

- `MAX_FRAMES_IN_FLIGHT` 是编译期常量，表示 CPU 端轮转深度
- `swapChainImages.size()` 是运行时值，由驱动决定
- **所有与 `acquireNextImage` 返回的 `imageIndex` 配对的资源，必须用 `swapChainImages.size()` 分配**
- **所有与 CPU 端轮转 `currentFrame` 配对的资源，必须用 `MAX_FRAMES_IN_FLIGHT` 分配**
- 在双缓冲（典型场景）下两者值相等，混用不会暴露问题；在三缓冲下只有严格区分才安全

---

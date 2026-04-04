# VulkanBase 数组大小设计：swapChainImages.size() vs MAX_FRAMES_IN_FLIGHT

[返回目录](../../README.md)

---

## 1. 问题背景

`VulkanBase` 中所有与帧相关的数组同时存在两个概念：

| 概念 | 值 | 含义 |
|---|---|---|
| `MAX_FRAMES_IN_FLIGHT` | 2 | CPU 可同时"在飞"的帧数（CPU 提交后不等 GPU 完成） |
| `swapChainImages.size()` | 动态（通常 2~3） | 交换链图像数量，由驱动/硬件决定 |

两者在 Vulkan 渲染器中常被混淆，但含义不同。

---

## 2. 为什么基类用 `swapChainImages.size()`

### 2.1 一一对应关系

交换链中的每张图像（SwapChain Image）有且只有 **一个** 渲染目标。正确的一一对应如下：

```
swapChainImages[0] ←→ presentCompleteSemaphores[0] ←→ renderFinishedSemaphores[0] ←→ inFlightFences[0] ←→ commandBuffers[0]
swapChainImages[1] ←→ presentCompleteSemaphores[1] ←→ renderFinishedSemaphores[1] ←→ inFlightFences[1] ←→ commandBuffers[1]
swapChainImages[2] ←→ presentCompleteSemaphores[2] ←→ renderFinishedSemaphores[2] ←→ inFlightFences[2] ←→ commandBuffers[2]  (如果存在)
```

这意味着 `acquireNextImage` 返回的 `imageIndex` 索引可以直接用于访问所有这些数组。

### 2.2 代码证据

当前 `VulkanBase` 的实现（`VulkanBase_commands.cpp`）已经使用 `swapChainImages.size()`：

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

| 场景 | `swapChainImages.size()` | `MAX_FRAMES_IN_FLIGHT` | 后果 |
|---|---|---|---|
| 双缓冲（典型） | 2 | 2 | 相等，安全 |
| 三缓冲（高刷新率显示器） | 3 | 2 | **越界访问**：用 `imageIndex=2` 访问 `[2]` 时崩溃 |
| `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` 返回的 `minImageCount` > 2 | ≥3 | 2 | 同上 |

如果用 `MAX_FRAMES_IN_FLIGHT` 分配数组，而 `imageIndex` 来自 `acquireNextImage`（范围是 `[0, swapChainImages.size())`），当 `swapChainImages.size() > MAX_FRAMES_IN_FLIGHT` 时：

```
std::vector<T> arr(MAX_FRAMES_IN_FLIGHT);  // size = 2
arr[imageIndex];                            // imageIndex = 2 → 越界！
```

这是 **未定义行为（UB）**，轻则数据错乱，重则程序崩溃。

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
submit() 之后             │
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

## 4. 子类中的修复（12_clustered 为例）

`ClusteredRenderer` 的原始代码在多处混淆了两个概念，导致越界。以下是修复原则：

### 4.1 所有数组必须与 `swapChainImages.size()` 对齐

| 资源类型 | 修复前 | 修复后 |
|---|---|---|
| `commandBuffers` | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |
| `computeCommandBuffers` | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |
| `computeCompleteSemaphores` | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |
| `presentCompleteSemaphores` | 基类已正确 | — |
| `renderFinishedSemaphores` | 基类已正确 | — |
| `inFlightFences` | 基类已正确 | — |
| `clusteredDescriptorSets` | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |
| `computeDescriptorSets` | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |
| `MeshBuffer.Buffers[]` (UBO/SSBO) | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |
| `uiFrameBuffers` | `MAX_FRAMES_IN_FLIGHT` | `swapChainImages.size()` |

### 4.2 索引统一使用 `imageIndex`

```cpp
// ✅ 正确：所有数组都用 imageIndex 索引
auto& cmdBuf = commandBuffers[imageIndex];
auto& computeCmdBuf = computeCommandBuffers[imageIndex];
auto& clusteredDS = clusteredDescriptorSets[imageIndex];
auto& computeDS = computeDescriptorSets[imageIndex];
auto& uiFB = uiFrameBuffers[imageIndex];
```

---

## 5. 结论

```
swapChainImages.size()  ← 用于分配所有渲染相关数组的大小
imageIndex              ← 用于索引这些数组（来自 acquireNextImage）
MAX_FRAMES_IN_FLIGHT    ← 仅用于 CPU 端帧轮转计数器 (currentFrame)
```

根本原因：**GPU 渲染的目标是 SwapChain Image，不是"第 N 帧"**。每张图像对应一套独立的资源（command buffer、semaphore、fence 等），所以数组大小必须与 `swapChainImages` 的数量一致。

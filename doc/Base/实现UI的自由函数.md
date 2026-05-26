# UI 的自由函数与 pImpl 惯用法

## 1. 什么是自由函数

**自由函数**（Free Function）是指不隶属于任何类的顶层函数（C 风格的过程式函数）。在 `VulkanBase_UI` 中，自由函数是 UI 功能实现的核心载体。

在 `VulkanBase_UI.cpp` 中可以看到这些自由函数：

- `vkrInitVulkanUI(VulkanBase* self)` — 初始化 ImGui context、字体纹理、Descriptor Set、Pipeline、FrameBuffers
- `vkrShutdownUI(VulkanBase* self)` — 销毁所有 UI 资源
- `vkrUpdateUIFrame(VulkanBase* self)` — 每帧调用：设置 ImGuiIO（DisplaySize、DeltaTime、鼠标）、触发 NewFrame → updateUIPanel() → Render
- `vkrReallocateUIMemory(VulkanBase* self, uint32_t frameIndex, const ImDrawData* drawData)` — 根据 ImGui DrawData 大小动态重分配顶点/索引缓冲
- `vkrRecordUICmdBuffer(VulkanBase* self, vk::CommandBuffer& cmdBuffer, uint32_t frameIndex)` — 将 ImDrawData 拷贝到 GPU buffer，按 DrawCmd 逐个设置 Scissor 并 drawIndexed

每个自由函数的第一个参数都是 `VulkanBase* self`，通过这个指针直接访问 `VulkanBase` 的私有成员（如 `device`、`commandPool`、`swapChainImages`、`uiPipeline` 等）。

## 2. 自由函数 + inline 成员包装器模式

`VulkanBase_UI` 的实现采用**自由函数 + inline 成员包装器**的组合模式：

```
自由函数（VulkanBase_UI.cpp）     inline 成员包装器（VulkanBase.h）
────────────────────────────     ────────────────────────────────
vkrInitVulkanUI(self)      ←     inline bool initVulkanUI()
vkrShutdownUI(self)        ←     inline void shutdownVulkanUI()
vkrUpdateUIFrame(self)     ←     inline void updateUIFrame()
vkrReallocateUIMemory(...) ←     inline void reallocateUIMemory(...)
vkrRecordUICmdBuffer(...)  ←     inline void recordUICmdBuffer(...)
```

`VulkanBase` 中的成员函数是 `inline` 包装器，直接转发调用到对应的自由函数。子类（如 `PBRRenderer`、`ShadowRenderer`）无需了解内部实现，只需要调用成员函数即可：

```cpp
// 子类只需要调用成员函数
void PBRRenderer::render() {
    updateUIFrame(currentFrame);         // 内部调用 vkrUpdateUIFrame(this)
    recordCommandBuffer(imageIndex);
}

void PBRRenderer::recordCommandBuffer(uint32_t imageIndex) {
    // ... 主渲染 pass ...
    recordUICmdBuffer(commandBuffer, imageIndex); // 内部调用 vkrRecordUICmdBuffer
}
```

**这种设计的好处**：

1. **实现与接口分离** — 自由函数的实现集中在 `.cpp` 中，不污染头文件，隐藏了 `VulkanBase` 的内部细节
2. **统一入口** — 子类调用成员函数即可，感知不到自由函数的存在
3. **避免过度耦合** — 自由函数可以直接访问 `VulkanBase` 的 private 成员，而 public 接口保持简洁

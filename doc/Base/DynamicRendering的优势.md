# Dynamic Rendering 的优势

## 1. 概述

本项目全面采用 Vulkan 1.3 Core 引入的 **Dynamic Rendering**（动态渲染）替代传统的 `RenderPass` + `Framebuffer` 模式。从 Vulkan 1.3 起，`vkCmdBeginRendering` / `vkCmdEndRendering` 成为 Core 特性，不再是 `VK_KHR_dynamic_rendering` 扩展。

---

## 2. 核心机制

### 2.1 设备特性启用

`VulkanBase_core.cpp` — `pickPhysicalDevice()` 和 `createLogicalDevice()` 中强制要求设备支持 Vulkan 1.3：

```300:310:src/Base/VulkanBase_core.cpp
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
            featureChain = {
                {.features = {.samplerAnisotropy = true}},
                {.shaderDrawParameters = VK_TRUE },
                {.synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE },
                {.extendedDynamicState = VK_TRUE}
        };
```

选择逻辑也要求 `dynamicRendering == true` 才认为设备合格：

```220:225:src/Base/VulkanBase_core.cpp
            bool supportsRequiredFeatures =
                features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
```

### 2.2 Pipeline 创建 — 无 RenderPass

传统模式：先创建 `RenderPass` → 再创建 `Pipeline` 时传入 `renderPass`。
动态渲染模式：Pipeline 创建时 `renderPass = nullptr`，改为通过 `PipelineRenderingCreateInfo` 附加到 `pNext` 链：

```222:244:src/pbr/PBRRenderer.cpp
        const vk::Format depthFormat = findDepthFormat();
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
            {
                .stageCount = 2,
                .pStages = shaderStages,
                ...
                .layout = pbrPipelineLayout,
                .renderPass = nullptr              // 不再传入 RenderPass
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        pbrPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
```

### 2.3 渲染命令 — beginRendering / endRendering

录制阶段通过 `vk::RenderingInfo` 动态指定本次渲染使用的附件：

```345:361:src/pbr/PBRRenderer.cpp
    const vk::RenderingInfo renderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(...);
    commandBuffer.bindPipeline(...);
    commandBuffer.drawIndexed(...);
    commandBuffer.endRendering();
```

---

## 3. 核心优势

### 3.1 消除 RenderPass / Framebuffer 对象

| 对比项           | 传统模式                              | 动态渲染                                    |
| ---------------- | ------------------------------------- | ------------------------------------------- |
| 附件数量         | 编译时固定                            | 每帧可变（通过 `RenderingInfo`）            |
| GBuffer 布局变更 | 需要重建 RenderPass + 所有 Pipeline   | 仅改 `PipelineRenderingCreateInfo` 格式数组 |
| SwapChain 重设   | 需要重建所有 RenderPass + Framebuffer | 仅改 `beginRendering` 的 extent             |
| 临时附件         | 需要提前创建完整 RenderPass           | 运行时按需指定                              |

以 `PostFXRenderer` 子项目为例例 — 同一帧内有 5 种完全不同的附件组合（GBuffer 4 MRT、Lighting 1 color、BloomExtract、BloomBlur、Composite），传统模式需要 5 个不同的 RenderPass 对象，而动态渲染只需 5 份不同的 `vk::RenderingInfo`。

### 3.2 天然适配 GBuffer 多目标渲染（MRT）

PostFXRenderer 的 GBuffer Pass 用 4 个颜色附件 + 1 个深度附件：

```1365:1381:src/postfx/PostFXRenderer.cpp
    vk::RenderingInfo gbufferRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(gbufferColorAttachments.size()),  // 4
        .pColorAttachments = gbufferColorAttachments.data(),
        .pDepthAttachment = &gbufferDepthAttachment
    };

    commandBuffer.beginRendering(gbufferRenderingInfo);
    // ... 一次 drawIndexed 输出到 albedo / normal / material / depth
    commandBuffer.endRendering();
```

创建 Pipeline 时同样通过 `PipelineRenderingCreateInfo` 指定 4 个颜色格式：

```398:401:src/postfx/PostFXRenderer.cpp
            {
                .colorAttachmentCount = static_cast<uint32_t>(gbufferColorFormats.size()),
                .pColorAttachmentFormats = gbufferColorFormats.data(),
                .depthAttachmentFormat = gbuffer.depth.format
            }
```

### 3.3 简化 SwapChain 的 Resize 

`recreateSwapChain` 只需重新创建 ImageView 和 Depth Image，所有 Pipeline 无需重建（因为无 RenderPass 依赖）：

- `PostFXRenderer::recreateGBufferSizedResources()` — 只重建 GBuffer Image/ImageView 和 DescriptorSet，Pipeline 保持不变
- `VulkanBase::createDepthResources()` — 重建深度图像，Pipeline 通过 `PipelineRenderingCreateInfo` 重新引用

对比传统模式：需要销毁 → 重建 `RenderPass` → 重建所有 `Framebuffer` → 重建所有 `Pipeline`。

### 3.4 联动 Dynamic State 

本项目同时启用了 `VK_EXT_extended_dynamic_state`：

```302:303:src/clustered/ClusteredRenderer.cpp
        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ ... };
```

这意味着 Viewport 和 Scissor 也在录制时动态设置，不写入 Pipeline。Pipeline 与渲染目标完全解耦，任意尺寸的附件均可复用同一 Pipeline。

---

## 4. 与传统 RenderPass 模式的完整对比

### 4.1 对象数量

以 PostFXRenderer 为例，传统模式需要：

| 对象类型        | 数量                            |
| --------------- | ------------------------------- |
| `vkRenderPass`  | 5（对应 5 个 Pass）             |
| `vkFramebuffer` | 5 × `MAX_FRAMES_IN_FLIGHT` = 10 |
| `vkPipeline`    | 仍然相同                        |

动态渲染模式：`vkRenderPass = 0`，`vkFramebuffer = 0`（通过 `beginRendering` 隐式引用 ImageView）。

### 4.2 代码模板对比

**传统模式 Pipeline 创建**：

```cpp
vk::RenderPass renderPass = /* 创建 RenderPass */;
vk::GraphicsPipelineCreateInfo createInfo{
    ...
    .renderPass = renderPass,   // 传入 RenderPass
    ...
};
```

**动态渲染 Pipeline 创建**：

```cpp
vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
    {
        .renderPass = nullptr,  // 不传入 renderPass
        ...
    },
    {
        .colorAttachmentCount = N,
        .pColorAttachmentFormats = formats,
        .depthAttachmentFormat = depthFormat
    }
};
vk::Pipeline pipeline = device.createGraphicsPipeline(nullptr, chain.get<GraphicsPipelineCreateInfo>());
```

### 4.3 渲染命令对比

**传统模式**：

```cpp
vk::RenderPassBeginInfo renderPassInfo{
    .renderPass = renderPass,
    .framebuffer = framebuffer,
    ...
};
cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
cmdBuffer.setViewport(...);
cmdBuffer.bindPipeline(...);
cmdBuffer.drawIndexed(...);
cmdBuffer.endRenderPass();
```

**动态渲染模式**：

```cpp
vk::RenderingInfo renderingInfo{
    .colorAttachmentCount = N,
    .pColorAttachments = attachments,
    .pDepthAttachment = &depthAttach,
};
cmdBuffer.beginRendering(renderingInfo);
cmdBuffer.setViewport(...);   // 动态设置
cmdBuffer.bindPipeline(...);
cmdBuffer.drawIndexed(...);
cmdBuffer.endRendering();
```

---

## 5. 多 Pass 渲染中的优势（PostFXRenderer 案例）

PostFXRenderer 每帧执行 5 个渲染 Pass，每个 Pass 使用完全不同的附件布局：

| Pass          | 颜色附件数                         | 深度附件 | 动态渲染优势                        |
| ------------- | ---------------------------------- | -------- | ----------------------------------- |
| GBuffer       | 4（albedo/normal/material/未定义） | ✅        | 4 MRT 一次完成                      |
| Lighting      | 1                                  | ❌        | 仅输出到临时缓冲区                  |
| BloomExtract  | 1                                  | ❌        | 复用 Pipeline，仅改 `RenderingInfo` |
| BloomBlur H/V | 1                                  | ❌        | 同一 Pipeline，H/V 坐标互换         |
| Composite     | 1                                  | ❌        | 从临时缓冲区读回 SwapChain          |

动态渲染的优势体现为：

1. **Pipeline 复用**：GBuffer/Lighting/Bloom/Composite 各用 1 个 Pipeline（用 5 次），传统模式需要 5 个 RenderPass + 多个 Framebuffer。
2. **附件自由组合**：`beginRendering` 可在任何时候为任何 Pass 指定任意格式/数量的附件。
3. **无需 subpass**：多 subpass 间的隐式 layout 转换被显式的 `transition_image_layout` 替代（本项目手动管理每 Pass 前的 barrier）。

---

## 6. 总结

| 优势                        | 说明                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------- |
| **无需 RenderPass 对象**    | 无需预创建 `vkRenderPass`，减少资源管理复杂度                                         |
| **附件数量在运行时决定**    | GBuffer 的 4 MRT、PostFX 的多种 Pass 组合无需为每种组合创建 RenderPass                |
| **简化 SwapChain Resize**   | Pipeline 无 RenderPass 依赖，仅重建 Image/ImageView 和 DescriptorSet                  |
| **Pipeline 格式与录制解耦** | `PipelineRenderingCreateInfo` 在创建时指定格式，`beginRendering` 在录制时指定实际附件 |
| **简化多 Pass 管理**        | 每个 Pass 通过不同的 `RenderingInfo` 声明附件，无需 subpass 依赖链                    |

 `vk::StructureChain` 讲 `GraphicsPipelineCreateInfo` 与 `PipelineRenderingCreateInfo` 进行组合，分离了 **Pipeline 创建** 与 **渲染目标声明** ，是现代 Vulkan 渲染器的推荐架构。

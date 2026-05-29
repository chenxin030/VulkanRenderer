#include "VolumetricRenderer.h"

#include <Base/Mesh.h>
#include <Base/VulkanBase_UI.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <imgui.h>

struct InstanceData
{
    glm::mat4 model;
    glm::vec4 color;
};

struct SceneUBO
{
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec3 camPos;
    float pad0;
};

struct VolumetricUBOParams
{
    float density;
    float scattering;
    float absorption;
    float anisotropicG;
    float stepSize;
    float maxDistance;
    float temporalFactor;
    float shadowStrength;
    float intensity;
    float nearZ;
    float farZ;
    glm::vec2 renderScale;
    glm::vec2 jitter;
    float historyValid;
};

#include <glm/gtc/matrix_transform.hpp>

void VolumetricRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool VolumetricRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 5.0f, 15.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -20.0f);
    return VulkanBase::initVulkan("VulkanRenderer - 14_volumetric");
}

bool VolumetricRenderer::prepareResource()
{
    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    generateCube(planeMesh);
    createVertexBuffer(planeMesh);
    createIndexBuffer(planeMesh);

    createSceneBuffers();
    if (!createSceneDescriptorSetLayout()) return false;
    if (!createSceneDescriptorPool()) return false;
    createSceneDescriptorSets();
    if (!createScenePipeline()) return false;

    if (!createVolumetricResources()) return false;
    if (!createVolumetricDescriptorSetLayout()) return false;
    if (!createVolumetricDescriptorPool()) return false;
    createVolumetricDescriptorSets();
    if (!createVolumetricPipeline()) return false;

    if (!initUI()) return false;

    return true;
}

bool VolumetricRenderer::initUI()
{
    return initVulkanUI();
}

void VolumetricRenderer::createSceneBuffers()
{
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));

    instanceCount = 20;
    createStorageBuffers(instanceBufferResources, sizeof(InstanceData) * instanceCount,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer);
}

bool VolumetricRenderer::createSceneDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        };

        sceneDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create scene descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool VolumetricRenderer::createSceneDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        };

        sceneDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create scene descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void VolumetricRenderer::createSceneDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *sceneDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *sceneDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    sceneDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceBufferInfo{ .buffer = *instanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(InstanceData) * instanceCount };

        std::vector<vk::WriteDescriptorSet> writes = {
            {.dstSet = *sceneDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            {.dstSet = *sceneDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
        };

        device.updateDescriptorSets(writes, nullptr);
    }
}

bool VolumetricRenderer::createScenePipeline()
{
    try
    {
        auto vertShaderCode = readFile(VK_SHADERS_DIR "volumetric_scene_vert.spv");
        auto fragShaderCode = readFile(VK_SHADERS_DIR "volumetric_scene_frag.spv");

        vk::raii::ShaderModule vertShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = vertShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data()) });
        vk::raii::ShaderModule fragShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = fragShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data()) });

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertShaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragShaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        scenePipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*sceneDescriptorSetLayout
            });

        std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
            {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex },
            {.binding = 1, .stride = sizeof(InstanceData), .inputRate = vk::VertexInputRate::eInstance },
        };

        std::vector<vk::VertexInputAttributeDescription> attributes = {
            {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = 0 },
            {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = 12 },
            {.location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = 24 },
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size()),
            .pVertexBindingDescriptions = vertexInputBindings.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        vk::Format sceneColorFormat = vk::Format::eR32G32B32A32Sfloat;

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
            {
                .stageCount = 2,
                .pStages = shaderStages,
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = scenePipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &sceneColorFormat
            }
        };

        scenePipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create scene pipeline: " << e.what() << std::endl;
        return false;
    }
}

void VolumetricRenderer::updateSceneBuffers(uint32_t currentFrame)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };

    std::memcpy(sceneUboResources.BuffersMapped[currentFrame], &sceneUbo, sizeof(SceneUBO));

    auto* instances = static_cast<InstanceData*>(instanceBufferResources.BuffersMapped[currentFrame]);

    for (uint32_t i = 0; i < instanceCount; i++)
    {
        float x = (i % 5) * 4.0f - 8.0f;
        float z = (i / 5) * 4.0f - 8.0f;
        float y = 1.0f;

        instances[i].model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) * glm::scale(glm::mat4(1.0f), glm::vec3(1.5f));

        float hue = static_cast<float>(i) / static_cast<float>(instanceCount);
        instances[i].color = glm::vec4(hue, 0.7f, 0.9f, 1.0f);
    }
}

void VolumetricRenderer::render()
{
    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    updateUIFrame();

    updateSceneBuffers(currentFrame);

    // Recreate volumetric resources if render scale changed
    if (volumetricRenderScale != lastVolumetricRenderScale)
    {
        device.waitIdle();
        recreateVolumetricSizedResources();
        lastVolumetricRenderScale = volumetricRenderScale;
        volumetricHistoryValid = false;
    }

    updateVolumetricBuffers(currentFrame);
    recordCommandBuffer(imageIndex);

    const vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
    };
    graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };

    try
    {
        result = presentQueue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    if (result == vk::Result::eSuboptimalKHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VolumetricRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    uint32_t volWidth = static_cast<uint32_t>(swapChainExtent.width * volumetricRenderScale);
    uint32_t volHeight = static_cast<uint32_t>(swapChainExtent.height * volumetricRenderScale);

    // Transition volumetric color for rendering
    transition_image_layout(commandBuffer, volumetricColorData.textureImage, volumetricColorLayout, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    volumetricColorLayout = vk::ImageLayout::eColorAttachmentOptimal;

    // ========== Pass 1: Scene Rendering to Volumetric Buffers ==========
    vk::RenderingAttachmentInfo sceneColorAttachment{
        .imageView = volumetricColorData.textureImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
    };

    vk::RenderingInfo sceneRenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = {volWidth, volHeight}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &sceneColorAttachment
    };

    commandBuffer.beginRendering(sceneRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(volWidth), static_cast<float>(volHeight), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(volWidth, volHeight)));

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *scenePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *scenePipelineLayout, 0, *sceneDescriptorSets[currentFrame], nullptr);

    // Draw floor
    commandBuffer.bindVertexBuffers(0, *planeMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*planeMesh.indexBuffer, 0, vk::IndexType::eUint16);
    commandBuffer.drawIndexed(static_cast<uint32_t>(planeMesh.indices.size()), 1, 0, 0, 0);

    // Draw cubes
    commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
    commandBuffer.bindVertexBuffers(1, *instanceBufferResources.Buffers[currentFrame], { 0 });
    commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexType::eUint16);
    commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), instanceCount, 0, 0, 0);

    commandBuffer.endRendering();

    // ========== Pass 2: Volumetric Ray Marching ==========
    // Transition volumetric color to shader read for sampling
    transition_image_layout(commandBuffer, volumetricColorData.textureImage, volumetricColorLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
    volumetricColorLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    // Transition history texture for sampling
    uint32_t historyReadIndex = (currentFrame + 1) % 2;
    vk::ImageLayout oldLayout = volumetricHistoryLayouts[historyReadIndex];
    if (oldLayout == vk::ImageLayout::eUndefined)
    {
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = vk::AccessFlags2{},
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = volumetricHistoryData[historyReadIndex].textureImage,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }
        };
        vk::DependencyInfo dependency_info{ .dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
        commandBuffer.pipelineBarrier2(dependency_info);
    }
    else
    {
        transition_image_layout(commandBuffer, volumetricHistoryData[historyReadIndex].textureImage, oldLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
            {}, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
    }
    volumetricHistoryLayouts[historyReadIndex] = vk::ImageLayout::eShaderReadOnlyOptimal;
    volumetricHistoryValid = true;

    // Transition velocity texture for sampling
    if (volumetricVelocityLayout == vk::ImageLayout::eUndefined)
    {
        vk::ImageMemoryBarrier2 velocityBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = vk::AccessFlags2{},
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = volumetricVelocityData.textureImage,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }
        };
        vk::DependencyInfo velocityDepInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &velocityBarrier };
        commandBuffer.pipelineBarrier2(velocityDepInfo);
        volumetricVelocityLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    updateVolumetricDescriptorSet(currentFrame, historyReadIndex);

    // Transition swap chain image to color attachment
    transition_image_layout(commandBuffer, swapChainImages[imageIndex], swapChainImageLayouts[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    // ========== Pass 2: Volumetric Ray Marching ==========
    vk::RenderingAttachmentInfo volumetricColorAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
    };

    vk::RenderingInfo volumetricRenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &volumetricColorAttachment
    };

    commandBuffer.beginRendering(volumetricRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *volumetricPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *volumetricPipelineLayout, 0, *volumetricDescriptorSets[currentFrame], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    // ========== Pass 3: UI Rendering ==========
    // UI rendering - need to transition back to color attachment optimal
    transition_image_layout(swapChainImages[imageIndex], swapChainImageLayouts[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    vk::RenderingAttachmentInfo uiAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };

    vk::RenderingInfo uiRenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachmentInfo
    };

    commandBuffer.beginRendering(uiRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUICmdBuffer(commandBuffer, currentFrame);
    commandBuffer.endRendering();

    // Transition for presentation
    transition_image_layout(swapChainImages[imageIndex], swapChainImageLayouts[imageIndex], vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite, {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eBottomOfPipe, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;

    commandBuffer.end();
}

void VolumetricRenderer::cleanup()
{
    device.waitIdle();
    shutdownVulkanUI();
}

void VolumetricRenderer::recreateSwapChain()
{
    VulkanBase::recreateSwapChain();

    recreateVolumetricSizedResources();
    volumetricHistoryValid = false;
}

void VolumetricRenderer::recreateVolumetricSizedResources()
{
    // Destroy old offscreen resources
    volumetricColorData.textureImageView = vk::raii::ImageView(nullptr);
    volumetricColorData.textureImage = vk::raii::Image(nullptr);
    volumetricColorData.textureImageMemory = vk::raii::DeviceMemory(nullptr);

    volumetricVelocityData.textureImageView = vk::raii::ImageView(nullptr);
    volumetricVelocityData.textureImage = vk::raii::Image(nullptr);
    volumetricVelocityData.textureImageMemory = vk::raii::DeviceMemory(nullptr);

    for (int i = 0; i < 2; ++i)
    {
        volumetricHistoryData[i].textureImageView = vk::raii::ImageView(nullptr);
        volumetricHistoryData[i].textureImage = vk::raii::Image(nullptr);
        volumetricHistoryData[i].textureImageMemory = vk::raii::DeviceMemory(nullptr);
        volumetricHistoryLayouts[i] = vk::ImageLayout::eUndefined;
    }

    // Destroy old descriptor sets and pool
    volumetricDescriptorSets = vk::raii::DescriptorSets(nullptr);
    volumetricDescriptorPool = vk::raii::DescriptorPool(nullptr);

    // Recreate at new size
    if (!createVolumetricResources())
    {
        throw std::runtime_error("failed to recreate volumetric resources");
    }
    if (!createVolumetricDescriptorPool())
    {
        throw std::runtime_error("failed to recreate volumetric descriptor pool");
    }
    createVolumetricDescriptorSets();
}

bool VolumetricRenderer::createVolumetricResources()
{
    uint32_t width = static_cast<uint32_t>(swapChainExtent.width * volumetricRenderScale);
    uint32_t height = static_cast<uint32_t>(swapChainExtent.height * volumetricRenderScale);

    createUniformBuffers(volumetricParamsUboResources, sizeof(VolumetricUBOParams));

    // Volumetric color (half-res) - stores view position and depth
    volumetricColorData.mipLevels = 1;
    createImage(width, height, 1, vk::Format::eR32G32B32A32Sfloat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, volumetricColorData);
    volumetricColorData.textureImageView = createImageView(volumetricColorData.textureImage, vk::Format::eR32G32B32A32Sfloat, vk::ImageAspectFlagBits::eColor, 1);

    // History buffers (full-res)
    for (int i = 0; i < 2; ++i)
    {
        volumetricHistoryData[i].mipLevels = 1;
        createImage(width, height, 1, vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, volumetricHistoryData[i]);
        volumetricHistoryData[i].textureImageView = createImageView(volumetricHistoryData[i].textureImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1);
    }

    // Velocity buffer
    volumetricVelocityData.mipLevels = 1;
    createImage(width, height, 1, vk::Format::eR32G32Sfloat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, volumetricVelocityData);
    volumetricVelocityData.textureImageView = createImageView(volumetricVelocityData.textureImage, vk::Format::eR32G32Sfloat, vk::ImageAspectFlagBits::eColor, 1);

    // Create sampler
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::False,
        .maxAnisotropy = 1.0f,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatOpaqueBlack,
        .unnormalizedCoordinates = vk::False
    };
    defaultSampler = vk::raii::Sampler(device, samplerInfo);

    volumetricColorLayout = vk::ImageLayout::eUndefined;
    volumetricHistoryLayouts[0] = vk::ImageLayout::eUndefined;
    volumetricHistoryLayouts[1] = vk::ImageLayout::eUndefined;
    volumetricVelocityLayout = vk::ImageLayout::eUndefined;

    return true;
}

bool VolumetricRenderer::createVolumetricDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };

        volumetricDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create volumetric descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool VolumetricRenderer::createVolumetricDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3 },
        };

        volumetricDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create volumetric descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void VolumetricRenderer::createVolumetricDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *volumetricDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *volumetricDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    volumetricDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);
}

void VolumetricRenderer::updateVolumetricDescriptorSet(uint32_t frameIndex, uint32_t historyReadIndex)
{
    vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *volumetricParamsUboResources.Buffers[frameIndex], .offset = 0, .range = sizeof(VolumetricUBOParams) };
    vk::DescriptorImageInfo sceneColorInfo{ .sampler = *defaultSampler, .imageView = *volumetricColorData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    vk::DescriptorImageInfo historyImageInfo{ .sampler = *defaultSampler, .imageView = *volumetricHistoryData[historyReadIndex].textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    vk::DescriptorImageInfo velocityImageInfo{ .sampler = *defaultSampler, .imageView = *volumetricVelocityData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

    std::vector<vk::WriteDescriptorSet> writes = {
        {.dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo },
        {.dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &historyImageInfo },
        {.dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &sceneColorInfo },
        {.dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &velocityImageInfo },
    };

    device.updateDescriptorSets(writes, nullptr);
}

bool VolumetricRenderer::createVolumetricPipeline()
{
    try
    {
        auto vertShaderCode = readFile(VK_SHADERS_DIR "volumetric_vert.spv");
        auto fragShaderCode = readFile(VK_SHADERS_DIR "volumetric_frag.spv");

        vk::raii::ShaderModule vertShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = vertShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data()) });
        vk::raii::ShaderModule fragShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = fragShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data()) });

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertShaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragShaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        volumetricPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*volumetricDescriptorSetLayout
            });

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount = 0, .vertexAttributeDescriptionCount = 0 };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eOne,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eOne,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
            {
                .stageCount = 2,
                .pStages = shaderStages,
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = volumetricPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat
            }
        };

        volumetricPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create volumetric pipeline: " << e.what() << std::endl;
        return false;
    }
}

void VolumetricRenderer::updateVolumetricBuffers(uint32_t currentFrame)
{
    volumetricJitterCurrent = glm::vec2(halton(volumetricFrameCounter, 2), halton(volumetricFrameCounter, 3)) - 0.5f;

    VolumetricUBOParams params{};
    params.density = volumetricParams.density;
    params.scattering = volumetricParams.scattering;
    params.absorption = volumetricParams.absorption;
    params.anisotropicG = volumetricParams.anisotropicG;
    params.stepSize = volumetricParams.stepSize;
    params.maxDistance = volumetricParams.maxDistance;
    params.temporalFactor = volumetricParams.temporalFactor;
    params.shadowStrength = volumetricParams.shadowStrength;
    params.intensity = volumetricParams.intensity;
    params.nearZ = 0.1f;
    params.farZ = 100.0f;
    params.renderScale = glm::vec2(volumetricRenderScale);
    params.jitter = volumetricJitterCurrent;
    params.historyValid = volumetricHistoryValid ? 1.0f : 0.0f;

    std::memcpy(volumetricParamsUboResources.BuffersMapped[currentFrame], &params, sizeof(VolumetricUBOParams));

    updateVolumetricHistory(glm::mat4(1.0f));
    volumetricFrameCounter++;
}

void VolumetricRenderer::recordVolumetric(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    (void)commandBuffer;
    (void)imageIndex;
    // Placeholder for Phase 1 implementation
}

void VolumetricRenderer::updateVolumetricHistory(const glm::mat4& currentViewProj)
{
    volumetricPrevViewProj = currentViewProj;
    volumetricJitterPrev = volumetricJitterCurrent;
}

void VolumetricRenderer::updateUIPanel()
{
    ImGui::Begin("Volumetric Lighting");

    ImGui::Checkbox("Enable Volumetric", &volumetricEnabled);

    if (ImGui::CollapsingHeader("Ray Marching Parameters", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Density", &volumetricParams.density, 0.01f, 1.0f);
        ImGui::SliderFloat("Scattering", &volumetricParams.scattering, 0.1f, 2.0f);
        ImGui::SliderFloat("Absorption", &volumetricParams.absorption, 0.0f, 1.0f);
        ImGui::SliderFloat("Anisotropic G", &volumetricParams.anisotropicG, -0.9f, 0.9f);
        ImGui::SliderFloat("Step Size", &volumetricParams.stepSize, 0.1f, 2.0f);
        ImGui::SliderFloat("Max Distance", &volumetricParams.maxDistance, 10.0f, 100.0f);
    }

    if (ImGui::CollapsingHeader("Temporal Reprojection", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Temporal Factor", &volumetricParams.temporalFactor, 0.5f, 0.99f);
        ImGui::Checkbox("Freeze History", &volumetricFreezeHistory);
    }

    if (ImGui::CollapsingHeader("Visual Quality"))
    {
        ImGui::SliderFloat("Render Scale", &volumetricRenderScale, 0.25f, 1.0f);
        ImGui::SliderFloat("Intensity", &volumetricParams.intensity, 0.1f, 5.0f);
        ImGui::SliderFloat("Shadow Strength", &volumetricParams.shadowStrength, 0.0f, 1.0f);
    }

    ImGui::Text("Frame: %llu", volumetricFrameCounter);

    ImGui::End();
}

// =============================================================================
// Utils
// =============================================================================

float VolumetricRenderer::halton(uint32_t index, uint32_t base)
{
    float result = 0.0f;
    float f = 1.0f;
    while (index > 0)
    {
        f /= static_cast<float>(base);
        result += f * static_cast<float>(index % base);
        index /= base;
    }
    return result;
}

glm::vec2 VolumetricRenderer::hammersley(uint32_t i, uint32_t n)
{
    return glm::vec2(static_cast<float>(i) / static_cast<float>(n), halton(i, 2));
}

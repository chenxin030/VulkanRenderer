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

struct VolumetricParamsUBO
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
};

#include <glm/gtc/matrix_transform.hpp>

void VolumetricRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool VolumetricRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 5.0f, 15.0f));
    camera.pitch = -0.3f;  // Look down slightly
    return VulkanBase::initVulkan("VulkanRenderer - 14_volumetric");
}

bool VolumetricRenderer::prepareResource()
{
    // Generate scene geometry
    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    generatePlane(planeMesh, 50.0f, 50.0f);
    createVertexBuffer(planeMesh);
    createIndexBuffer(planeMesh);

    // Scene resources
    createSceneBuffers();
    if (!createSceneDescriptorSetLayout()) return false;
    if (!createSceneDescriptorPool()) return false;
    createSceneDescriptorSets();
    if (!createScenePipeline()) return false;

    // Volumetric resources (Phase 1)
    if (!createVolumetricResources()) return false;
    if (!createVolumetricDescriptorSetLayout()) return false;
    if (!createVolumetricDescriptorPool()) return false;
    createVolumetricDescriptorSets();
    if (!createVolumetricPipeline()) return false;

    // UI
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

    // Scene instances: cubes scattered around
    instanceCount = 20;
    createStorageBuffers(instanceBufferResources, sizeof(InstanceData) * instanceCount);
}

bool VolumetricRenderer::createSceneDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
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
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
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
            { .dstSet = *sceneDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *sceneDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
        };

        device.updateDescriptorSets(writes, nullptr);
    }
}

bool VolumetricRenderer::createScenePipeline()
{
    try
    {
        auto vertShaderCode = readFile(VK_SHADERS_DIR "14_volumetric/volumetric_vert.spv");
        auto fragShaderCode = readFile(VK_SHADERS_DIR "14_volumetric/volumetric_scene_frag.spv");

        vk::raii::ShaderModule vertShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = vertShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data()) });
        vk::raii::ShaderModule fragShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = fragShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data()) });

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertShaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragShaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
            { .binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex },
            { .binding = 1, .stride = sizeof(InstanceData), .inputRate = vk::VertexInputRate::eInstance },
        };

        auto vertexAttributes = Vertex::getAttributeDescriptions();
        std::vector<vk::VertexInputAttributeDescription> instanceAttributes = {
            { .location = 3, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = 0 },
            { .location = 4, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = 16 },
            { .location = 5, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = 32 },
            { .location = 6, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = 48 },
            { .location = 7, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = 64 },
        };

        std::vector<vk::VertexInputAttributeDescription> attributes = vertexAttributes;
        attributes.insert(attributes.end(), instanceAttributes.begin(), instanceAttributes.end());

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size()),
            .pVertexBindingDescriptions = vertexInputBindings.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = false };

        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{ .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .lineWidth = 1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1 };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = false, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{ .depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = vk::CompareOp::eLess };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*sceneDescriptorSetLayout };

        scenePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &colorBlending,
            .pDepthStencilState = &depthStencil,
            .layout = *scenePipelineLayout,
            .renderPass = *renderPass,
            .subpass = 0
        };

        scenePipeline = vk::raii::GraphicsPipeline(device, nullptr, pipelineInfo);
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
    SceneUBO sceneUbo{};
    sceneUbo.projection = camera.GetProjectionMatrix();
    sceneUbo.view = camera.GetViewMatrix();
    sceneUbo.camPos = camera.Position;

    std::memcpy(sceneUboResources.BuffersMapped[currentFrame], &sceneUbo, sizeof(SceneUBO));

    // Update instance data
    auto* instances = static_cast<InstanceData*>(instanceBufferResources.BuffersMapped[currentFrame]);

    // Generate cube positions in a grid pattern
    for (uint32_t i = 0; i < instanceCount; i++)
    {
        float x = (i % 5) * 4.0f - 8.0f;
        float z = (i / 5) * 4.0f - 8.0f;
        float y = 1.0f;

        instances[i].model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) * glm::scale(glm::mat4(1.0f), glm::vec3(1.5f));

        // Varied colors
        float hue = static_cast<float>(i) / static_cast<float>(instanceCount);
        instances[i].color = glm::vec4(hue, 0.7f, 0.9f, 1.0f);  // HSV colors
    }
}

void VolumetricRenderer::render()
{
    device.waitForFences(*inFlightFences[currentFrame], true, UINT64_MAX);
    device.resetFences(*inFlightFences[currentFrame]);

    uint32_t imageIndex = 0;
    try
    {
        auto [acqResult, acquiredImageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame]);
        imageIndex = acquiredImageIndex;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        recreateSwapChain();
        return;
    }

    updateUIFrame();

    updateSceneBuffers(currentFrame);
    updateVolumetricBuffers(currentFrame);

    commandBuffers[currentFrame].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[currentFrame]
    };

    graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

    try
    {
        auto presentResult = presentQueue.presentKHR(vk::PresentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[currentFrame],
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex
        });

        if (presentResult == vk::Result::eSuboptimalKHR)
        {
            recreateSwapChain();
        }
    }
    catch (const vk::OutOfDateKHRError&)
    {
        recreateSwapChain();
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VolumetricRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    vk::CommandBufferBeginInfo beginInfo{};
    commandBuffers[currentFrame].begin(beginInfo);

    // Render scene to swap chain
    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };

    vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = *renderPass,
        .framebuffer = *swapChainFramebuffers[imageIndex],
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()
    };

    renderPassInfo.renderArea = vk::Rect2D{ .offset = { 0, 0 }, .extent = swapChainExtent };

    commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport{ .width = static_cast<float>(swapChainExtent.width), .height = static_cast<float>(swapChainExtent.height), .minDepth = 0.0f, .maxDepth = 1.0f };
    commandBuffers[currentFrame].setViewport(0, viewport);

    vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = swapChainExtent };
    commandBuffers[currentFrame].setScissor(0, scissor);

    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *scenePipeline);
    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *scenePipelineLayout, 0, *sceneDescriptorSets[currentFrame], nullptr);

    // Draw floor
    commandBuffers[currentFrame].bindVertexBuffers(0, *planeMesh.vertexBuffer.buffer, { 0 });
    commandBuffers[currentFrame].bindIndexBuffer(*planeMesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    commandBuffers[currentFrame].drawIndexed(planeMesh.indexCount, 1, 0, 0, 0);

    // Draw cubes
    commandBuffers[currentFrame].bindVertexBuffers(0, *cubeMesh.vertexBuffer.buffer, { 0 });
    commandBuffers[currentFrame].bindVertexBuffers(1, *instanceBufferResources.Buffers[currentFrame], { 0 });
    commandBuffers[currentFrame].bindIndexBuffer(*cubeMesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    commandBuffers[currentFrame].drawIndexed(cubeMesh.indexCount, instanceCount, 0, 0, 0);

    commandBuffers[currentFrame].endRenderPass();

    // Render volumetric lighting (Phase 1+)
    if (volumetricEnabled)
    {
        recordVolumetric(commandBuffers[currentFrame], imageIndex);
    }

    // UI
    {
        vk::RenderPassBeginInfo uiRenderPassInfo{
            .renderPass = *uiRenderPass,
            .framebuffer = *uiFramebuffers[imageIndex],
        };
        uiRenderPassInfo.renderArea = vk::Rect2D{ .offset = { 0, 0 }, .extent = swapChainExtent };

        commandBuffers[currentFrame].beginRenderPass(uiRenderPassInfo, vk::SubpassContents::eInline);
        recordUI(commandBuffers[currentFrame]);
        commandBuffers[currentFrame].endRenderPass();
    }

    commandBuffers[currentFrame].end();
}

void VolumetricRenderer::cleanup()
{
    device.waitIdle();

    scenePipeline.reset();
    scenePipelineLayout.reset();
    sceneDescriptorPool.reset();
    sceneDescriptorSetLayout.reset();
    sceneUboResources.cleanup(device);
    instanceBufferResources.cleanup(device);
    cubeMesh.cleanup(device);
    planeMesh.cleanup(device);

    volumetricColorData.cleanup(device);
    volumetricDepthData.cleanup(device);
    volumetricHistoryData[0].cleanup(device);
    volumetricHistoryData[1].cleanup(device);
    volumetricVelocityData.cleanup(device);
    volumetricPipeline.reset();
    volumetricPipelineLayout.reset();
    volumetricDescriptorPool.reset();
    volumetricDescriptorSetLayout.reset();
    volumetricParamsUboResources.cleanup(device);

    cleanupVulkanUI();
}

void VolumetricRenderer::recreateSwapChain()
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    device.waitIdle();
    recreateVolumetricResources();
    VulkanBase::recreateSwapChain();

    recreateVolumetricResources();
}

// =============================================================================
// Volumetric Rendering (Phase 1+)
// =============================================================================

bool VolumetricRenderer::createVolumetricResources()
{
    uint32_t width = static_cast<uint32_t>(swapChainExtent.width * volumetricRenderScale);
    uint32_t height = static_cast<uint32_t>(swapChainExtent.height * volumetricRenderScale);

    createUniformBuffers(volumetricParamsUboResources, sizeof(VolumetricParamsUBO));

    volumetricColorData = createRenderTarget(width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    volumetricDepthData = createRenderTarget(width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    volumetricHistoryData[0] = createRenderTarget(width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    volumetricHistoryData[1] = createRenderTarget(width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
    volumetricVelocityData = createRenderTarget(width, height, vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);

    volumetricColorLayout = vk::ImageLayout::eUndefined;
    volumetricDepthLayout = vk::ImageLayout::eUndefined;
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
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
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
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3 },
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
    vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *volumetricParamsUboResources.Buffers[frameIndex], .offset = 0, .range = sizeof(VolumetricParamsUBO) };
    vk::DescriptorImageInfo historyImageInfo{ .sampler = *defaultSampler, .imageView = *volumetricHistoryData[historyReadIndex].textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    vk::DescriptorImageInfo depthImageInfo{ .sampler = *defaultSampler, .imageView = *volumetricDepthData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    vk::DescriptorImageInfo velocityImageInfo{ .sampler = *defaultSampler, .imageView = *volumetricVelocityData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

    std::vector<vk::WriteDescriptorSet> writes = {
        { .dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo },
        { .dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &historyImageInfo },
        { .dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthImageInfo },
        { .dstSet = *volumetricDescriptorSets[frameIndex], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &velocityImageInfo },
    };

    device.updateDescriptorSets(writes, nullptr);
}

bool VolumetricRenderer::createVolumetricPipeline()
{
    try
    {
        auto vertShaderCode = readFile(VK_SHADERS_DIR "14_volumetric/volumetric_vert.spv");
        auto fragShaderCode = readFile(VK_SHADERS_DIR "14_volumetric/volumetric_frag.spv");

        vk::raii::ShaderModule vertShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = vertShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data()) });
        vk::raii::ShaderModule fragShaderModule(device, vk::ShaderModuleCreateInfo{ .codeSize = fragShaderCode.size(), .pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data()) });

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = *vertShaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = *fragShaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount = 0, .vertexAttributeDescriptionCount = 0 };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = false };

        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{ .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .lineWidth = 1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1 };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = true, .srcColorBlendFactor = vk::BlendFactor::eOne, .dstColorBlendFactor = vk::BlendFactor::eOne, .colorBlendOp = vk::BlendOp::eAdd, .srcAlphaBlendFactor = vk::BlendFactor::eOne, .dstAlphaBlendFactor = vk::BlendFactor::eOne, .alphaBlendOp = vk::BlendOp::eAdd, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*volumetricDescriptorSetLayout };

        volumetricPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &colorBlending,
            .layout = *volumetricPipelineLayout,
            .renderPass = *renderPass,
            .subpass = 0
        };

        volumetricPipeline = vk::raii::GraphicsPipeline(device, nullptr, pipelineInfo);
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

    VolumetricParamsUBO params{};
    params.density = volumetricParams.density;
    params.scattering = volumetricParams.scattering;
    params.absorption = volumetricParams.absorption;
    params.anisotropicG = volumetricParams.anisotropicG;
    params.stepSize = volumetricParams.stepSize;
    params.maxDistance = volumetricParams.maxDistance;
    params.temporalFactor = volumetricParams.temporalFactor;
    params.shadowStrength = volumetricParams.shadowStrength;
    params.intensity = volumetricParams.intensity;
    params.nearZ = camera.GetNearPlane();
    params.farZ = camera.GetFarPlane();
    params.renderScale = glm::vec2(volumetricRenderScale);
    params.jitter = volumetricJitterCurrent;

    std::memcpy(volumetricParamsUboResources.BuffersMapped[currentFrame], &params, sizeof(VolumetricParamsUBO));

    uint32_t historyRead = volumetricHistoryReadIndex;
    uint32_t historyWrite = 1 - volumetricHistoryReadIndex;
    updateVolumetricDescriptorSet(currentFrame, historyRead);

    updateVolumetricHistory(camera.GetProjectionMatrix() * camera.GetViewMatrix());
    volumetricFrameCounter++;
}

void VolumetricRenderer::recordVolumetric(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    uint32_t width = static_cast<uint32_t>(swapChainExtent.width * volumetricRenderScale);
    uint32_t height = static_cast<uint32>(swapChainExtent.height * volumetricRenderScale);

    // Full-screen quad for volumetric rendering
    // Transition to color attachment layout
    transitionImageLayout(commandBuffer, *volumetricColorData.textureImage, vk::Format::eR16G16B16A16Sfloat, volumetricColorLayout, vk::ImageLayout::eColorAttachmentOptimal);
    volumetricColorLayout = vk::ImageLayout::eColorAttachmentOptimal;

    // Record volumetric pass to intermediate target
    vk::RenderPassBeginInfo volumetricRenderPassInfo{
        .renderPass = *renderPass,
        .framebuffer = *swapChainFramebuffers[imageIndex],
        .renderArea = vk::Rect2D{ .extent = { width, height } }
    };

    std::array<vk::ClearValue, 1> volumetricClearValues{};
    volumetricClearValues[0].color = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f };

    commandBuffer.beginRenderPass(volumetricRenderPassInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport{ .width = static_cast<float>(width), .height = static_cast<float>(height), .minDepth = 0.0f, .maxDepth = 1.0f };
    commandBuffer.setViewport(0, viewport);

    vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = { width, height } };
    commandBuffer.setScissor(0, scissor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *volumetricPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *volumetricPipelineLayout, 0, *volumetricDescriptorSets[currentFrame], nullptr);
    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRenderPass();

    // Transition to shader read for next frame
    transitionImageLayout(commandBuffer, *volumetricColorData.textureImage, vk::Format::eR16G16B16A16Sfloat, volumetricColorLayout, vk::ImageLayout::eShaderReadOnlyOptimal);
    volumetricColorLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    volumetricHistoryReadIndex = 1 - volumetricHistoryReadIndex;
}

void VolumetricRenderer::updateVolumetricHistory(const glm::mat4& currentViewProj)
{
    volumetricPrevViewProj = currentViewProj;
    volumetricJitterPrev = volumetricJitterCurrent;
}

void VolumetricRenderer::recreateVolumetricResources()
{
    if (!volumetricColorData.textureImage)
    {
        createVolumetricResources();
    }
    volumetricColorData = createRenderTarget(swapChainExtent.width, swapChainExtent.height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
}

// =============================================================================
// UI
// =============================================================================

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

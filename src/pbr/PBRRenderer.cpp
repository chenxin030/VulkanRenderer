#include "PBRRenderer.h"

#include <Base/VulkanBase_UI.h>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>

struct PBRInstanceData {
    glm::mat4 model;
    float metallic;
    float roughness;
    alignas(16) glm::vec3 color;
};

struct SceneUBO {
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec3 camPos;
};

struct PointLight {
    glm::vec4 position; // w is intensity or unused
    glm::vec4 color;    // w is intensity
};

struct LightUBO {
    PointLight lights[4];
};

void PBRRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool PBRRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, -1.0f, 15.0f));
    if (!VulkanBase::initVulkan("VulkanRenderer - 2_pbr")) return false;
    return true;
}

bool PBRRenderer::initUI()
{
    return initVulkanUI();
}

void PBRRenderer::updateUIPanel()
{
    ImGui::SetNextWindowSize(ImVec2(300.0f, 200.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("PBR Material Grid", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Text("7x7 sphere grid: metallic (X) vs roughness (Y)");
    ImGui::Spacing();

    ImGui::Separator();
    ImGui::Text("Instance Count");
    static int count = 49;
    if (ImGui::SliderInt("Count", &count, 1, 100)) {
        instanceCount = static_cast<uint32_t>(count);
    }

    ImGui::Separator();
    ImGui::Text("Point Lights");
    ImGui::Text("4 lights with animated motion");

    ImGui::End();
}

bool PBRRenderer::prepareResource()
{
    generateSphere(sphereMesh, 1.0f, 100);
    createVertexBuffer(sphereMesh);
    createIndexBuffer(sphereMesh);

    createPBRBuffers();
    if (!createPBRDescriptorSetLayout()) {
        std::cerr << "Failed to create PBR DescriptorSetLayout" << std::endl;
        return false;
    }
    if (!createPBRDescriptorPool()) {
        std::cerr << "Failed to create PBR DescriptorPool" << std::endl;
        return false;
    }
    if (!createPBRPipeline()) {
        std::cerr << "Failed to create PBR Pipeline" << std::endl;
        return false;
    }
    createPBRDescriptorSets();

    if (!initUI()) {
        std::cerr << "Failed to initialize UI" << std::endl;
        return false;
    }

    return true;
}

bool PBRRenderer::createPBRDescriptorSetLayout()
{
    try {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 2, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };

        pbrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        });
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create PBR descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool PBRRenderer::createPBRDescriptorPool()
{
    try {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u }, // scene + light
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        };

        pbrDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        });
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create PBR descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void PBRRenderer::createPBRDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pbrDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *pbrDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    pbrInstanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceBufferInfo{ .buffer = *pbrInstanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(PBRInstanceData) * instanceCount };
        vk::DescriptorBufferInfo lightBufferInfo{ .buffer = *lightUboResources.Buffers[i], .offset = 0, .range = sizeof(LightUBO) };

        std::vector<vk::WriteDescriptorSet> descriptorWrites = {
            { .dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
            { .dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &lightBufferInfo },
        };

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }
}

bool PBRRenderer::createPBRPipeline()
{
    try {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "pbr.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        const auto bindingDescription = Vertex::getBindingDescription();
        const auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

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
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        pbrPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*pbrDescriptorSetLayout,
            .pushConstantRangeCount = 0
        });

        const vk::Format depthFormat = findDepthFormat();
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
                .layout = pbrPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        pbrPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create PBR graphics pipeline: " << e.what() << std::endl;
        return false;
    }
}

void PBRRenderer::createPBRBuffers()
{
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(lightUboResources, sizeof(LightUBO));
    createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * instanceCount);
}

void PBRRenderer::updatePBRInstanceBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1;
    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    // Instances: 7*7 grid (49 instances).
    std::vector<PBRInstanceData> instances;
    instances.reserve(instanceCount);
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            glm::mat4 model(1.0f);
            model = glm::translate(model, glm::vec3(static_cast<float>(x) * 2.2f, static_cast<float>(y) * 2.2f, 0.0f));

            const float fx = static_cast<float>(x + 3) / 6.0f;
            const float fy = static_cast<float>(y + 3) / 6.0f;
            const float metallic = fx;
            const float roughness = std::clamp(fy, 0.04f, 1.0f);

            const glm::vec3 color(1.0f, 0.86f, 0.57f);
            instances.push_back(PBRInstanceData{ model, metallic, roughness, color });
        }
    }
    std::memcpy(pbrInstanceBufferResources.BuffersMapped[frameIndex], instances.data(), sizeof(PBRInstanceData) * instances.size());

    static auto startTime = std::chrono::high_resolution_clock::now();
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float>(currentTime - startTime).count();

    LightUBO lightUbo{};
    lightUbo.lights[0] = { .position = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 400.0f) };
    lightUbo.lights[1] = { .position = glm::vec4(-20.0f, -10.0f, 10.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 50.0f) };
    lightUbo.lights[2] = { .position = glm::vec4(std::sin(time * 0.5f) * 12.0f, 5.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 150.0f) };
    lightUbo.lights[3] = { .position = glm::vec4(0.0f, std::cos(time * 0.5f) * 12.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 150.0f) };
    std::memcpy(lightUboResources.BuffersMapped[frameIndex], &lightUbo, sizeof(lightUbo));
}

void PBRRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    transition_image_layout(
        swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(
        depthData.textureImage,
        depthImageLayout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    const vk::ClearValue clearColor = vk::ClearColorValue(0.2f, 0.2f, 0.2f, 1.0f);
    const vk::RenderingAttachmentInfo attachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };
    const vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    const vk::RenderingAttachmentInfo depthAttachmentInfo{
        .imageView = depthData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth
    };
    const vk::RenderingInfo renderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);
    commandBuffer.bindVertexBuffers(0, *sphereMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*sphereMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(sphereMesh.indices)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pbrPipelineLayout, 0, *pbrInstanceBufferResources.descriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(sphereMesh.indices.size()), instanceCount, 0, 0, 0);
    commandBuffer.endRendering();

    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUICmdBuffer(commandBuffer, currentFrame);

    transition_image_layout(
        swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;

    commandBuffer.end();
}

void PBRRenderer::render()
{
    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    updatePBRInstanceBuffers(currentFrame);
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

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

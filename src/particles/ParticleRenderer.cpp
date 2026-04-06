#include "ParticleRenderer.h"

#include <Base/Mesh.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

void ParticleRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool ParticleRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 5.0f, 25.0f));
    rng.seed(1337u);

    const bool ok = VulkanBase::initVulkan("VulkanRenderer - 14_particles");
    return ok;
}

bool ParticleRenderer::prepareResource()
{
    if (!createParticleBuffers()) return false;
    if (!createParticleDescriptorSetLayouts()) return false;
    if (!createParticleDescriptorPools()) return false;
    createParticleDescriptorSets();

    if (!createParticlePipeline()) return false;
    if (!createComputePipeline()) return false;

    if (!createComputeCommandPool()) return false;
    if (!createComputeCommandBuffers()) return false;
    if (!createComputeSyncObjects()) return false;

    if (!initUI()) return false;

    generateInitialParticles();
    return true;
}

bool ParticleRenderer::createParticleBuffers()
{
    try
    {
        createStorageBuffers(particleBufferResources, sizeof(Particle) * MAX_PARTICLES);
        createUniformBuffers(particleParamsBufferResources, sizeof(ParticleParamsUBO));
        createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create particle buffers: " << e.what() << std::endl;
        return false;
    }
}

bool ParticleRenderer::createParticleDescriptorSetLayouts()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> renderBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex }
        };

        particleDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(renderBindings.size()),
            .pBindings = renderBindings.data() });

        std::vector<vk::DescriptorSetLayoutBinding> computeBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute }
        };

        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(computeBindings.size()),
            .pBindings = computeBindings.data() });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create descriptor set layouts: " << e.what() << std::endl;
        return false;
    }
}

bool ParticleRenderer::createParticleDescriptorPools()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> renderPoolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
        };

        particleDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(renderPoolSizes.size()),
            .pPoolSizes = renderPoolSizes.data() });

        std::vector<vk::DescriptorPoolSize> computePoolSizes = {
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
        };

        computeDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(computePoolSizes.size()),
            .pPoolSizes = computePoolSizes.data() });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create descriptor pools: " << e.what() << std::endl;
        return false;
    }
}

void ParticleRenderer::createParticleDescriptorSets()
{
    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *particleDescriptorSetLayout);
        particleDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
            .descriptorPool = *particleDescriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data() });

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::DescriptorBufferInfo sceneInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
            vk::DescriptorBufferInfo particleInfo{ .buffer = *particleBufferResources.Buffers[i], .offset = 0, .range = sizeof(Particle) * MAX_PARTICLES };

            std::vector<vk::WriteDescriptorSet> writes = {
                { .dstSet = *particleDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
                { .dstSet = *particleDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &particleInfo }
            };
            device.updateDescriptorSets(writes, nullptr);
        }
    }

    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);
        computeDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
            .descriptorPool = *computeDescriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data() });

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::DescriptorBufferInfo particleInfo{ .buffer = *particleBufferResources.Buffers[i], .offset = 0, .range = sizeof(Particle) * MAX_PARTICLES };
            vk::DescriptorBufferInfo paramsInfo{ .buffer = *particleParamsBufferResources.Buffers[i], .offset = 0, .range = sizeof(ParticleParamsUBO) };

            std::vector<vk::WriteDescriptorSet> writes = {
                { .dstSet = *computeDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &particleInfo },
                { .dstSet = *computeDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo }
            };
            device.updateDescriptorSets(writes, nullptr);
        }
    }
}

bool ParticleRenderer::createParticlePipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "particle_render.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::ePointList,
            .primitiveRestartEnable = vk::False
        };

        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
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
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        particlePipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*particleDescriptorSetLayout,
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
                .layout = particlePipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        particlePipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create particle graphics pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool ParticleRenderer::createComputePipeline()
{
    try
    {
        vk::raii::ShaderModule computeShader = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "particle_update_comp.spv"));

        vk::PipelineShaderStageCreateInfo computeShaderStage{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = computeShader,
            .pName = "compMain"
        };

        computePipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*computeDescriptorSetLayout,
            .pushConstantRangeCount = 0
        });

        computePipeline = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{
            .stage = computeShaderStage,
            .layout = computePipelineLayout
        });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool ParticleRenderer::createComputeCommandPool()
{
    try
    {
        computeCommandPool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.computeFamily.value()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute command pool: " << e.what() << std::endl;
        return false;
    }
}

bool ParticleRenderer::createComputeCommandBuffers()
{
    try
    {
        computeCommandBuffers.clear();
        computeCommandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *computeCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };

        auto buffers = device.allocateCommandBuffers(allocInfo);
        for (auto& buffer : buffers)
        {
            computeCommandBuffers.emplace_back(std::move(buffer));
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to allocate compute command buffers: " << e.what() << std::endl;
        return false;
    }
}

bool ParticleRenderer::createComputeSyncObjects()
{
    try
    {
        computeCompleteSemaphores.clear();
        computeCompleteSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            computeCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute sync objects: " << e.what() << std::endl;
        return false;
    }
}

void ParticleRenderer::generateInitialParticles()
{
    for (uint32_t i = 0; i < MAX_PARTICLES; ++i)
    {
        Particle particle{};
        particle.position = glm::vec3(0.0f, 0.0f, 0.0f);
        particle.velocity = glm::vec3(dist11(rng) * spread, velocity + dist01(rng) * velocity, dist11(rng) * spread);
        particle.lifetime = dist01(rng) * particleLifetime;
        particle.maxLifetime = particleLifetime;
        particle.color = glm::vec4(1.0f, 0.5f, 0.1f, 1.0f);
        particle.size = particleSize;
        particle.padding = glm::vec3(0.0f);

        std::memcpy(static_cast<std::byte*>(particleBufferResources.BuffersMapped[0]) + i * sizeof(Particle), &particle, sizeof(Particle));
    }

    for (uint32_t frame = 1; frame < MAX_FRAMES_IN_FLIGHT; ++frame)
    {
        std::memcpy(
            particleBufferResources.BuffersMapped[frame],
            particleBufferResources.BuffersMapped[0],
            sizeof(Particle) * MAX_PARTICLES);
    }
}

void ParticleRenderer::updateParticleBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(
            glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f,
            1000.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1;
    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    ParticleParamsUBO params{
        .emitterPosition = glm::vec3(0.0f, 0.0f, 0.0f),
        .emissionRate = emissionRate,
        .gravity = glm::vec3(0.0f, -gravity, 0.0f),
        .particleLifetime = particleLifetime,
        .particleSize = particleSize,
        .turbulenceStrength = turbulenceStrength,
        .spread = spread,
        .velocity = velocity
    };
    std::memcpy(particleParamsBufferResources.BuffersMapped[frameIndex], &params, sizeof(params));
}

void ParticleRenderer::recordComputeCommandBuffer(uint32_t frameIndex)
{
    auto& commandBuffer = computeCommandBuffers[frameIndex];
    commandBuffer.begin({});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        *computePipelineLayout,
        0,
        *computeDescriptorSets[frameIndex],
        nullptr);

    const uint32_t groupCount = (MAX_PARTICLES + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE;
    commandBuffer.dispatch(groupCount, 1, 1);

    commandBuffer.end();
}

void ParticleRenderer::recordCommandBuffer(uint32_t imageIndex)
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

    const vk::ClearValue clearColor = vk::ClearColorValue(0.05f, 0.05f, 0.1f, 1.0f);
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
        .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height),
            0.0f,
            1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *particlePipeline);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *particlePipelineLayout,
        0,
        *particleDescriptorSets[currentFrame],
        nullptr);
    commandBuffer.draw(MAX_PARTICLES, 1, 0, 0);

    commandBuffer.endRendering();

    const vk::RenderingAttachmentInfo uiAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    const vk::RenderingInfo uiRenderingInfo{
        .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachmentInfo
    };

    commandBuffer.beginRendering(uiRenderingInfo);
    commandBuffer.setViewport(
        0,
        vk::Viewport(
            0.0f,
            0.0f,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height),
            0.0f,
            1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUI(commandBuffer);
    commandBuffer.endRendering();

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

void ParticleRenderer::render()
{
    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [acquireResult, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX,
        *presentCompleteSemaphores[currentFrame],
        nullptr);

    if (acquireResult == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    updateParticleBuffers(currentFrame);
    updateUIFrame();

    recordComputeCommandBuffer(currentFrame);
    computeQueue.submit(vk::SubmitInfo{
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = &*computeCommandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*computeCompleteSemaphores[currentFrame]
        }, nullptr);

    commandBuffers[currentFrame].reset();
    recordCommandBuffer(imageIndex);

    const std::array<vk::Semaphore, 2> waitSemaphores = {
        *presentCompleteSemaphores[currentFrame],
        *computeCompleteSemaphores[currentFrame]
    };
    const std::array<vk::PipelineStageFlags, 2> waitStages = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eVertexInput
    };
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages.data(),
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

    vk::Result presentResult = vk::Result::eSuccess;
    try
    {
        presentResult = presentQueue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    if (presentResult == vk::Result::eSuboptimalKHR ||
        presentResult == vk::Result::eErrorOutOfDateKHR ||
        framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

bool ParticleRenderer::initUI()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> uiBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment }
        };
        uiDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(uiBindings.size()),
            .pBindings = uiBindings.data() });

        std::vector<vk::DescriptorPoolSize> uiPoolSizes = {
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
        };
        uiDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(uiPoolSizes.size()),
            .pPoolSizes = uiPoolSizes.data() });

        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge
        };
        uiFontTexture.textureSampler = vk::raii::Sampler(device, samplerInfo);

        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->AddFontDefault();

        unsigned char* fontPixels = nullptr;
        int fontWidth = 0;
        int fontHeight = 0;
        io.Fonts->GetTexDataAsRGBA32(&fontPixels, &fontWidth, &fontHeight);

        uiFontTexture.mipLevels = 1;
        createImage(
            static_cast<uint32_t>(fontWidth),
            static_cast<uint32_t>(fontHeight),
            1,
            vk::Format::eR8G8B8A8Unorm,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            uiFontTexture);

        transitionImageLayout(
            uiFontTexture.textureImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            1);

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        const vk::DeviceSize uploadSize = static_cast<vk::DeviceSize>(fontWidth) * static_cast<vk::DeviceSize>(fontHeight) * 4u;

        createBuffer(
            uploadSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory);

        void* mappedFontData = stagingBufferMemory.mapMemory(0, uploadSize);
        std::memcpy(mappedFontData, fontPixels, static_cast<size_t>(uploadSize));
        stagingBufferMemory.unmapMemory();

        auto cmdBuffer = beginSingleTimeCommands();
        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = static_cast<uint32_t>(fontWidth),
            .bufferImageHeight = static_cast<uint32_t>(fontHeight),
            .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { static_cast<uint32_t>(fontWidth), static_cast<uint32_t>(fontHeight), 1 }
        };
        cmdBuffer->copyBufferToImage(*stagingBuffer, *uiFontTexture.textureImage, vk::ImageLayout::eTransferDstOptimal, region);
        endSingleTimeCommands(*cmdBuffer);

        transitionImageLayout(
            uiFontTexture.textureImage,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            1);

        uiFontTexture.textureImageView = createImageView(
            uiFontTexture.textureImage,
            vk::Format::eR8G8B8A8Unorm,
            vk::ImageAspectFlagBits::eColor,
            1);

        io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(0)));

        std::vector<vk::DescriptorSetLayout> uiLayouts(MAX_FRAMES_IN_FLIGHT, *uiDescriptorSetLayout);
        uiDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
            .descriptorPool = *uiDescriptorPool,
            .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
            .pSetLayouts = uiLayouts.data() });

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::DescriptorImageInfo imageInfo{
                .sampler = *uiFontTexture.textureSampler,
                .imageView = *uiFontTexture.textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::WriteDescriptorSet write{
                .dstSet = *uiDescriptorSets[i],
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo
            };
            device.updateDescriptorSets(write, nullptr);
        }

        vk::raii::ShaderModule uiShaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "imgui.spv"));
        vk::PipelineShaderStageCreateInfo uiVertShaderStage{ .stage = vk::ShaderStageFlagBits::eVertex, .module = uiShaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo uiFragShaderStage{ .stage = vk::ShaderStageFlagBits::eFragment, .module = uiShaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo uiShaderStages[] = { uiVertShaderStage, uiFragShaderStage };

        vk::VertexInputBindingDescription uiBindingDescription{
            .binding = 0,
            .stride = sizeof(ImDrawVert),
            .inputRate = vk::VertexInputRate::eVertex
        };
        std::array<vk::VertexInputAttributeDescription, 3> uiAttributeDescriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, uv)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(ImDrawVert, col))
        };
        vk::PipelineVertexInputStateCreateInfo uiVertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &uiBindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(uiAttributeDescriptions.size()),
            .pVertexAttributeDescriptions = uiAttributeDescriptions.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo uiInputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };

        vk::PipelineViewportStateCreateInfo uiViewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo uiRasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f
        };
        vk::PipelineMultisampleStateCreateInfo uiMultisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineDepthStencilStateCreateInfo uiDepthStencil{
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
            .depthCompareOp = vk::CompareOp::eAlways,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState uiColorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        vk::PipelineColorBlendStateCreateInfo uiColorBlending{ .attachmentCount = 1, .pAttachments = &uiColorBlendAttachment };

        std::vector uiDynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo uiDynamicState{
            .dynamicStateCount = static_cast<uint32_t>(uiDynamicStates.size()),
            .pDynamicStates = uiDynamicStates.data()
        };

        vk::PushConstantRange uiPushConstants{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset = 0,
            .size = 16u
        };
        uiPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*uiDescriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &uiPushConstants
        });

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> uiPipelineChain = {
            {
                .stageCount = 2,
                .pStages = uiShaderStages,
                .pVertexInputState = &uiVertexInputInfo,
                .pInputAssemblyState = &uiInputAssembly,
                .pViewportState = &uiViewportState,
                .pRasterizationState = &uiRasterizer,
                .pMultisampleState = &uiMultisampling,
                .pDepthStencilState = &uiDepthStencil,
                .pColorBlendState = &uiColorBlending,
                .pDynamicState = &uiDynamicState,
                .layout = uiPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat
            }
        };

        uiPipeline = vk::raii::Pipeline(device, nullptr, uiPipelineChain.get<vk::GraphicsPipelineCreateInfo>());

        uiFrameBuffers.clear();
        uiFrameBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            uiFrameBuffers.emplace_back(
                vk::raii::Buffer{ nullptr },
                vk::raii::DeviceMemory{ nullptr },
                nullptr,
                0,
                0,
                vk::raii::Buffer{ nullptr },
                vk::raii::DeviceMemory{ nullptr },
                nullptr,
                0,
                0);
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to initialize UI: " << e.what() << std::endl;
        return false;
    }
}

void ParticleRenderer::updateUIFrame()
{
    if (ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height));
    io.DeltaTime = platform->frameTimer > 0.0f ? platform->frameTimer : (1.0f / 60.0f);

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(platform->window, &mouseX, &mouseY);
    io.MousePos = ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
    io.MouseDown[0] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    io.MouseDown[1] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    io.MouseDown[2] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    ImGui::NewFrame();
    updateParticleUI();
    ImGui::Render();

    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->TotalVtxCount <= 0)
    {
        for (auto& frameBuffer : uiFrameBuffers)
        {
            frameBuffer.vertexSize = 0;
            frameBuffer.indexSize = 0;
        }
        return;
    }

    // Only touch resources for the frame we just synchronized with.
    // Updating/reallocating other in-flight frame buffers can destroy buffers still in GPU use.
    auto& frameBuffer = uiFrameBuffers[currentFrame];
    frameBuffer.vertexSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
    frameBuffer.indexSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

    if (frameBuffer.vertexBufferSize < frameBuffer.vertexSize)
    {
        if (frameBuffer.vertexMapped != nullptr)
        {
            frameBuffer.vertexBufferMemory.unmapMemory();
            frameBuffer.vertexMapped = nullptr;
        }
        frameBuffer.vertexBuffer = vk::raii::Buffer(nullptr);
        frameBuffer.vertexBufferMemory = vk::raii::DeviceMemory(nullptr);
        createBuffer(
            frameBuffer.vertexSize,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            frameBuffer.vertexBuffer,
            frameBuffer.vertexBufferMemory);
        frameBuffer.vertexMapped = frameBuffer.vertexBufferMemory.mapMemory(0, frameBuffer.vertexSize);
        frameBuffer.vertexBufferSize = frameBuffer.vertexSize;
    }

    if (frameBuffer.indexBufferSize < frameBuffer.indexSize)
    {
        if (frameBuffer.indexMapped != nullptr)
        {
            frameBuffer.indexBufferMemory.unmapMemory();
            frameBuffer.indexMapped = nullptr;
        }
        frameBuffer.indexBuffer = vk::raii::Buffer(nullptr);
        frameBuffer.indexBufferMemory = vk::raii::DeviceMemory(nullptr);
        createBuffer(
            frameBuffer.indexSize,
            vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            frameBuffer.indexBuffer,
            frameBuffer.indexBufferMemory);
        frameBuffer.indexMapped = frameBuffer.indexBufferMemory.mapMemory(0, frameBuffer.indexSize);
        frameBuffer.indexBufferSize = frameBuffer.indexSize;
    }

    if (!frameBuffer.vertexMapped)
    {
        frameBuffer.vertexMapped = frameBuffer.vertexBufferMemory.mapMemory(0, frameBuffer.vertexBufferSize);
    }
    if (!frameBuffer.indexMapped)
    {
        frameBuffer.indexMapped = frameBuffer.indexBufferMemory.mapMemory(0, frameBuffer.indexBufferSize);
    }

    uint8_t* vtxDst = static_cast<uint8_t*>(frameBuffer.vertexMapped);
    uint8_t* idxDst = static_cast<uint8_t*>(frameBuffer.indexMapped);
    for (int32_t cmdListIndex = 0; cmdListIndex < drawData->CmdListsCount; ++cmdListIndex)
    {
        const ImDrawList* cmdList = drawData->CmdLists[cmdListIndex];
        std::memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
        std::memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size * sizeof(ImDrawVert);
        idxDst += cmdList->IdxBuffer.Size * sizeof(ImDrawIdx);
    }

    frameBuffer.vertexBufferMemory.unmapMemory();
    frameBuffer.indexBufferMemory.unmapMemory();
    frameBuffer.vertexMapped = nullptr;
    frameBuffer.indexMapped = nullptr;
}

void ParticleRenderer::updateParticleUI()
{
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f));
    ImGui::SetNextWindowSize(ImVec2(300.0f, 280.0f));
    ImGui::Begin("Particle System", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::Text("GPU Particle System (100k particles)");
    ImGui::Separator();

    ImGui::SliderFloat("Emission Rate", &emissionRate, 100.0f, 50000.0f, "%.0f/s");
    ImGui::SliderFloat("Lifetime", &particleLifetime, 0.5f, 10.0f, "%.1f s");
    ImGui::SliderFloat("Particle Size", &particleSize, 0.01f, 1.0f, "%.2f");
    ImGui::SliderFloat("Gravity", &gravity, 0.0f, 30.0f, "%.1f");
    ImGui::SliderFloat("Velocity", &velocity, 0.5f, 20.0f, "%.1f");
    ImGui::SliderFloat("Spread", &spread, 0.1f, 5.0f, "%.1f");
    ImGui::SliderFloat("Turbulence", &turbulenceStrength, 0.0f, 5.0f, "%.1f");

    ImGui::Separator();
    ImGui::Text("WASD: Move Camera");
    ImGui::Text("Q/E: Up/Down");

    ImGui::End();
}

void ParticleRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
{
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->TotalVtxCount == 0)
    {
        return;
    }

    auto& frameBuffer = uiFrameBuffers[currentFrame];
    if (frameBuffer.vertexSize == 0 || frameBuffer.indexSize == 0)
    {
        return;
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *uiPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *uiPipelineLayout, 0, *uiDescriptorSets[currentFrame], nullptr);

    vk::DeviceSize offsets[] = { 0 };
    commandBuffer.bindVertexBuffers(0, *frameBuffer.vertexBuffer, offsets);
    commandBuffer.bindIndexBuffer(
        *frameBuffer.indexBuffer,
        0,
        sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

    struct UiPushConsts
    {
        glm::vec2 scale;
        glm::vec2 translate;
    };

    UiPushConsts pushConsts{};
    pushConsts.scale = glm::vec2(2.0f / float(drawData->DisplaySize.x), 2.0f / float(drawData->DisplaySize.y));
    pushConsts.translate = glm::vec2(-1.0f, -1.0f);
    commandBuffer.pushConstants(
        *uiPipelineLayout,
        vk::ShaderStageFlagBits::eVertex,
        0,
        vk::ArrayProxy<const UiPushConsts>(1, &pushConsts));

    int32_t globalVertexOffset = 0;
    uint32_t globalIndexOffset = 0;
    const ImVec2 clipOff = drawData->DisplayPos;
    const ImVec2 clipScale = ImVec2(1.0f, 1.0f);

    for (int32_t listIndex = 0; listIndex < drawData->CmdListsCount; ++listIndex)
    {
        const ImDrawList* cmdList = drawData->CmdLists[listIndex];
        uint32_t cmdListIndexOffset = 0;

        for (int32_t cmdIndex = 0; cmdIndex < cmdList->CmdBuffer.Size; ++cmdIndex)
        {
            const ImDrawCmd* cmd = &cmdList->CmdBuffer[cmdIndex];

            ImVec4 clipRect;
            clipRect.x = (cmd->ClipRect.x - clipOff.x) * clipScale.x;
            clipRect.y = (cmd->ClipRect.y - clipOff.y) * clipScale.y;
            clipRect.z = (cmd->ClipRect.z - clipOff.x) * clipScale.x;
            clipRect.w = (cmd->ClipRect.w - clipOff.y) * clipScale.y;

            if (clipRect.x < float(swapChainExtent.width) &&
                clipRect.y < float(swapChainExtent.height) &&
                clipRect.z >= 0.0f &&
                clipRect.w >= 0.0f)
            {
                vk::Rect2D scissor;
                scissor.offset.x = static_cast<int32_t>(clipRect.x > 0.0f ? clipRect.x : 0.0f);
                scissor.offset.y = static_cast<int32_t>(clipRect.y > 0.0f ? clipRect.y : 0.0f);
                float scissorWidth = clipRect.z - clipRect.x;
                float scissorHeight = clipRect.w - clipRect.y;
                if (scissorWidth < 0.0f) scissorWidth = 0.0f;
                if (scissorHeight < 0.0f) scissorHeight = 0.0f;
                scissor.extent.width = static_cast<uint32_t>(scissorWidth);
                scissor.extent.height = static_cast<uint32_t>(scissorHeight);

                commandBuffer.setScissor(0, scissor);
                commandBuffer.drawIndexed(cmd->ElemCount, 1, globalIndexOffset + cmdListIndexOffset, globalVertexOffset, 0);
            }

            cmdListIndexOffset += cmd->ElemCount;
        }

        globalIndexOffset += static_cast<uint32_t>(cmdList->IdxBuffer.Size);
        globalVertexOffset += cmdList->VtxBuffer.Size;
    }
}

void ParticleRenderer::cleanup()
{
    device.waitIdle();
    shutdownUI();

    computeCompleteSemaphores.clear();
    computeCommandBuffers.clear();
}

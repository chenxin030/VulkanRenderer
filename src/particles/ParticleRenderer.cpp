#include "ParticleRenderer.h"

#include <Base/Mesh.h>
#include <Base/VulkanBase_UI.h>

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
    recordUICmdBuffer(commandBuffer);
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
    return initVulkanUI();
}

void ParticleRenderer::updateUIPanel()
{
    ImGui::SetNextWindowSize(ImVec2(340.0f, 280.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Particles", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Separator();
    ImGui::Text("Emitter");
    ImGui::SliderFloat("Emission Rate", &emissionRate, 0.0f, 20000.0f);
    ImGui::SliderFloat("Particle Lifetime (s)", &particleLifetime, 0.5f, 10.0f);
    ImGui::SliderFloat("Velocity", &velocity, 0.5f, 30.0f);
    ImGui::SliderFloat("Spread", &spread, 0.1f, 5.0f);

    ImGui::Separator();
    ImGui::Text("Appearance");
    ImGui::SliderFloat("Particle Size", &particleSize, 0.01f, 1.0f);

    ImGui::Separator();
    ImGui::Text("Physics");
    ImGui::SliderFloat("Gravity", &gravity, 0.0f, 30.0f);
    ImGui::SliderFloat("Turbulence", &turbulenceStrength, 0.0f, 5.0f);

    ImGui::End();
}

void ParticleRenderer::cleanup()
{
    device.waitIdle();
    shutdownVulkanUI();

    computeCompleteSemaphores.clear();
    computeCommandBuffers.clear();
}

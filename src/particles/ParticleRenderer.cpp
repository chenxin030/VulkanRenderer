#include "ParticleRenderer.h"

#include <Base/Mesh.h>

#include <algorithm>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

static_assert(sizeof(ParticleRenderer::Particle) % 16 == 0, "Particle must be 16-byte aligned");
static_assert(sizeof(ParticleRenderer::ParticleParamsUBO) % 16 == 0, "ParticleParamsUBO must be 16-byte aligned");
static_assert(sizeof(ParticleRenderer::SceneUBO) % 16 == 0, "SceneUBO must be 16-byte aligned");

void ParticleRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool ParticleRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 5.0f, 25.0f));
    rng.seed(1337u);
    return VulkanBase::initVulkan("VulkanRenderer - 14_particles");
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
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *particleDescriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };
        particleDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
            vk::DescriptorBufferInfo particleBufferInfo{ .buffer = *particleBufferResources.Buffers[i], .offset = 0, .range = sizeof(Particle) * MAX_PARTICLES };

            std::vector<vk::WriteDescriptorSet> descriptorWrites = {
                { .dstSet = *particleDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
                { .dstSet = *particleDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &particleBufferInfo }
            };
            device.updateDescriptorSets(descriptorWrites, nullptr);
        }
    }

    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *computeDescriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };
        computeDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo particleBufferInfo{ .buffer = *particleBufferResources.Buffers[i], .offset = 0, .range = sizeof(Particle) * MAX_PARTICLES };
            vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *particleParamsBufferResources.Buffers[i], .offset = 0, .range = sizeof(ParticleParamsUBO) };

            std::vector<vk::WriteDescriptorSet> descriptorWrites = {
                { .dstSet = *computeDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &particleBufferInfo },
                { .dstSet = *computeDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo }
            };
            device.updateDescriptorSets(descriptorWrites, nullptr);
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

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::ePointSize };
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
        vk::raii::ShaderModule computeShader = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "particle_update.spv"));
        vk::PipelineShaderStageCreateInfo computeShaderStage{ .stage = vk::ShaderStageFlagBits::eCompute, .module = computeShader, .pName = "compMain" };

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
            .queueFamilyIndex = queueFamilyIndices.computeFamily.value(),
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer
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
        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *computeCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };
        computeCommandBuffers = device.allocateCommandBuffers(allocInfo);
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
        computeCompleteSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            computeCompleteSemaphores[i] = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{});
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
    for (uint32_t i = 0; i < MAX_PARTICLES; i++)
    {
        Particle p;
        p.position = glm::vec3(0.0f, 0.0f, 0.0f);
        p.velocity = glm::vec3(dist11(rng) * spread, velocity + dist01(rng) * velocity, dist11(rng) * spread);
        p.lifetime = dist01(rng) * particleLifetime;
        p.maxLifetime = particleLifetime;
        p.color = glm::vec4(1.0f, 0.5f, 0.1f, 1.0f);
        p.size = particleSize;
        p.padding = glm::vec3(0.0f);

        std::memcpy(particleBufferResources.BuffersMapped[0] + i * sizeof(Particle), &p, sizeof(Particle));
    }
    for (uint32_t f = 1; f < MAX_FRAMES_IN_FLIGHT; f++)
    {
        std::memcpy(particleBufferResources.BuffersMapped[f], particleBufferResources.BuffersMapped[0], sizeof(Particle) * MAX_PARTICLES);
    }
}

void ParticleRenderer::updateParticleBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 1000.0f),
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
    auto& cmdBuffer = computeCommandBuffers[frameIndex];
    cmdBuffer.begin({});

    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *computeDescriptorSets[frameIndex], nullptr);

    uint32_t groupCount = (MAX_PARTICLES + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE;
    cmdBuffer.dispatch(groupCount, 1, 1);

    cmdBuffer.end();
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
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.setPointSize(particleSize * 100.0f);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *particlePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *particlePipelineLayout, 0, *particleDescriptorSets[currentFrame], nullptr);
    commandBuffer.draw(MAX_PARTICLES, 1, 0, 0);
    commandBuffer.endRendering();

    recordUI(commandBuffer);

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

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);

    updateParticleBuffers(currentFrame);
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

    const vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*computeCompleteSemaphores[currentFrame],
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
    result = presentQueue.presentKHR(presentInfo);

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
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

        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .extent = { 1, 1, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .format = vk::Format::eR8G8B8A8Unorm,
            .tiling = vk::ImageTiling::eOptimal,
            .initialLayout = vk::ImageLayout::eUndefined,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled
        };
        uiFontTexture.textureImage = vk::raii::Image(device, imageInfo);
        vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(*uiFontTexture.textureImage);
        vk::MemoryAllocateInfo allocInfo{ .allocationSize = memReqs.size, .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal) };
        uiFontTexture.textureImageMemory = vk::raii::DeviceMemory(device, allocInfo);
        device.bindImageMemory(*uiFontTexture.textureImage, *uiFontTexture.textureImageMemory, 0);
        uiFontTexture.textureImageView = createImageView(uiFontTexture.textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, 1);

        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->AddFontDefault();

        unsigned char* fontPixels = nullptr;
        int fontWidth = 0, fontHeight = 0;
        io.Fonts->GetTexDataAsRGBA32(&fontPixels, &fontWidth, &fontHeight);

        vk::BufferCreateInfo bufferInfo{ .size = static_cast<vk::DeviceSize>(fontWidth * fontHeight * 4), .usage = vk::BufferUsageFlagBits::eTransferSrc };
        vk::raii::Buffer fontStagingBuffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements fontMemReqs = device.getBufferMemoryRequirements(*fontStagingBuffer);
        vk::MemoryAllocateInfo fontAllocInfo{ .allocationSize = fontMemReqs.size, .memoryTypeIndex = findMemoryType(fontMemReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent) };
        vk::raii::DeviceMemory fontStagingMemory = vk::raii::DeviceMemory(device, fontAllocInfo);
        device.bindBufferMemory(*fontStagingBuffer, *fontStagingMemory, 0);

        void* fontData = fontStagingMemory.mapMemory(0, bufferInfo.size);
        std::memcpy(fontData, fontPixels, static_cast<size_t>(bufferInfo.size));
        fontStagingMemory.unmapMemory();

        transitionImageLayout(uiFontTexture.textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
        auto cmdBuffer = beginSingleTimeCommands();
        vk::BufferImageCopy region{ .bufferOffset = 0, .bufferRowLength = static_cast<uint32_t>(fontWidth), .bufferImageHeight = static_cast<uint32_t>(fontHeight), .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, .imageOffset = {0, 0, 0}, .imageExtent = {static_cast<uint32_t>(fontWidth), static_cast<uint32_t>(fontHeight), 1} };
        cmdBuffer->copyBufferToImage(*fontStagingBuffer, *uiFontTexture.textureImage, vk::ImageLayout::eTransferDstOptimal, region);
        endSingleTimeCommands(*cmdBuffer);
        transitionImageLayout(uiFontTexture.textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

        io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(0)));

        std::vector<vk::DescriptorSetLayout> uiLayouts(MAX_FRAMES_IN_FLIGHT, *uiDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo uiAllocInfo{ .descriptorPool = *uiDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = uiLayouts.data() };
        uiDescriptorSets = vk::raii::DescriptorSets(device, uiAllocInfo);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorImageInfo imageInfo{ .sampler = *uiFontTexture.textureSampler, .imageView = *uiFontTexture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
            vk::WriteDescriptorSet write{ .dstSet = *uiDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &imageInfo };
            device.updateDescriptorSets(write, nullptr);
        }

        vk::raii::ShaderModule uiShaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "imgui.spv"));
        vk::PipelineShaderStageCreateInfo uiVertShaderStage{ .stage = vk::ShaderStageFlagBits::eVertex, .module = uiShaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo uiFragShaderStage{ .stage = vk::ShaderStageFlagBits::eFragment, .module = uiShaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo uiShaderStages[] = { uiVertShaderStage, uiFragShaderStage };

        vk::PipelineVertexInputStateCreateInfo uiVertexInputInfo{ .vertexBindingDescriptionCount = 0, .vertexAttributeDescriptionCount = 0 };
        vk::PipelineInputAssemblyStateCreateInfo uiInputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList };

        vk::PipelineViewportStateCreateInfo uiViewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo uiRasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.0f };
        vk::PipelineMultisampleStateCreateInfo uiMultisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1 };
        vk::PipelineColorBlendAttachmentState uiColorBlendAttachment{ .blendEnable = vk::True, .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha, .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha, .colorBlendOp = vk::BlendOp::eAdd, .srcAlphaBlendFactor = vk::BlendFactor::eOne, .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha, .alphaBlendOp = vk::BlendOp::eAdd, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo uiColorBlending{ .attachmentCount = 1, .pAttachments = &uiColorBlendAttachment };

        std::vector uiDynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo uiDynamicState{ .dynamicStateCount = static_cast<uint32_t>(uiDynamicStates.size()), .pDynamicStates = uiDynamicStates.data() };

        uiPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{ .setLayoutCount = 1, .pSetLayouts = &*uiDescriptorSetLayout });

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> uiPipelineChain = {
            {
                .stageCount = 2,
                .pStages = uiShaderStages,
                .pVertexInputState = &uiVertexInputInfo,
                .pInputAssemblyState = &uiInputAssembly,
                .pViewportState = &uiViewportState,
                .pRasterizationState = &uiRasterizer,
                .pMultisampleState = &uiMultisampling,
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

        uiFrameBuffers.resize(MAX_FRAMES_IN_FLIGHT);
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
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height));

    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->TotalVtxCount == 0)
    {
        for (auto& fb : uiFrameBuffers)
        {
            fb.vertexSize = 0;
            fb.indexSize = 0;
        }
        return;
    }

    for (int32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        auto& fb = uiFrameBuffers[i];
        fb.vertexSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
        fb.indexSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

        if (!fb.vertexBuffer || fb.vertexBufferSize < fb.vertexSize)
        {
            fb.vertexBuffer = nullptr;
            fb.vertexBufferMemory = nullptr;
            vk::BufferCreateInfo vbInfo{ .size = fb.vertexSize, .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer };
            fb.vertexBuffer = vk::raii::Buffer(device, vbInfo);
            vk::MemoryRequirements vbMemReqs = device.getBufferMemoryRequirements(*fb.vertexBuffer);
            vk::MemoryAllocateInfo vbAllocInfo{ .allocationSize = vbMemReqs.size, .memoryTypeIndex = findMemoryType(vbMemReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible) };
            fb.vertexBufferMemory = vk::raii::DeviceMemory(device, vbAllocInfo);
            device.bindBufferMemory(*fb.vertexBuffer, *fb.vertexBufferMemory, 0);
            fb.vertexBufferSize = fb.vertexSize;
            fb.vertexMapped = nullptr;
        }

        if (!fb.indexBuffer || fb.indexBufferSize < fb.indexSize)
        {
            fb.indexBuffer = nullptr;
            fb.indexBufferMemory = nullptr;
            vk::BufferCreateInfo ibInfo{ .size = fb.indexSize, .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer };
            fb.indexBuffer = vk::raii::Buffer(device, ibInfo);
            vk::MemoryRequirements ibMemReqs = device.getBufferMemoryRequirements(*fb.indexBuffer);
            vk::MemoryAllocateInfo ibAllocInfo{ .allocationSize = ibMemReqs.size, .memoryTypeIndex = findMemoryType(ibMemReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible) };
            fb.indexBufferMemory = vk::raii::DeviceMemory(device, ibAllocInfo);
            device.bindBufferMemory(*fb.indexBuffer, *fb.indexBufferMemory, 0);
            fb.indexBufferSize = fb.indexSize;
            fb.indexMapped = nullptr;
        }

        if (!fb.vertexMapped)
        {
            fb.vertexMapped = fb.vertexBufferMemory.mapMemory(0, fb.vertexBufferSize);
        }
        if (!fb.indexMapped)
        {
            fb.indexMapped = fb.indexBufferMemory.mapMemory(0, fb.indexBufferSize);
        }

        uint8_t* vtxDst = static_cast<uint8_t*>(fb.vertexMapped);
        uint8_t* idxDst = static_cast<uint8_t*>(fb.indexMapped);
        for (int n = 0; n < drawData->CmdListsCount; n++)
        {
            const ImDrawList* cmdList = drawData->CmdLists[n];
            std::memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
            std::memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
            vtxDst += cmdList->VtxBuffer.Size * sizeof(ImDrawVert);
            idxDst += cmdList->IdxBuffer.Size * sizeof(ImDrawIdx);
        }

        fb.vertexBufferMemory.unmapMemory();
        fb.indexBufferMemory.unmapMemory();
        fb.vertexMapped = nullptr;
        fb.indexMapped = nullptr;
    }
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
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->TotalVtxCount == 0) return;

    auto& fb = uiFrameBuffers[currentFrame];
    if (fb.vertexSize == 0 || fb.indexSize == 0) return;

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *uiPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *uiPipelineLayout, 0, *uiDescriptorSets[currentFrame], nullptr);

    vk::DeviceSize offsets[] = { 0 };
    commandBuffer.bindVertexBuffers(0, *fb.vertexBuffer, offsets);
    commandBuffer.bindIndexBuffer(*fb.indexBuffer, 0, vk::IndexType::eUint16);

    ImVec2 displayPos = drawData->DisplayPos;
    int32_t idxOffset = 0;
    for (int32_t n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        for (int32_t i = 0; i < cmdList->CmdBuffer.Size; i++)
        {
            const ImDrawCmd* cmd = &cmdList->CmdBuffer[i];
            ImVec4 clipRect;
            clipRect.x = cmd->ClipRect.x - displayPos.x;
            clipRect.y = cmd->ClipRect.y - displayPos.y;
            clipRect.z = cmd->ClipRect.z - displayPos.x;
            clipRect.w = cmd->ClipRect.w - displayPos.y;

            if (clipRect.x < swapChainExtent.width && clipRect.y < swapChainExtent.height && clipRect.z >= 0.0f && clipRect.w >= 0.0f)
            {
                vk::Rect2D scissor{
                    .offset = { .x = static_cast<int32_t>(std::max(0.0f, clipRect.x)), .y = static_cast<int32_t>(std::max(0.0f, clipRect.y)) },
                    .extent = { .width = static_cast<uint32_t>(clipRect.z - clipRect.x), .height = static_cast<uint32_t>(clipRect.w - clipRect.y) }
                };
                commandBuffer.setScissor(0, scissor);
                commandBuffer.drawIndexed(cmd->ElemCount, 1, idxOffset, 0, 0);
            }
            idxOffset += cmd->ElemCount;
        }
    }
}

void ParticleRenderer::cleanup()
{
    device.waitIdle();
    shutdownUI();

    computePipeline = nullptr;
    computePipelineLayout = nullptr;
    computeDescriptorPool = nullptr;
    computeDescriptorSetLayout = nullptr;
    computeCommandPool = nullptr;

    particlePipeline = nullptr;
    particlePipelineLayout = nullptr;
    particleDescriptorPool = nullptr;
    particleDescriptorSetLayout = nullptr;
}

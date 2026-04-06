#include "MultithreadedRenderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

void MultithreadedRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool MultithreadedRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 5.0f, 30.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 15_multithreaded");
}

bool MultithreadedRenderer::prepareResource()
{
    applyPreset(currentPreset);

    loadModel("viking_room.glb", mesh);
    createVertexBuffer(mesh);
    createIndexBuffer(mesh);

    LoadTextureFromFile("viking_room.png", texture);
    createTextureSampler(texture.textureSampler);

    if (!createSceneBuffers()) return false;
    if (!createDescriptors()) return false;
    if (!createPipeline()) return false;
    if (!createSecondaryCommandResources()) return false;
    updateDescriptorSets();

    if (!initFrameGraph()) return false;
    if (!initThreading()) return false;
    if (!initUI()) return false;
    return true;
}

bool MultithreadedRenderer::createSceneBuffers()
{
    try
    {
        createUniformBuffers(globalUboResources, sizeof(GlobalUBO));
        createStorageBuffers(instanceBufferResources, sizeof(InstanceData) * static_cast<vk::DeviceSize>(sceneInstanceCount));

        sceneBuffersCreated = true;
        return recreatePresetDrivenBuffers();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create multithreaded scene buffers: " << e.what() << std::endl;
        return false;
    }
}

bool MultithreadedRenderer::recreatePresetDrivenBuffers()
{
    if (!sceneBuffersCreated)
    {
        return false;
    }

    try
    {
        instanceBufferResources.clear();
        createStorageBuffers(instanceBufferResources, sizeof(InstanceData) * static_cast<vk::DeviceSize>(sceneInstanceCount));

        particleBufferResources.clear();
        lightBufferResources.clear();

        const vk::DeviceSize particleBytes = sizeof(glm::vec4) * 2ull * static_cast<vk::DeviceSize>(activeParticleCapacity);
        const vk::DeviceSize lightBytes = sizeof(glm::vec4) * 2ull * static_cast<vk::DeviceSize>(activeLightCount);

        createStorageBuffers(particleBufferResources, particleBytes);
        createStorageBuffers(lightBufferResources, lightBytes);

        updateDescriptorSets();

        presetResourcesDirty = false;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to recreate preset-driven buffers: " << e.what() << std::endl;
        return false;
    }
}

bool MultithreadedRenderer::createDescriptors()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex
            },
            {
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex
            },
            {
                .binding = 2,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment
            }
        };

        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data() });

        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
        };

        descriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data() });

        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        instanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data() });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create multithreaded descriptors: " << e.what() << std::endl;
        return false;
    }
}

void MultithreadedRenderer::updateDescriptorSets()
{
    if (instanceBufferResources.descriptorSets.size() != MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo globalBufferInfo{
            .buffer = *globalUboResources.Buffers[i],
            .offset = 0,
            .range = sizeof(GlobalUBO)
        };

        vk::DescriptorBufferInfo instanceBufferInfo{
            .buffer = *instanceBufferResources.Buffers[i],
            .offset = 0,
            .range = sizeof(InstanceData) * static_cast<vk::DeviceSize>(sceneInstanceCount)
        };

        vk::DescriptorImageInfo imageInfo{
            .sampler = texture.textureSampler,
            .imageView = texture.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        std::vector<vk::WriteDescriptorSet> writes = {
            {
                .dstSet = *instanceBufferResources.descriptorSets[i],
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &globalBufferInfo
            },
            {
                .dstSet = *instanceBufferResources.descriptorSets[i],
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &instanceBufferInfo
            },
            {
                .dstSet = *instanceBufferResources.descriptorSets[i],
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo
            }
        };

        device.updateDescriptorSets(writes, nullptr);
    }
}

bool MultithreadedRenderer::createPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "instanced.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
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

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout,
            .pushConstantRangeCount = 0
        };
        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::Format depthFormat = findDepthFormat();
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
                .layout = pipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        pipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create multithreaded pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool MultithreadedRenderer::createSecondaryCommandResources()
{
    try
    {
        secondaryCommandPools.clear();
        secondaryStaticCommandBuffers.clear();
        secondaryDynamicCommandBuffers.clear();

        secondaryCommandPools.reserve(MAX_FRAMES_IN_FLIGHT);
        secondaryStaticCommandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
        secondaryDynamicCommandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vk::CommandPoolCreateInfo poolInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
            };
            secondaryCommandPools.emplace_back(device, poolInfo);

            vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = *secondaryCommandPools.back(),
                .level = vk::CommandBufferLevel::eSecondary,
                .commandBufferCount = 2
            };

            vk::raii::CommandBuffers allocated(device, allocInfo);
            secondaryStaticCommandBuffers.emplace_back(std::move(allocated[0]));
            secondaryDynamicCommandBuffers.emplace_back(std::move(allocated[1]));
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create secondary command resources: " << e.what() << std::endl;
        return false;
    }
}

bool MultithreadedRenderer::initFrameGraph()
{
    frameGraph.reset();
    frameGraph.addPass("Compute: Particle Update", true);
    frameGraph.addPass("Compute: Visibility Culling", true);
    frameGraph.addPass("Graphics: Shadow", true);
    frameGraph.addPass("Graphics: Geometry", true);
    frameGraph.addPass("Graphics: Lighting", true);
    frameGraph.addPass("Graphics: Particle", true);
    frameGraph.addPass("Graphics: PostFX", true);
    return true;
}

bool MultithreadedRenderer::initThreading()
{
    threadPool.start(workerThreadCount);
    workerStats.resize(workerThreadCount);
    return true;
}

bool MultithreadedRenderer::initUI()
{
    if (!uiEnabled)
    {
        return true;
    }

    if (ImGui::GetCurrentContext() != nullptr)
    {
        return true;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;

    unsigned char* pixels = nullptr;
    int fontWidth = 0;
    int fontHeight = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &fontWidth, &fontHeight);

    vk::DeviceSize uploadSize = static_cast<vk::DeviceSize>(fontWidth) * static_cast<vk::DeviceSize>(fontHeight) * 4u;
    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(uploadSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* mapped = stagingBufferMemory.mapMemory(0, uploadSize);
    std::memcpy(mapped, pixels, static_cast<size_t>(uploadSize));
    stagingBufferMemory.unmapMemory();

    uiFontTexture.mipLevels = 1;
    createImage(static_cast<uint32_t>(fontWidth), static_cast<uint32_t>(fontHeight), 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, uiFontTexture);

    transitionImageLayout(uiFontTexture.textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    copyBufferToImage(stagingBuffer, uiFontTexture.textureImage, static_cast<uint32_t>(fontWidth), static_cast<uint32_t>(fontHeight));
    transitionImageLayout(uiFontTexture.textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    uiFontTexture.textureImageView = createImageView(uiFontTexture.textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, 1);

    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::False,
        .maxAnisotropy = 1.0f,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = 0.0f
    };
    uiFontTexture.textureSampler = vk::raii::Sampler(device, samplerInfo);

    std::vector<vk::DescriptorSetLayoutBinding> uiBindings = {
        {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment}
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(uiBindings.size()), .pBindings = uiBindings.data() };
    uiDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1}
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    uiDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *uiDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*uiDescriptorSetLayout
    };
    uiDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    vk::DescriptorImageInfo fontInfo{ .sampler = uiFontTexture.textureSampler, .imageView = uiFontTexture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    vk::WriteDescriptorSet write{ .dstSet = *uiDescriptorSets[0], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &fontInfo };
    device.updateDescriptorSets({ write }, nullptr);

    vk::PushConstantRange pushConstRange{ .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = 16u };
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*uiDescriptorSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstRange };
    uiPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "imgui.spv"));
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    vk::VertexInputBindingDescription bindingDescription{ .binding = 0, .stride = sizeof(ImDrawVert), .inputRate = vk::VertexInputRate::eVertex };
    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, pos)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, uv)),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(ImDrawVert, col))
    };
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
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::False,
        .depthWriteEnable = vk::False,
        .depthCompareOp = vk::CompareOp::eAlways,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

    std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

    vk::Format depthFormat = findDepthFormat();
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
            .layout = uiPipelineLayout,
            .renderPass = nullptr
        },
        {
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = depthFormat
        }
    };
    uiPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    uiFrameBuffers.resize(swapChainImages.size());
    return true;
}

void MultithreadedRenderer::updateFrameData(uint32_t frameIndex)
{
    GlobalUBO ubo{};
    ubo.view = camera.GetViewMatrix();
    ubo.proj = glm::perspective(glm::radians(camera.Zoom), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 300.0f);
    ubo.proj[1][1] *= -1.0f;

    std::memcpy(globalUboResources.BuffersMapped[frameIndex], &ubo, sizeof(ubo));
}

void MultithreadedRenderer::updateInstanceBuffer(uint32_t frameIndex)
{
    std::vector<InstanceData> instances(sceneInstanceCount);

    const uint32_t staticCount = currentPresetConfig.staticInstanceCount;
    const uint32_t dynamicCount = currentPresetConfig.dynamicInstanceCount;

    const uint32_t gridW = static_cast<uint32_t>(std::sqrt(static_cast<float>(std::max(1u, staticCount))));
    const float spacing = 2.8f;

    for (uint32_t i = 0; i < staticCount; ++i)
    {
        const uint32_t x = i % std::max(1u, gridW);
        const uint32_t z = i / std::max(1u, gridW);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3((static_cast<float>(x) - gridW * 0.5f) * spacing, 0.0f, (static_cast<float>(z) - gridW * 0.5f) * spacing));
        model = glm::scale(model, glm::vec3(0.35f));

        instances[i].model = model;
    }

    const float t = static_cast<float>(glfwGetTime());
    for (uint32_t i = 0; i < dynamicCount; ++i)
    {
        const uint32_t idx = staticCount + i;
        const float a = 0.13f * static_cast<float>(i) + t * 0.8f;
        const float r = 20.0f + 0.02f * static_cast<float>(i % 200);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(std::cos(a) * r, std::sin(a * 1.7f) * 1.8f + 1.0f, std::sin(a) * r));
        model = glm::rotate(model, t + 0.1f * static_cast<float>(i), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(0.35f));

        instances[idx].model = model;
    }

    std::memcpy(instanceBufferResources.BuffersMapped[frameIndex], instances.data(), sizeof(InstanceData) * instances.size());
}

void MultithreadedRenderer::updateUIFrame()
{
    const float dt = platform ? platform->frameTimer : 0.0f;
    frameMs = dt * 1000.0f;
    fps = dt > 0.0f ? 1.0f / dt : 0.0f;

    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height));
    io.DeltaTime = dt > 0.0f ? dt : (1.0f / 60.0f);

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(platform->window, &mouseX, &mouseY);
    io.MousePos = ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
    io.MouseDown[0] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    io.MouseDown[1] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    io.MouseDown[2] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    ImGui::NewFrame();
    updateProfilerUI();
    ImGui::Render();
}

const char* MultithreadedRenderer::presetLabel(ScenePreset preset)
{
    switch (preset)
    {
    case ScenePreset::Small: return "Small";
    case ScenePreset::Medium: return "Medium";
    case ScenePreset::Large: return "Large";
    default: return "Unknown";
    }
}

MultithreadedRenderer::ScenePresetConfig MultithreadedRenderer::presetConfig(ScenePreset preset)
{
    switch (preset)
    {
    case ScenePreset::Small:
        return { .staticInstanceCount = 1000, .dynamicInstanceCount = 200, .particleCount = 50000, .pointLightCount = 128, .particleEmissionRate = 3000.0f };
    case ScenePreset::Large:
        return { .staticInstanceCount = 5000, .dynamicInstanceCount = 1000, .particleCount = 200000, .pointLightCount = 512, .particleEmissionRate = 12000.0f };
    case ScenePreset::Medium:
    default:
        return { .staticInstanceCount = 2500, .dynamicInstanceCount = 500, .particleCount = 100000, .pointLightCount = 256, .particleEmissionRate = 6000.0f };
    }
}

void MultithreadedRenderer::applyPreset(ScenePreset preset)
{
    currentPreset = preset;
    currentPresetConfig = presetConfig(preset);

    sceneInstanceCount = currentPresetConfig.staticInstanceCount + currentPresetConfig.dynamicInstanceCount;
    activeParticleCapacity = currentPresetConfig.particleCount;
    activeLightCount = currentPresetConfig.pointLightCount;
    activeParticleEmissionRate = currentPresetConfig.particleEmissionRate;

    presetResourcesDirty = true;
}

void MultithreadedRenderer::updateProfilerUI()
{
    ImGui::Begin("Multithreaded Vulkan Profiler");

    ImGui::Checkbox("Enable Multi-thread Recording", &enableMultiThreadRecording);
    ImGui::Checkbox("Enable Async Compute", &enableAsyncCompute);

    const char* presetItems[] = { "Small", "Medium", "Large" };
    int presetIndex = static_cast<int>(currentPreset);
    if (ImGui::Combo("Scene Preset", &presetIndex, presetItems, IM_ARRAYSIZE(presetItems)))
    {
        applyPreset(static_cast<ScenePreset>(presetIndex));
    }

    int workers = static_cast<int>(workerThreadCount);
    if (ImGui::SliderInt("Worker Threads", &workers, 1, 16))
    {
        workerThreadCount = static_cast<uint32_t>(workers);
        rebuildThreadPoolIfNeeded();
    }

    ImGui::Text("Preset: %s", presetLabel(currentPreset));
    ImGui::Text("Static/Dynamic: %u / %u", currentPresetConfig.staticInstanceCount, currentPresetConfig.dynamicInstanceCount);
    ImGui::Text("Particles: %u @ %.0f/s", activeParticleCapacity, activeParticleEmissionRate);
    ImGui::Text("Point lights: %u", activeLightCount);

    uint32_t totalDrawCalls = 0;
    float totalRecordMs = 0.0f;
    for (const auto& stat : workerStats)
    {
        totalDrawCalls += stat.drawCalls;
        totalRecordMs += stat.recordMs;
    }

    ImGui::Text("Frame: %.2f ms", frameMs);
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Draw Calls (mock): %u", totalDrawCalls);
    ImGui::Text("CPU Record (sum): %.3f ms", totalRecordMs);

    ImGui::Separator();
    ImGui::Text("Auto Benchmark");
    ImGui::SliderFloat("Duration (sec)", &benchmarkDurationSeconds, 2.0f, 30.0f, "%.1f");

    const bool benchmarkRunning = benchmarkMode == BenchmarkMode::SingleThread || benchmarkMode == BenchmarkMode::MultiThread;
    if (!benchmarkRunning)
    {
        if (ImGui::Button("Start Auto Benchmark"))
        {
            startAutoBenchmark();
        }
    }
    else
    {
        const char* phase = benchmarkMode == BenchmarkMode::SingleThread ? "Single-thread" : "Multi-thread";
        ImGui::Text("Running: %s", phase);
        ImGui::Text("Progress: %.2f / %.2f sec", benchmarkElapsedSeconds, benchmarkDurationSeconds);
    }

    if (benchmarkMode == BenchmarkMode::Done || benchmarkSingleStats.valid || benchmarkMultiStats.valid)
    {
        ImGui::Separator();
        ImGui::Text("Benchmark Compare (ms)");

        ImGui::Columns(7, "benchmark_table");
        ImGui::Text("Mode"); ImGui::NextColumn();
        ImGui::Text("Frame Avg"); ImGui::NextColumn();
        ImGui::Text("Frame P95"); ImGui::NextColumn();
        ImGui::Text("Frame P99"); ImGui::NextColumn();
        ImGui::Text("Record Avg"); ImGui::NextColumn();
        ImGui::Text("Record P95"); ImGui::NextColumn();
        ImGui::Text("Record P99"); ImGui::NextColumn();
        ImGui::Separator();

        const auto drawBenchmarkRow = [](const char* label, const BenchmarkStats& s)
        {
            ImGui::Text("%s", label); ImGui::NextColumn();
            ImGui::Text("%.3f", s.frameMs.avg); ImGui::NextColumn();
            ImGui::Text("%.3f", s.frameMs.p95); ImGui::NextColumn();
            ImGui::Text("%.3f", s.frameMs.p99); ImGui::NextColumn();
            ImGui::Text("%.3f", s.recordMs.avg); ImGui::NextColumn();
            ImGui::Text("%.3f", s.recordMs.p95); ImGui::NextColumn();
            ImGui::Text("%.3f", s.recordMs.p99); ImGui::NextColumn();
        };

        drawBenchmarkRow("Single", benchmarkSingleStats);
        drawBenchmarkRow("Multi", benchmarkMultiStats);
        ImGui::Columns(1);

        if (benchmarkSingleStats.valid && benchmarkMultiStats.valid && benchmarkSingleStats.recordMs.avg > 0.0001f)
        {
            const float improve = (benchmarkSingleStats.recordMs.avg - benchmarkMultiStats.recordMs.avg) / benchmarkSingleStats.recordMs.avg * 100.0f;
            ImGui::Text("Record Avg Improvement: %.2f%%", improve);
        }
    }

    ImGui::End();
}

MultithreadedRenderer::WorkerRecordStats MultithreadedRenderer::recordWorkerRange(uint32_t frameIndex, RenderBatch batch, bool dynamicBatch)
{
    (void)frameIndex;

    const auto t0 = std::chrono::high_resolution_clock::now();

    WorkerRecordStats stats{};
    stats.drawCalls = batch.end - batch.begin;
    if (dynamicBatch)
    {
        stats.dynamicDrawCalls = stats.drawCalls;
    }
    else
    {
        stats.staticDrawCalls = stats.drawCalls;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration<float, std::milli>(t1 - t0);
    stats.recordMs = duration.count();
    return stats;
}

void MultithreadedRenderer::dispatchWorkerRecording(uint32_t frameIndex)
{
    const uint32_t staticCount = currentPresetConfig.staticInstanceCount;
    const uint32_t dynamicCount = currentPresetConfig.dynamicInstanceCount;

    if (!enableMultiThreadRecording)
    {
        WorkerRecordStats merged{};
        if (staticCount > 0)
        {
            auto s = recordWorkerRange(frameIndex, { 0u, staticCount }, false);
            merged.drawCalls += s.drawCalls;
            merged.staticDrawCalls += s.staticDrawCalls;
            merged.recordMs += s.recordMs;
        }
        if (dynamicCount > 0)
        {
            auto d = recordWorkerRange(frameIndex, { 0u, dynamicCount }, true);
            merged.drawCalls += d.drawCalls;
            merged.dynamicDrawCalls += d.dynamicDrawCalls;
            merged.recordMs += d.recordMs;
        }
        workerStats.assign(1, merged);
        return;
    }

    const uint32_t workerCount = std::max(1u, workerThreadCount);
    const uint32_t staticWorkers = std::max(1u, (workerCount * staticCount) / std::max(1u, sceneInstanceCount));
    const uint32_t dynamicWorkers = std::max(1u, workerCount - staticWorkers);

    const auto staticBatches = RenderBatcher::splitEvenly(staticCount, staticWorkers);
    const auto dynamicBatches = RenderBatcher::splitEvenly(dynamicCount, dynamicWorkers);

    std::vector<std::future<WorkerRecordStats>> futures;
    futures.reserve(staticBatches.size() + dynamicBatches.size());

    for (const auto& batch : staticBatches)
    {
        futures.push_back(threadPool.enqueue([this, frameIndex, batch]()
        {
            return recordWorkerRange(frameIndex, batch, false);
        }));
    }

    for (const auto& batch : dynamicBatches)
    {
        futures.push_back(threadPool.enqueue([this, frameIndex, batch]()
        {
            return recordWorkerRange(frameIndex, batch, true);
        }));
    }

    workerStats.clear();
    workerStats.reserve(futures.size());
    for (auto& future : futures)
    {
        workerStats.push_back(future.get());
    }
}

void MultithreadedRenderer::recordPrimaryCommandBuffer(uint32_t imageIndex)
{
    if (imageIndex >= swapChainImages.size())
    {
        return;
    }

    vk::raii::CommandBuffer& primary = commandBuffers[currentFrame];
    primary.reset();
    primary.begin({});

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

    const vk::ClearValue clearColor = vk::ClearColorValue(0.02f, 0.02f, 0.03f, 1.0f);
    const vk::RenderingAttachmentInfo colorAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };
    const vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    const vk::RenderingAttachmentInfo depthAttachment{
        .imageView = depthData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth
    };

    const vk::RenderingInfo renderingInfo{
        .flags = vk::RenderingFlagBits::eContentsSecondaryCommandBuffers,
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment
    };

    primary.beginRendering(renderingInfo);

    vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{
        .flags = {},
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapChainImageFormat,
        .depthAttachmentFormat = findDepthFormat(),
        .stencilAttachmentFormat = vk::Format::eUndefined,
        .rasterizationSamples = vk::SampleCountFlagBits::e1
    };

    vk::CommandBufferInheritanceInfo inheritanceInfo{};
    inheritanceInfo.setPNext(&inheritanceRenderingInfo);

    const auto recordSecondary = [&](vk::raii::CommandBuffer& secondary, uint32_t firstInstance, uint32_t instanceCount)
    {
        secondary.reset();
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
            .pInheritanceInfo = &inheritanceInfo
        };
        secondary.begin(beginInfo);

        secondary.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        secondary.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

        secondary.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        secondary.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *instanceBufferResources.descriptorSets[currentFrame], nullptr);
        secondary.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
        secondary.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);

        if (instanceCount > 0)
        {
            secondary.drawIndexed(static_cast<uint32_t>(mesh.indices.size()), instanceCount, 0, 0, firstInstance);
        }

        secondary.end();
    };

    const uint32_t staticCount = currentPresetConfig.staticInstanceCount;
    const uint32_t dynamicCount = currentPresetConfig.dynamicInstanceCount;

    recordSecondary(secondaryStaticCommandBuffers[currentFrame], 0u, staticCount);
    recordSecondary(secondaryDynamicCommandBuffers[currentFrame], staticCount, dynamicCount);

    std::vector<vk::CommandBuffer> secondaryToExecute;
    if (staticCount > 0)
    {
        secondaryToExecute.push_back(*secondaryStaticCommandBuffers[currentFrame]);
    }
    if (dynamicCount > 0)
    {
        secondaryToExecute.push_back(*secondaryDynamicCommandBuffers[currentFrame]);
    }

    if (!secondaryToExecute.empty())
    {
        primary.executeCommands(secondaryToExecute);
    }

    primary.endRendering();

    vk::RenderingAttachmentInfo uiAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingInfo uiRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachmentInfo
    };

    primary.beginRendering(uiRenderingInfo);
    primary.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    primary.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUI(primary, imageIndex);
    primary.endRendering();

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

    primary.end();
}

void MultithreadedRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex)
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr || uiPipeline == vk::raii::Pipeline(nullptr))
    {
        return;
    }

    ImDrawData* drawData = ImGui::GetDrawData();
    if (drawData == nullptr || drawData->TotalVtxCount <= 0)
    {
        return;
    }

    auto& fb = uiFrameBuffers[frameIndex];
    size_t vertexBytes = static_cast<size_t>(drawData->TotalVtxCount) * sizeof(ImDrawVert);
    size_t indexBytes = static_cast<size_t>(drawData->TotalIdxCount) * sizeof(ImDrawIdx);

    if (fb.vertexBuffer == vk::raii::Buffer(nullptr) || fb.vertexSize < vertexBytes)
    {
        if (fb.vertexMapped != nullptr)
        {
            fb.vertexBufferMemory.unmapMemory();
            fb.vertexMapped = nullptr;
        }
        fb.vertexBuffer = vk::raii::Buffer(nullptr);
        fb.vertexBufferMemory = vk::raii::DeviceMemory(nullptr);
        createBuffer(vertexBytes, vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, fb.vertexBuffer, fb.vertexBufferMemory);
        fb.vertexMapped = fb.vertexBufferMemory.mapMemory(0, vertexBytes);
        fb.vertexSize = vertexBytes;
    }

    if (fb.indexBuffer == vk::raii::Buffer(nullptr) || fb.indexSize < indexBytes)
    {
        if (fb.indexMapped != nullptr)
        {
            fb.indexBufferMemory.unmapMemory();
            fb.indexMapped = nullptr;
        }
        fb.indexBuffer = vk::raii::Buffer(nullptr);
        fb.indexBufferMemory = vk::raii::DeviceMemory(nullptr);
        createBuffer(indexBytes, vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, fb.indexBuffer, fb.indexBufferMemory);
        fb.indexMapped = fb.indexBufferMemory.mapMemory(0, indexBytes);
        fb.indexSize = indexBytes;
    }

    ImDrawVert* vtxDst = reinterpret_cast<ImDrawVert*>(fb.vertexMapped);
    ImDrawIdx* idxDst = reinterpret_cast<ImDrawIdx*>(fb.indexMapped);
    for (int n = 0; n < drawData->CmdListsCount; ++n)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        std::memcpy(vtxDst, cmdList->VtxBuffer.Data, static_cast<size_t>(cmdList->VtxBuffer.Size) * sizeof(ImDrawVert));
        std::memcpy(idxDst, cmdList->IdxBuffer.Data, static_cast<size_t>(cmdList->IdxBuffer.Size) * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *uiPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *uiPipelineLayout, 0, *uiDescriptorSets[0], nullptr);
    commandBuffer.bindVertexBuffers(0, *fb.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*fb.indexBuffer, 0, sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

    struct UIPushConst
    {
        glm::vec2 scale;
        glm::vec2 translate;
    };

    UIPushConst pc{};
    pc.scale = glm::vec2(2.0f / drawData->DisplaySize.x, 2.0f / drawData->DisplaySize.y);
    pc.translate = glm::vec2(-1.0f, -1.0f);
    commandBuffer.pushConstants(*uiPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, vk::ArrayProxy<const UIPushConst>(1, &pc));

    int32_t globalVtxOffset = 0;
    uint32_t globalIdxOffset = 0;

    for (int n = 0; n < drawData->CmdListsCount; ++n)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        uint32_t idxOffset = 0;

        for (int cmdI = 0; cmdI < cmdList->CmdBuffer.Size; ++cmdI)
        {
            const ImDrawCmd* drawCmd = &cmdList->CmdBuffer[cmdI];

            ImVec4 clipRect = drawCmd->ClipRect;
            clipRect.x -= drawData->DisplayPos.x;
            clipRect.y -= drawData->DisplayPos.y;
            clipRect.z -= drawData->DisplayPos.x;
            clipRect.w -= drawData->DisplayPos.y;

            if (clipRect.x < static_cast<float>(swapChainExtent.width) && clipRect.y < static_cast<float>(swapChainExtent.height) && clipRect.z >= 0.0f && clipRect.w >= 0.0f)
            {
                vk::Rect2D scissor;
                scissor.offset.x = static_cast<int32_t>(clipRect.x > 0.0f ? clipRect.x : 0.0f);
                scissor.offset.y = static_cast<int32_t>(clipRect.y > 0.0f ? clipRect.y : 0.0f);
                scissor.extent.width = static_cast<uint32_t>(clipRect.z - clipRect.x);
                scissor.extent.height = static_cast<uint32_t>(clipRect.w - clipRect.y);
                commandBuffer.setScissor(0, scissor);
                commandBuffer.drawIndexed(drawCmd->ElemCount, 1, globalIdxOffset + idxOffset, globalVtxOffset, 0);
            }

            idxOffset += drawCmd->ElemCount;
        }

        globalIdxOffset += static_cast<uint32_t>(cmdList->IdxBuffer.Size);
        globalVtxOffset += cmdList->VtxBuffer.Size;
    }
}

MultithreadedRenderer::PercentileStats MultithreadedRenderer::calcPercentiles(const std::vector<float>& samples)
{
    PercentileStats out{};
    if (samples.empty())
    {
        return out;
    }

    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    float sum = 0.0f;
    for (float v : sorted)
    {
        sum += v;
    }
    out.avg = sum / static_cast<float>(sorted.size());

    const auto getPercentile = [&sorted](float p)
    {
        if (sorted.empty()) return 0.0f;
        const float idx = (p / 100.0f) * static_cast<float>(sorted.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(idx));
        const size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) return sorted[lo];
        const float t = idx - static_cast<float>(lo);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * t;
    };

    out.p95 = getPercentile(95.0f);
    out.p99 = getPercentile(99.0f);
    return out;
}

MultithreadedRenderer::BenchmarkStats MultithreadedRenderer::buildBenchmarkStats(const std::vector<float>& frameSamples, const std::vector<float>& recordSamples)
{
    BenchmarkStats stats{};
    stats.frameMs = calcPercentiles(frameSamples);
    stats.recordMs = calcPercentiles(recordSamples);
    stats.sampleCount = static_cast<uint32_t>(std::min(frameSamples.size(), recordSamples.size()));
    stats.valid = stats.sampleCount > 0;
    return stats;
}

void MultithreadedRenderer::resetBenchmarkSamples()
{
    benchmarkSingleFrameSamples.clear();
    benchmarkSingleRecordSamples.clear();
    benchmarkMultiFrameSamples.clear();
    benchmarkMultiRecordSamples.clear();

    benchmarkSingleStats = BenchmarkStats{};
    benchmarkMultiStats = BenchmarkStats{};
    benchmarkElapsedSeconds = 0.0f;
    benchmarkMode = BenchmarkMode::None;
}

void MultithreadedRenderer::startAutoBenchmark()
{
    resetBenchmarkSamples();

    benchmarkRestoreMultiThreadState = enableMultiThreadRecording;
    enableMultiThreadRecording = false;

    benchmarkMode = BenchmarkMode::SingleThread;
    benchmarkElapsedSeconds = 0.0f;
    benchmarkAutoRun = true;
}

void MultithreadedRenderer::updateBenchmarkFlow(float dt, float cpuRecordMs)
{
    if (!benchmarkAutoRun)
    {
        return;
    }

    if (benchmarkMode == BenchmarkMode::SingleThread)
    {
        benchmarkSingleFrameSamples.push_back(frameMs);
        benchmarkSingleRecordSamples.push_back(cpuRecordMs);
    }
    else if (benchmarkMode == BenchmarkMode::MultiThread)
    {
        benchmarkMultiFrameSamples.push_back(frameMs);
        benchmarkMultiRecordSamples.push_back(cpuRecordMs);
    }

    benchmarkElapsedSeconds += dt;
    if (benchmarkElapsedSeconds < benchmarkDurationSeconds)
    {
        return;
    }

    if (benchmarkMode == BenchmarkMode::SingleThread)
    {
        benchmarkSingleStats = buildBenchmarkStats(benchmarkSingleFrameSamples, benchmarkSingleRecordSamples);
        enableMultiThreadRecording = true;
        benchmarkMode = BenchmarkMode::MultiThread;
        benchmarkElapsedSeconds = 0.0f;
        return;
    }

    if (benchmarkMode == BenchmarkMode::MultiThread)
    {
        benchmarkMultiStats = buildBenchmarkStats(benchmarkMultiFrameSamples, benchmarkMultiRecordSamples);
        enableMultiThreadRecording = benchmarkRestoreMultiThreadState;
        benchmarkMode = BenchmarkMode::Done;
        benchmarkAutoRun = false;
    }
}

void MultithreadedRenderer::rebuildThreadPoolIfNeeded()
{
    threadPool.start(std::max(1u, workerThreadCount));
    workerStats.assign(workerThreadCount, WorkerRecordStats{});
}

void MultithreadedRenderer::render()
{
    if (swapChainImages.empty())
    {
        return;
    }

    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [acquireResult, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (acquireResult == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);

    if (presetResourcesDirty)
    {
        device.waitIdle();
        if (!recreatePresetDrivenBuffers())
        {
            throw std::runtime_error("failed to recreate preset-driven buffers");
        }
    }

    updateFrameData(currentFrame);
    updateInstanceBuffer(currentFrame);
    updateUIFrame();

    gpuProfiler.beginFrame();
    dispatchWorkerRecording(currentFrame);
    gpuProfiler.endFrame();

    float totalRecordMs = 0.0f;
    for (const auto& stat : workerStats)
    {
        totalRecordMs += stat.recordMs;
    }
    const float dt = platform ? platform->frameTimer : (1.0f / 60.0f);
    updateBenchmarkFlow(dt, totalRecordMs);

    recordPrimaryCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
    };
    graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

    vk::PresentInfoKHR presentInfo{
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

    if ((presentResult == vk::Result::eSuboptimalKHR) || (presentResult == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void MultithreadedRenderer::cleanup()
{
    threadPool.stop();
    instanceBufferResources.clear();
    particleBufferResources.clear();
    lightBufferResources.clear();
    globalUboResources.clear();
    shutdownUI();
    VulkanBase::cleanupSwapChain();
}

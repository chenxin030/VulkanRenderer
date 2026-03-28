#include "DeferredRenderer.h"

#include <Base/Mesh.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <imgui.h>

struct DeferredInstanceData
{
    glm::mat4 model;
    glm::vec4 baseColor;
    glm::vec4 material; // x: metallic, y: roughness, z: ao, w: unused
};

struct SceneUBO
{
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 invProjection;
    glm::mat4 invView;
    glm::vec4 camPos;
};

struct PointLight
{
    glm::vec4 position;
    glm::vec4 color;
};

struct LightUBO
{
    PointLight lights[4];
};

struct DeferredSettingsUBO
{
    glm::vec4 params0; // x ambient, y exposure, z gamma, w lightScale
    glm::ivec4 debug;  // x debugView
};

void DeferredRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool DeferredRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 0.0f, 15.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 5_deferredPBR");
}

bool DeferredRenderer::prepareResource()
{
    generateSphere(sphereMesh, 1.0f, 100);
    createVertexBuffer(sphereMesh);
    createIndexBuffer(sphereMesh);

    createDeferredBuffers();

    if (!createGBufferDescriptorSetLayout()) return false;
    if (!createDeferredDescriptorSetLayout()) return false;

    if (!createGBufferDescriptorPool()) return false;
    if (!createDeferredDescriptorPool()) return false;

    if (!createGBufferResources()) return false;

    if (!createGBufferPipeline()) return false;
    if (!createDeferredPipeline()) return false;

    createGBufferDescriptorSets();
    createDeferredDescriptorSets();

    if (!initUI()) return false;

    return true;
}

void DeferredRenderer::cleanup()
{
    shutdownUI();
    destroyGBufferResources();
}

void DeferredRenderer::createDeferredBuffers()
{
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(lightUboResources, sizeof(LightUBO));
    createUniformBuffers(deferredSettingsUboResources, sizeof(DeferredSettingsUBO));
    createStorageBuffers(gbufferInstanceBufferResources, sizeof(DeferredInstanceData) * instanceCount);
}

bool DeferredRenderer::createGBufferDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        };

        gbufferDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create gbuffer descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool DeferredRenderer::createDeferredDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 4, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 5, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };

        deferredDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create deferred descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool DeferredRenderer::createGBufferDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        };

        gbufferDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create gbuffer descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

bool DeferredRenderer::createDeferredDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u },
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
        };

        deferredDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create deferred descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void DeferredRenderer::createGBufferDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *gbufferDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *gbufferDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    gbufferInstanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceBufferInfo{ .buffer = *gbufferInstanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(DeferredInstanceData) * instanceCount };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *gbufferInstanceBufferResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *gbufferInstanceBufferResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

void DeferredRenderer::createDeferredDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *deferredDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *deferredDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    lightUboResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorImageInfo albedoInfo{ .sampler = gbufferSampler, .imageView = gbufferAlbedo.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo normalInfo{ .sampler = gbufferSampler, .imageView = gbufferNormal.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo materialInfo{ .sampler = gbufferSampler, .imageView = gbufferMaterial.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo depthInfo{ .sampler = gbufferSampler, .imageView = gbufferDepth.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo lightBufferInfo{ .buffer = *lightUboResources.Buffers[i], .offset = 0, .range = sizeof(LightUBO) };
        vk::DescriptorBufferInfo settingsBufferInfo{ .buffer = *deferredSettingsUboResources.Buffers[i], .offset = 0, .range = sizeof(DeferredSettingsUBO) };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &albedoInfo },
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo },
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &materialInfo },
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo },
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &lightBufferInfo },
            { .dstSet = *lightUboResources.descriptorSets[i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &settingsBufferInfo },
        };

        device.updateDescriptorSets(writes, nullptr);
    }
}

bool DeferredRenderer::createGBufferResources()
{
    try
    {
        destroyGBufferResources();

        gbufferAlbedo.format = vk::Format::eR16G16B16A16Sfloat;
        gbufferNormal.format = vk::Format::eR16G16B16A16Sfloat;
        gbufferMaterial.format = vk::Format::eR16G16B16A16Sfloat;
        gbufferDepth.format = vk::Format::eD32Sfloat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbufferAlbedo.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbufferAlbedo.texture);
        gbufferAlbedo.texture.textureImageView = createImageView(gbufferAlbedo.texture.textureImage, gbufferAlbedo.format, vk::ImageAspectFlagBits::eColor, 1);
        gbufferAlbedo.layout = vk::ImageLayout::eUndefined;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbufferNormal.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbufferNormal.texture);
        gbufferNormal.texture.textureImageView = createImageView(gbufferNormal.texture.textureImage, gbufferNormal.format, vk::ImageAspectFlagBits::eColor, 1);
        gbufferNormal.layout = vk::ImageLayout::eUndefined;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbufferMaterial.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbufferMaterial.texture);
        gbufferMaterial.texture.textureImageView = createImageView(gbufferMaterial.texture.textureImage, gbufferMaterial.format, vk::ImageAspectFlagBits::eColor, 1);
        gbufferMaterial.layout = vk::ImageLayout::eUndefined;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbufferDepth.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbufferDepth.texture);
        gbufferDepth.texture.textureImageView = createImageView(gbufferDepth.texture.textureImage, gbufferDepth.format, vk::ImageAspectFlagBits::eDepth, 1);
        gbufferDepth.layout = vk::ImageLayout::eUndefined;

        if (gbufferSampler == vk::raii::Sampler(nullptr))
        {
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
                .maxLod = 0.0f,
                .borderColor = vk::BorderColor::eFloatOpaqueWhite,
                .unnormalizedCoordinates = vk::False
            };
            gbufferSampler = vk::raii::Sampler(device, samplerInfo);
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create gbuffer resources: " << e.what() << std::endl;
        return false;
    }
}

void DeferredRenderer::destroyGBufferResources()
{
    gbufferAlbedo.texture.textureImageView = nullptr;
    gbufferAlbedo.texture.textureImage = nullptr;
    gbufferAlbedo.texture.textureImageMemory = nullptr;

    gbufferNormal.texture.textureImageView = nullptr;
    gbufferNormal.texture.textureImage = nullptr;
    gbufferNormal.texture.textureImageMemory = nullptr;

    gbufferMaterial.texture.textureImageView = nullptr;
    gbufferMaterial.texture.textureImage = nullptr;
    gbufferMaterial.texture.textureImageMemory = nullptr;

    gbufferDepth.texture.textureImageView = nullptr;
    gbufferDepth.texture.textureImage = nullptr;
    gbufferDepth.texture.textureImageMemory = nullptr;

    gbufferAlbedo.layout = vk::ImageLayout::eUndefined;
    gbufferNormal.layout = vk::ImageLayout::eUndefined;
    gbufferMaterial.layout = vk::ImageLayout::eUndefined;
    gbufferDepth.layout = vk::ImageLayout::eUndefined;
}

bool DeferredRenderer::createGBufferPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "deferred_gbuffer.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        const auto bindingDescription = Vertex::getBindingDescription();
        const auto attributeDescriptions = Vertex::getPositionNormalAttributeDescriptions();
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

        std::array<vk::PipelineColorBlendAttachmentState, 3> blendAttachments{
            vk::PipelineColorBlendAttachmentState{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA },
            vk::PipelineColorBlendAttachmentState{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA },
            vk::PipelineColorBlendAttachmentState{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA }
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = static_cast<uint32_t>(blendAttachments.size()),
            .pAttachments = blendAttachments.data()
        };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        gbufferPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*gbufferDescriptorSetLayout,
            .pushConstantRangeCount = 0
        });

        std::array<vk::Format, 3> gbufferColorFormats = {
            gbufferAlbedo.format,
            gbufferNormal.format,
            gbufferMaterial.format
        };

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
                .layout = gbufferPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = static_cast<uint32_t>(gbufferColorFormats.size()),
                .pColorAttachmentFormats = gbufferColorFormats.data(),
                .depthAttachmentFormat = gbufferDepth.format
            }
        };

        gbufferPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create gbuffer pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool DeferredRenderer::createDeferredPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "deferred_lighting.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
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
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        deferredPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*deferredDescriptorSetLayout,
            .pushConstantRangeCount = 0
        });

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
                .layout = deferredPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat
            }
        };

        deferredPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create deferred lighting pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool DeferredRenderer::initUI()
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
    memcpy(mapped, pixels, static_cast<size_t>(uploadSize));
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

    uiFrameBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    return true;
}

void DeferredRenderer::shutdownUI()
{
    if (ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    device.waitIdle();

    for (auto& fb : uiFrameBuffers)
    {
        if (fb.vertexMapped != nullptr)
        {
            fb.vertexBufferMemory.unmapMemory();
            fb.vertexMapped = nullptr;
        }
        if (fb.indexMapped != nullptr)
        {
            fb.indexBufferMemory.unmapMemory();
            fb.indexMapped = nullptr;
        }
    }
    uiFrameBuffers.clear();

    uiPipeline = vk::raii::Pipeline(nullptr);
    uiPipelineLayout = vk::raii::PipelineLayout(nullptr);
    uiDescriptorSets = vk::raii::DescriptorSets(nullptr);
    uiDescriptorPool = vk::raii::DescriptorPool(nullptr);
    uiDescriptorSetLayout = vk::raii::DescriptorSetLayout(nullptr);

    uiFontTexture.textureSampler = vk::raii::Sampler(nullptr);
    uiFontTexture.textureImageView = vk::raii::ImageView(nullptr);
    uiFontTexture.textureImage = vk::raii::Image(nullptr);
    uiFontTexture.textureImageMemory = vk::raii::DeviceMemory(nullptr);

    ImGui::DestroyContext();
}

void DeferredRenderer::updateDeferredUI()
{
    ImGui::SetNextWindowSize(ImVec2(560.0f, 320.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Deferred PBR", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Columns(2, "DeferredColumns", false);

    ImGui::TextUnformatted("Lighting");
    ImGui::Separator();
    ImGui::Checkbox("Animate Lights", &animateLights);
    ImGui::SliderFloat("Ambient", &ambientStrength, 0.0f, 0.2f);
    ImGui::SliderFloat("Exposure", &exposure, 0.2f, 3.0f);
    ImGui::SliderFloat("Gamma", &gamma, 1.2f, 3.0f);
    ImGui::SliderFloat("Light Scale", &lightIntensityScale, 0.1f, 4.0f);

    ImGui::NextColumn();

    ImGui::TextUnformatted("Debug");
    ImGui::Separator();
    const char* views[] = { "Lit", "Albedo", "Normal", "Material", "Depth" };
    ImGui::Combo("Debug View", &debugView, views, 5);

    ImGui::Spacing();
    ImGui::TextUnformatted("GBuffer Channels:");
    ImGui::BulletText("Albedo: base color");
    ImGui::BulletText("Normal: encoded world normal");
    ImGui::BulletText("Material: metallic/roughness/ao");
    ImGui::BulletText("Depth: scene depth");

    ImGui::Columns(1);
    ImGui::End();
}

void DeferredRenderer::updateUIFrame()
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr)
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
    updateDeferredUI();
    ImGui::Render();
}

void DeferredRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
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

    auto& fb = uiFrameBuffers[currentFrame];
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
    for (int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        memcpy(vtxDst, cmdList->VtxBuffer.Data, static_cast<size_t>(cmdList->VtxBuffer.Size) * sizeof(ImDrawVert));
        memcpy(idxDst, cmdList->IdxBuffer.Data, static_cast<size_t>(cmdList->IdxBuffer.Size) * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *uiPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *uiPipelineLayout, 0, *uiDescriptorSets[0], nullptr);
    commandBuffer.bindVertexBuffers(0, *fb.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*fb.indexBuffer, 0, sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

    struct UiPushConsts
    {
        glm::vec2 scale;
        glm::vec2 translate;
    };

    UiPushConsts pc;
    pc.scale = glm::vec2(2.0f / float(drawData->DisplaySize.x), 2.0f / float(drawData->DisplaySize.y));
    pc.translate = glm::vec2(-1.0f, -1.0f);
    commandBuffer.pushConstants(*uiPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, vk::ArrayProxy<const UiPushConsts>(1, &pc));

    int32_t globalVertexOffset = 0;
    uint32_t globalIndexOffset = 0;
    ImVec2 clipOff = drawData->DisplayPos;
    ImVec2 clipScale = ImVec2(1.0f, 1.0f);

    for (int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        uint32_t indexOffset = 0;
        for (int cmdI = 0; cmdI < cmdList->CmdBuffer.Size; cmdI++)
        {
            const ImDrawCmd* cmd = &cmdList->CmdBuffer[cmdI];
            ImVec4 clipRect;
            clipRect.x = (cmd->ClipRect.x - clipOff.x) * clipScale.x;
            clipRect.y = (cmd->ClipRect.y - clipOff.y) * clipScale.y;
            clipRect.z = (cmd->ClipRect.z - clipOff.x) * clipScale.x;
            clipRect.w = (cmd->ClipRect.w - clipOff.y) * clipScale.y;

            if (clipRect.x < float(swapChainExtent.width) && clipRect.y < float(swapChainExtent.height) && clipRect.z >= 0.0f && clipRect.w >= 0.0f)
            {
                vk::Rect2D scissor;
                scissor.offset.x = static_cast<int32_t>(clipRect.x > 0.0f ? clipRect.x : 0.0f);
                scissor.offset.y = static_cast<int32_t>(clipRect.y > 0.0f ? clipRect.y : 0.0f);
                float scissorW = clipRect.z - clipRect.x;
                float scissorH = clipRect.w - clipRect.y;
                if (scissorW < 0.0f) scissorW = 0.0f;
                if (scissorH < 0.0f) scissorH = 0.0f;
                scissor.extent.width = static_cast<uint32_t>(scissorW);
                scissor.extent.height = static_cast<uint32_t>(scissorH);
                commandBuffer.setScissor(0, scissor);
                commandBuffer.drawIndexed(cmd->ElemCount, 1, globalIndexOffset + indexOffset, globalVertexOffset, 0);
            }

            indexOffset += cmd->ElemCount;
        }
        globalIndexOffset += static_cast<uint32_t>(cmdList->IdxBuffer.Size);
        globalVertexOffset += cmdList->VtxBuffer.Size;
    }
}

void DeferredRenderer::updateDeferredBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 120.0f),
        .view = camera.GetViewMatrix(),
        .camPos = glm::vec4(camera.Position, 1.0f)
    };
    sceneUbo.projection[1][1] *= -1;
    sceneUbo.invProjection = glm::inverse(sceneUbo.projection);
    sceneUbo.invView = glm::inverse(sceneUbo.view);
    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    std::vector<DeferredInstanceData> instances;
    instances.reserve(instanceCount);
    for (int y = -3; y <= 3; ++y)
    {
        for (int x = -3; x <= 3; ++x)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, glm::vec3(static_cast<float>(x) * 2.2f, static_cast<float>(y) * 2.2f, 0.0f));

            const float fx = static_cast<float>(x + 3) / 6.0f;
            const float fy = static_cast<float>(y + 3) / 6.0f;

            const float metallic = fx;
            const float roughness = std::clamp(fy, 0.04f, 1.0f);
            const float ao = 1.0f;

            const glm::vec4 color(1.0f, 0.86f, 0.57f, 1.0f);
            const glm::vec4 material(metallic, roughness, ao, 0.0f);
            instances.push_back(DeferredInstanceData{ model, color, material });
        }
    }
    std::memcpy(gbufferInstanceBufferResources.BuffersMapped[frameIndex], instances.data(), sizeof(DeferredInstanceData) * instances.size());

    static auto startTime = std::chrono::high_resolution_clock::now();
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float>(currentTime - startTime).count();

    float animSin = animateLights ? std::sin(time * 0.5f) : 0.0f;
    float animCos = animateLights ? std::cos(time * 0.7f) : 1.0f;

    LightUBO lightUbo{};
    lightUbo.lights[0] = { .position = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 400.0f * lightIntensityScale) };
    lightUbo.lights[1] = { .position = glm::vec4(-20.0f, -10.0f, 10.0f, 1.0f), .color = glm::vec4(1.0f, 0.9f, 0.8f, 80.0f * lightIntensityScale) };
    lightUbo.lights[2] = { .position = glm::vec4(animSin * 12.0f, 5.0f, 8.0f, 1.0f), .color = glm::vec4(0.8f, 1.0f, 1.0f, 180.0f * lightIntensityScale) };
    lightUbo.lights[3] = { .position = glm::vec4(0.0f, animCos * 12.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 0.8f, 1.0f, 180.0f * lightIntensityScale) };
    std::memcpy(lightUboResources.BuffersMapped[frameIndex], &lightUbo, sizeof(lightUbo));

    DeferredSettingsUBO settings{};
    settings.params0 = glm::vec4(ambientStrength, exposure, gamma, lightIntensityScale);
    settings.debug = glm::ivec4(debugView, 0, 0, 0);
    std::memcpy(deferredSettingsUboResources.BuffersMapped[frameIndex], &settings, sizeof(settings));
}

bool DeferredRenderer::recreateDeferredSizedResources()
{
    if (!createGBufferResources()) return false;
    createDeferredDescriptorSets();
    return true;
}

void DeferredRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    transition_image_layout(
        gbufferAlbedo.texture.textureImage,
        gbufferAlbedo.layout,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    gbufferAlbedo.layout = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(
        gbufferNormal.texture.textureImage,
        gbufferNormal.layout,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    gbufferNormal.layout = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(
        gbufferMaterial.texture.textureImage,
        gbufferMaterial.layout,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    gbufferMaterial.layout = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(
        gbufferDepth.texture.textureImage,
        gbufferDepth.layout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    gbufferDepth.layout = vk::ImageLayout::eDepthAttachmentOptimal;

    std::array<vk::RenderingAttachmentInfo, 3> gbufferColorAttachments = {
        vk::RenderingAttachmentInfo{
            .imageView = gbufferAlbedo.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
        },
        vk::RenderingAttachmentInfo{
            .imageView = gbufferNormal.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.5f, 0.5f, 1.0f, 1.0f})
        },
        vk::RenderingAttachmentInfo{
            .imageView = gbufferMaterial.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.6f, 1.0f, 0.0f})
        }
    };

    vk::RenderingAttachmentInfo gbufferDepthAttachment{
        .imageView = gbufferDepth.texture.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
    };

    vk::RenderingInfo gbufferRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(gbufferColorAttachments.size()),
        .pColorAttachments = gbufferColorAttachments.data(),
        .pDepthAttachment = &gbufferDepthAttachment
    };

    commandBuffer.beginRendering(gbufferRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *gbufferPipeline);
    commandBuffer.bindVertexBuffers(0, *sphereMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*sphereMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(sphereMesh.indices)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *gbufferPipelineLayout, 0, *gbufferInstanceBufferResources.descriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(sphereMesh.indices.size()), instanceCount, 0, 0, 0);
    commandBuffer.endRendering();

    transition_image_layout(
        gbufferAlbedo.texture.textureImage,
        gbufferAlbedo.layout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    gbufferAlbedo.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(
        gbufferNormal.texture.textureImage,
        gbufferNormal.layout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    gbufferNormal.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(
        gbufferMaterial.texture.textureImage,
        gbufferMaterial.layout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    gbufferMaterial.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(
        gbufferDepth.texture.textureImage,
        gbufferDepth.layout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth);
    gbufferDepth.layout = vk::ImageLayout::eShaderReadOnlyOptimal;

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

    vk::RenderingAttachmentInfo swapchainAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(std::array<float, 4>{0.03f, 0.03f, 0.03f, 1.0f})
    };

    vk::RenderingInfo lightingRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &swapchainAttachment
    };

    commandBuffer.beginRendering(lightingRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *deferredPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *deferredPipelineLayout, 0, *lightUboResources.descriptorSets[currentFrame], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

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

    commandBuffer.beginRendering(uiRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, float(swapChainExtent.width), float(swapChainExtent.height), 0.0f, 1.0f));
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

void DeferredRenderer::render()
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
        if (!recreateDeferredSizedResources())
        {
            throw std::runtime_error("failed to recreate deferred resources after swapchain resize");
        }
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    updateUIFrame();
    updateDeferredBuffers(currentFrame);
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
    result = presentQueue.presentKHR(presentInfo);

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
        if (!recreateDeferredSizedResources())
        {
            throw std::runtime_error("failed to recreate deferred resources after present");
        }
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

#include "PostFXRenderer.h"

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
    glm::vec4 params0; // x: ambient, y: exposure, z: gamma, w: lightScale
    glm::ivec4 debug;  // x: debugView
};

struct PostFXSettingsUBO
{
    float exposure;
    float gamma;
    float bloomThreshold;
    float bloomIntensity;
    float chromaticAberration;
    float vignetteIntensity;
    int toneMappingMode;
    int bloomEnabled;
    int chromaticAberrationEnabled;
    int vignetteEnabled;
    int padding0;
    int padding1;
    int padding2;
};

void PostFXRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool PostFXRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 0.0f, 15.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 9_postfx");
}

bool PostFXRenderer::prepareResource()
{
    generateSphere(sphereMesh, 1.0f, 100);
    createVertexBuffer(sphereMesh);
    createIndexBuffer(sphereMesh);

    createDeferredBuffers();

    if (!createGBufferDescriptorSetLayout()) return false;
    if (!createGBufferDescriptorPool()) return false;
    if (!createGBufferResources()) return false;
    if (!createGBufferPipeline()) return false;
    createGBufferDescriptorSets();

    if (!createDeferredDescriptorSetLayout()) return false;
    if (!createDeferredDescriptorPool()) return false;
    if (!createDeferredPipeline()) return false;
    createDeferredDescriptorSets();

    if (!createPostDescriptorSetLayout()) return false;
    if (!createPostDescriptorPool()) return false;
    if (!createBloomExtractPipeline()) return false;
    if (!createBlurPipelines()) return false;
    if (!createBloomCompositePipeline()) return false;

    if (!createPostBuffers()) return false;
    createPostDescriptorSets();
    updatePostDescriptorSets();

    if (!initUI()) return false;

    return true;
}

void PostFXRenderer::cleanup()
{
    shutdownUI();

    // Destroy descriptor pools first (automatically frees all their sets)
    gbufferDescriptorPool = nullptr;
    deferredDescriptorPool = nullptr;
    bloomExtractDescriptorPool = nullptr;
    blurDescriptorPool = nullptr;
    bloomCompositeDescriptorPool = nullptr;
    uiDescriptorPool = nullptr;

    destroyPostBuffersPerFrame();
    destroyGBufferResources();
}

void PostFXRenderer::createDeferredBuffers()
{
    const uint32_t sc = static_cast<uint32_t>(swapChainImages.size());
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO), sc);
    createUniformBuffers(lightUboResources, sizeof(LightUBO), sc);
    createUniformBuffers(deferredSettingsUboResources, sizeof(DeferredSettingsUBO), sc);
    createUniformBuffers(postFxSettingsUboResources, sizeof(PostFXSettingsUBO), sc);
    createStorageBuffers(instanceBufferResources, sizeof(DeferredInstanceData) * instanceCount, vk::BufferUsageFlagBits::eStorageBuffer, sc);
}

bool PostFXRenderer::createGBufferDescriptorSetLayout()
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

bool PostFXRenderer::createGBufferDescriptorPool()
{
    try
    {
        const uint32_t sc = static_cast<uint32_t>(swapChainImages.size());
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = sc },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = sc },
        };

        gbufferDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = sc,
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

void PostFXRenderer::createGBufferDescriptorSets()
{
    const uint32_t sc = static_cast<uint32_t>(swapChainImages.size());
    std::vector<vk::DescriptorSetLayout> layouts(sc, *gbufferDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *gbufferDescriptorPool,
        .descriptorSetCount = sc,
        .pSetLayouts = layouts.data()
    };

    instanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < sc; ++i)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceBufferInfo{ .buffer = *instanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(DeferredInstanceData) * instanceCount };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *instanceBufferResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *instanceBufferResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool PostFXRenderer::createGBufferResources()
{
    try
    {
        destroyGBufferResources();

        // HDR formats for post-processing support
        gbuffer.albedo.format = vk::Format::eR16G16B16A16Sfloat;
        gbuffer.normal.format = vk::Format::eR16G16B16A16Sfloat;
        gbuffer.material.format = vk::Format::eR16G16B16A16Sfloat;
        gbuffer.depth.format = vk::Format::eD32Sfloat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbuffer.albedo.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbuffer.albedo.texture);
        gbuffer.albedo.texture.textureImageView = createImageView(gbuffer.albedo.texture.textureImage, gbuffer.albedo.format, vk::ImageAspectFlagBits::eColor, 1);
        gbuffer.albedo.layout = vk::ImageLayout::eUndefined;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbuffer.normal.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbuffer.normal.texture);
        gbuffer.normal.texture.textureImageView = createImageView(gbuffer.normal.texture.textureImage, gbuffer.normal.format, vk::ImageAspectFlagBits::eColor, 1);
        gbuffer.normal.layout = vk::ImageLayout::eUndefined;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbuffer.material.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbuffer.material.texture);
        gbuffer.material.texture.textureImageView = createImageView(gbuffer.material.texture.textureImage, gbuffer.material.format, vk::ImageAspectFlagBits::eColor, 1);
        gbuffer.material.layout = vk::ImageLayout::eUndefined;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, gbuffer.depth.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, gbuffer.depth.texture);
        gbuffer.depth.texture.textureImageView = createImageView(gbuffer.depth.texture.textureImage, gbuffer.depth.format, vk::ImageAspectFlagBits::eDepth, 1);
        gbuffer.depth.layout = vk::ImageLayout::eUndefined;

        // Samplers
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
            .maxLod = 1.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite,
            .unnormalizedCoordinates = vk::False
        };
        gbufferSampler = vk::raii::Sampler(device, samplerInfo);

        vk::SamplerCreateInfo postSamplerInfo{
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
            .maxLod = 1.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueBlack,
            .unnormalizedCoordinates = vk::False
        };
        postSampler = vk::raii::Sampler(device, postSamplerInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create gbuffer resources: " << e.what() << std::endl;
        return false;
    }
}

void PostFXRenderer::destroyGBufferResources()
{
    gbuffer.albedo.texture.textureImageView = nullptr;
    gbuffer.albedo.texture.textureImage = nullptr;
    gbuffer.albedo.texture.textureImageMemory = nullptr;
    gbuffer.normal.texture.textureImageView = nullptr;
    gbuffer.normal.texture.textureImage = nullptr;
    gbuffer.normal.texture.textureImageMemory = nullptr;
    gbuffer.material.texture.textureImageView = nullptr;
    gbuffer.material.texture.textureImage = nullptr;
    gbuffer.material.texture.textureImageMemory = nullptr;
    gbuffer.depth.texture.textureImageView = nullptr;
    gbuffer.depth.texture.textureImage = nullptr;
    gbuffer.depth.texture.textureImageMemory = nullptr;
    gbuffer.albedo.layout = vk::ImageLayout::eUndefined;
    gbuffer.normal.layout = vk::ImageLayout::eUndefined;
    gbuffer.material.layout = vk::ImageLayout::eUndefined;
    gbuffer.depth.layout = vk::ImageLayout::eUndefined;
}

bool PostFXRenderer::createGBufferPipeline()
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
            gbuffer.albedo.format,
            gbuffer.normal.format,
            gbuffer.material.format
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
                .depthAttachmentFormat = gbuffer.depth.format
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

bool PostFXRenderer::recreateGBufferSizedResources()
{
    // Clear old descriptor sets before recreating pools (forces RAII dtors to free on the OLD pool)
    instanceBufferResources.descriptorSets = vk::raii::DescriptorSets(nullptr);
    lightUboResources.descriptorSets = vk::raii::DescriptorSets(nullptr);
    if (!createGBufferResources()) return false;
    if (!createGBufferDescriptorPool()) return false;
    createGBufferDescriptorSets();
    if (!createDeferredDescriptorPool()) return false;
    createDeferredDescriptorSets();
    return true;
}

bool PostFXRenderer::createDeferredDescriptorSetLayout()
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

bool PostFXRenderer::createDeferredDescriptorPool()
{
    try
    {
        // 7 bindings per set: albedo/normal/material/depth (sampler) + scene/light/settings (buffer)
        // deferredDescriptorPool only serves the deferred lighting pass (lightUboResources.descriptorSets)
        const uint32_t sc = static_cast<uint32_t>(swapChainImages.size());
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = sc * 4u },
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = sc * 3u },
        };

        deferredDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = sc,
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

void PostFXRenderer::createDeferredDescriptorSets()
{
    const uint32_t sc = static_cast<uint32_t>(swapChainImages.size());
    std::vector<vk::DescriptorSetLayout> layouts(sc, *deferredDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *deferredDescriptorPool,
        .descriptorSetCount = sc,
        .pSetLayouts = layouts.data()
    };

    lightUboResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < sc; ++i)
    {
        vk::DescriptorImageInfo albedoInfo{ .sampler = gbufferSampler, .imageView = gbuffer.albedo.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo normalInfo{ .sampler = gbufferSampler, .imageView = gbuffer.normal.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo materialInfo{ .sampler = gbufferSampler, .imageView = gbuffer.material.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo depthInfo{ .sampler = gbufferSampler, .imageView = gbuffer.depth.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

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

bool PostFXRenderer::createDeferredPipeline()
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

        // Write to our HDR scene buffer (not directly to swapchain)
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

        // Output to HDR scene buffer
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
                .pColorAttachmentFormats = &gbuffer.albedo.format  // HDR scene
            }
        };

        deferredPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create deferred pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool PostFXRenderer::createPostBuffers()
{
    try
    {
        destroyPostBuffersPerFrame();
        postBuffers.resize(swapChainImages.size());
        blurBuffers.resize(swapChainImages.size());

        const uint32_t w = swapChainExtent.width;
        const uint32_t h = swapChainExtent.height;
        const uint32_t bw = (w + 3) / 4; // quarter-res bloom
        const uint32_t bh = (h + 3) / 4;

        for (uint32_t i = 0; i < swapChainImages.size(); ++i)
        {
            // HDR scene buffer
            createImage(w, h, 1, vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
                vk::MemoryPropertyFlagBits::eDeviceLocal, postBuffers[i].color);
            postBuffers[i].color.textureImageView = createImageView(postBuffers[i].color.textureImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1);
            postBuffers[i].colorLayout = vk::ImageLayout::eUndefined;

            // Quarter-res bloom buffer
            createImage(bw, bh, 1, vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eDeviceLocal, postBuffers[i].bloom);
            postBuffers[i].bloom.textureImageView = createImageView(postBuffers[i].bloom.textureImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1);
            postBuffers[i].bloomLayout = vk::ImageLayout::eUndefined;

            // Blur ping-pong buffers
            createImage(bw, bh, 1, vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eDeviceLocal, blurBuffers[i].horizontal);
            blurBuffers[i].horizontal.textureImageView = createImageView(blurBuffers[i].horizontal.textureImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1);
            blurBuffers[i].hLayout = vk::ImageLayout::eUndefined;

            createImage(bw, bh, 1, vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eDeviceLocal, blurBuffers[i].vertical);
            blurBuffers[i].vertical.textureImageView = createImageView(blurBuffers[i].vertical.textureImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1);
            blurBuffers[i].vLayout = vk::ImageLayout::eUndefined;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create post buffers: " << e.what() << std::endl;
        return false;
    }
}

void PostFXRenderer::destroyPostBuffersPerFrame()
{
    for (auto& pb : postBuffers)
    {
        pb.color.textureImageView = nullptr;
        pb.color.textureImage = nullptr;
        pb.color.textureImageMemory = nullptr;
        pb.bloom.textureImageView = nullptr;
        pb.bloom.textureImage = nullptr;
        pb.bloom.textureImageMemory = nullptr;
        pb.colorLayout = vk::ImageLayout::eUndefined;
        pb.bloomLayout = vk::ImageLayout::eUndefined;
    }
    for (auto& bb : blurBuffers)
    {
        bb.horizontal.textureImageView = nullptr;
        bb.horizontal.textureImage = nullptr;
        bb.horizontal.textureImageMemory = nullptr;
        bb.vertical.textureImageView = nullptr;
        bb.vertical.textureImage = nullptr;
        bb.vertical.textureImageMemory = nullptr;
        bb.hLayout = vk::ImageLayout::eUndefined;
        bb.vLayout = vk::ImageLayout::eUndefined;
    }
    postBuffers.clear();
    blurBuffers.clear();
}

void PostFXRenderer::destroyPostBuffers()
{
    destroyPostBuffersPerFrame();
}

bool PostFXRenderer::recreatePostSizedResources()
{
    // Clear old descriptor sets before recreating pools (forces RAII dtors to free on the OLD pool)
    bloomExtractDescriptorSets = vk::raii::DescriptorSets(nullptr);
    blurDescriptorSets = vk::raii::DescriptorSets(nullptr);
    blurHDescriptorSets = vk::raii::DescriptorSets(nullptr);
    bloomCompositeDescriptorSets = vk::raii::DescriptorSets(nullptr);
    if (!createPostBuffers()) return false;
    if (!createPostDescriptorPool()) return false;
    createPostDescriptorSets();
    updatePostDescriptorSets();
    return true;
}

bool PostFXRenderer::createPostDescriptorSetLayout()
{
    try
    {
        // Bloom extract: scene color input
        std::vector<vk::DescriptorSetLayoutBinding> bloomExtractBindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            {.binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };
        bloomExtractDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bloomExtractBindings.size()),
            .pBindings = bloomExtractBindings.data()
            });
        // Blur: sampler2D input
        std::vector<vk::DescriptorSetLayoutBinding> blurBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };
        blurDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(blurBindings.size()),
            .pBindings = blurBindings.data()
        });

        // Bloom composite: scene + bloom
        std::vector<vk::DescriptorSetLayoutBinding> compositeBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 2, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };
        bloomCompositeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(compositeBindings.size()),
            .pBindings = compositeBindings.data()
        });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create post descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool PostFXRenderer::createPostDescriptorPool()
{
    try
    {
        // 4 sets per frame: bloomExtract + blurH + blurV + bloomComposite
        uint32_t maxPostSets = static_cast<uint32_t>(swapChainImages.size()) * 4u;

        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = maxPostSets },
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = static_cast<uint32_t>(swapChainImages.size()) * 2u },
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = maxPostSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        bloomExtractDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        blurDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        bloomCompositeDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create post descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void PostFXRenderer::createPostDescriptorSets()
{
    const uint32_t sc = static_cast<uint32_t>(swapChainImages.size());

    // Bloom extract descriptor sets (per-frame)
    std::vector<vk::DescriptorSetLayout> extractLayouts(sc, *bloomExtractDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo extractAllocInfo{
        .descriptorPool = *bloomExtractDescriptorPool,
        .descriptorSetCount = sc,
        .pSetLayouts = extractLayouts.data()
    };
    bloomExtractDescriptorSets = vk::raii::DescriptorSets(device, extractAllocInfo);

    // Blur descriptor sets: separate sets for H-blur (reads bloom) and V-blur (reads blurH)
    std::vector<vk::DescriptorSetLayout> blurLayouts(sc, *blurDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo blurAllocInfo{
        .descriptorPool = *blurDescriptorPool,
        .descriptorSetCount = sc,
        .pSetLayouts = blurLayouts.data()
    };
    blurDescriptorSets = vk::raii::DescriptorSets(device, blurAllocInfo);

    std::vector<vk::DescriptorSetLayout> blurHLayouts(sc, *blurDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo blurHAllocInfo{
        .descriptorPool = *blurDescriptorPool,
        .descriptorSetCount = sc,
        .pSetLayouts = blurHLayouts.data()
    };
    blurHDescriptorSets = vk::raii::DescriptorSets(device, blurHAllocInfo);

    // Bloom composite descriptor sets
    std::vector<vk::DescriptorSetLayout> compositeLayouts(sc, *bloomCompositeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo compositeAllocInfo{
        .descriptorPool = *bloomCompositeDescriptorPool,
        .descriptorSetCount = sc,
        .pSetLayouts = compositeLayouts.data()
    };
    bloomCompositeDescriptorSets = vk::raii::DescriptorSets(device, compositeAllocInfo);

    // One-time initialization of image bindings
    // Bloom extract: binding=0 scene color image, binding=1 settings (updated per-frame via memcpy)
    {
        std::vector<vk::DescriptorImageInfo> sceneInfos;
        std::vector<vk::DescriptorBufferInfo> settingsInfos;
        for (uint32_t i = 0; i < sc; ++i) {
            sceneInfos.push_back({ .sampler = postSampler, .imageView = postBuffers[i].color.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal });
            settingsInfos.push_back({ .buffer = *postFxSettingsUboResources.Buffers[i], .offset = 0, .range = sizeof(PostFXSettingsUBO) });
        }
        std::vector<vk::WriteDescriptorSet> writes;
        for (uint32_t i = 0; i < sc; ++i) {
            writes.push_back({ .dstSet = *bloomExtractDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &sceneInfos[i] });
            writes.push_back({ .dstSet = *bloomExtractDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &settingsInfos[i] });
        }
        device.updateDescriptorSets(writes, nullptr);
    }
    // Blur H: reads bloom texture
    {
        std::vector<vk::DescriptorImageInfo> blurHInfos;
        for (uint32_t i = 0; i < sc; ++i) {
            blurHInfos.push_back({ .sampler = postSampler, .imageView = postBuffers[i].bloom.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal });
        }
        std::vector<vk::WriteDescriptorSet> writes;
        for (uint32_t i = 0; i < sc; ++i) {
            writes.push_back({ .dstSet = *blurHDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &blurHInfos[i] });
        }
        device.updateDescriptorSets(writes, nullptr);
    }
    // Blur V: reads blurH result
    {
        std::vector<vk::DescriptorImageInfo> blurVInfos;
        for (uint32_t i = 0; i < sc; ++i) {
            blurVInfos.push_back({ .sampler = postSampler, .imageView = blurBuffers[i].horizontal.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal });
        }
        std::vector<vk::WriteDescriptorSet> writes;
        for (uint32_t i = 0; i < sc; ++i) {
            writes.push_back({ .dstSet = *blurDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &blurVInfos[i] });
        }
        device.updateDescriptorSets(writes, nullptr);
    }
    // Bloom composite: binding=0 scene color, binding=1 bloom, binding=2 settings (updated per-frame via memcpy)
    {
        std::vector<vk::DescriptorImageInfo> sceneInfos;
        std::vector<vk::DescriptorImageInfo> bloomInfos;
        std::vector<vk::DescriptorBufferInfo> settingsInfos;
        for (uint32_t i = 0; i < sc; ++i) {
            sceneInfos.push_back({ .sampler = postSampler, .imageView = postBuffers[i].color.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal });
            bloomInfos.push_back({ .sampler = postSampler, .imageView = postBuffers[i].bloom.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal });
            settingsInfos.push_back({ .buffer = *postFxSettingsUboResources.Buffers[i], .offset = 0, .range = sizeof(PostFXSettingsUBO) });
        }
        std::vector<vk::WriteDescriptorSet> writes;
        for (uint32_t i = 0; i < sc; ++i) {
            writes.push_back({ .dstSet = *bloomCompositeDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &sceneInfos[i] });
            writes.push_back({ .dstSet = *bloomCompositeDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &bloomInfos[i] });
            writes.push_back({ .dstSet = *bloomCompositeDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &settingsInfos[i] });
        }
        device.updateDescriptorSets(writes, nullptr);
    }
}

void PostFXRenderer::updatePostDescriptorSets()
{
    // Scene color is set per-frame in updatePostDescriptorSetsPerFrame() during recording
    // Here we only update static bindings; per-frame bindings updated in command buffer
    (void)this;
}

bool PostFXRenderer::createBloomExtractPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "bloom_extract.spv"));

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

        bloomExtractPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*bloomExtractDescriptorSetLayout,
            .pushConstantRangeCount = 0
        });

        vk::Format bloomFormat = vk::Format::eR16G16B16A16Sfloat;
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
                .layout = bloomExtractPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &bloomFormat
            }
        };

        bloomExtractPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create bloom extract pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool PostFXRenderer::createBlurPipelines()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "blur.spv"));

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

        blurHPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*blurDescriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &blurPushConstRange
        });

        blurVPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*blurDescriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &blurPushConstRange
        });

        vk::Format blurFormat = vk::Format::eR16G16B16A16Sfloat;

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChainH = {
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
                .layout = blurHPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &blurFormat
            }
        };
        blurHPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChainH.get<vk::GraphicsPipelineCreateInfo>());

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChainV = {
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
                .layout = blurVPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &blurFormat
            }
        };
        blurVPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChainV.get<vk::GraphicsPipelineCreateInfo>());

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create blur pipelines: " << e.what() << std::endl;
        return false;
    }
}

bool PostFXRenderer::createBloomCompositePipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "bloom_composite.spv"));

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

        bloomCompositePipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*bloomCompositeDescriptorSetLayout,
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
                .layout = bloomCompositePipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat
            }
        };

        bloomCompositePipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create bloom composite pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool PostFXRenderer::initUI()
{
    if (!uiEnabled) return true;
    if (ImGui::GetCurrentContext() != nullptr) return true;

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

    uiFrameBuffers.resize(swapChainImages.size());
    return true;
}

void PostFXRenderer::shutdownUI()
{
    if (ImGui::GetCurrentContext() == nullptr) return;

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
    uiDescriptorSetLayout = vk::raii::DescriptorSetLayout(nullptr);
    // Destroy pool before sets so each set's destructor finds a valid pool.
    uiDescriptorPool = vk::raii::DescriptorPool(nullptr);
    uiDescriptorSets = vk::raii::DescriptorSets(nullptr);

    uiFontTexture.textureSampler = vk::raii::Sampler(nullptr);
    uiFontTexture.textureImageView = vk::raii::ImageView(nullptr);
    uiFontTexture.textureImage = vk::raii::Image(nullptr);
    uiFontTexture.textureImageMemory = vk::raii::DeviceMemory(nullptr);

    ImGui::DestroyContext();
}

void PostFXRenderer::updatePostFXUI()
{
    ImGui::SetNextWindowSize(ImVec2(500.0f, 480.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Post-Processing FX", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::TextUnformatted("Tone Mapping");
    ImGui::Separator();
    const char* tmModes[] = { "Reinhard", "ACES Filmic" };
    ImGui::Combo("Tone Mapping", &toneMappingMode, tmModes, 2);
    ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f);
    ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);

    ImGui::Spacing();
    ImGui::TextUnformatted("Bloom");
    ImGui::Separator();
    ImGui::Checkbox("Enable Bloom", &bloomEnabled);
    ImGui::SliderFloat("Threshold", &bloomThreshold, 0.0f, 2.0f);
    ImGui::SliderFloat("Intensity", &bloomIntensity, 0.0f, 4.0f);
    ImGui::SliderFloat("Radius", &bloomRadius, 1.0f, 8.0f);

    ImGui::Spacing();
    ImGui::TextUnformatted("Lens Effects");
    ImGui::Separator();
    ImGui::Checkbox("Chromatic Aberration", &chromaticAberrationEnabled);
    ImGui::SliderFloat("CA Strength", &chromaticAberrationStrength, 0.0f, 0.01f);
    ImGui::Checkbox("Vignette", &vignetteEnabled);
    ImGui::SliderFloat("Vignette Intensity", &vignetteIntensity, 0.0f, 1.5f);

    ImGui::Spacing();
    ImGui::TextUnformatted("Scene");
    ImGui::Separator();
    ImGui::Checkbox("Animate Lights", &animateLights);
    ImGui::SliderFloat("Light Scale", &lightIntensityScale, 0.1f, 4.0f);

    ImGui::End();
}

void PostFXRenderer::updateUIFrame(uint32_t imageIndex)
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr) return;

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height));
    io.DeltaTime = platform->frameTimer > 0.0f ? platform->frameTimer : (1.0f / 60.0f);

    double mouseX = 0.0, mouseY = 0.0;
    glfwGetCursorPos(platform->window, &mouseX, &mouseY);
    io.MousePos = ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
    io.MouseDown[0] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    io.MouseDown[1] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    io.MouseDown[2] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    ImGui::NewFrame();
    updatePostFXUI();
    ImGui::Render();
}

void PostFXRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr || uiPipeline == vk::raii::Pipeline(nullptr)) return;

    ImDrawData* drawData = ImGui::GetDrawData();
    if (drawData == nullptr || drawData->TotalVtxCount <= 0) return;

    auto& fb = uiFrameBuffers[imageIndex];
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

    struct UiPushConsts { glm::vec2 scale; glm::vec2 translate; };
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

void PostFXRenderer::updateDeferredBuffers(uint32_t imageIndex)
{
    // Scene UBO
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 120.0f),
        .view = camera.GetViewMatrix(),
        .camPos = glm::vec4(camera.Position, 1.0f)
    };
    sceneUbo.projection[1][1] *= -1;
    sceneUbo.invProjection = glm::inverse(sceneUbo.projection);
    sceneUbo.invView = glm::inverse(sceneUbo.view);
    std::memcpy(sceneUboResources.BuffersMapped[imageIndex], &sceneUbo, sizeof(sceneUbo));

    // Instance data
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
    std::memcpy(instanceBufferResources.BuffersMapped[imageIndex], instances.data(), sizeof(DeferredInstanceData) * instances.size());

    // Light UBO
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
    std::memcpy(lightUboResources.BuffersMapped[imageIndex], &lightUbo, sizeof(lightUbo));

    // Deferred settings UBO
    DeferredSettingsUBO settings{};
    settings.params0 = glm::vec4(ambientStrength, exposure, gamma, lightIntensityScale);
    settings.debug = glm::ivec4(debugView, 0, 0, 0);
    std::memcpy(deferredSettingsUboResources.BuffersMapped[imageIndex], &settings, sizeof(settings));
}

void PostFXRenderer::updatePostSettingsBuffers(uint32_t imageIndex)
{
    PostFXSettingsUBO settingsUBO{
        .exposure = exposure, .gamma = gamma, .bloomThreshold = bloomThreshold,
        .bloomIntensity = bloomIntensity, .chromaticAberration = chromaticAberrationStrength,
        .vignetteIntensity = vignetteIntensity, .toneMappingMode = toneMappingMode,
        .bloomEnabled = bloomEnabled ? 1 : 0,
        .chromaticAberrationEnabled = chromaticAberrationEnabled ? 1 : 0,
        .vignetteEnabled = vignetteEnabled ? 1 : 0
    };
    std::memcpy(postFxSettingsUboResources.BuffersMapped[imageIndex], &settingsUBO, sizeof(settingsUBO));
}

void PostFXRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    recordGBufferPass(commandBuffer, imageIndex);
    recordDeferredPass(commandBuffer, imageIndex);
    recordBloomExtractPass(commandBuffer, imageIndex);
    recordBlurPasses(commandBuffer, imageIndex);
    recordBloomCompositePass(commandBuffer, imageIndex);
    recordUIPass(commandBuffer, imageIndex);

    commandBuffer.end();
}

void PostFXRenderer::recordGBufferPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    // Transition GBuffer attachments to color/depth attachment optimal
    auto transitionGBufferForGBuffer = [&](GBufferAttachment& attach, vk::ImageLayout targetLayout,
        vk::AccessFlagBits2 srcAccess, vk::AccessFlagBits2 dstAccess,
        vk::PipelineStageFlags2 srcStage, vk::PipelineStageFlags2 dstStage,
        vk::ImageAspectFlags aspect, bool isDepth)
    {
        if (isDepth)
        {
            transition_image_layout(attach.texture.textureImage, attach.layout, targetLayout,
                {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                aspect);
        }
        else
        {
            transition_image_layout(attach.texture.textureImage, attach.layout, targetLayout,
                {}, vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eAllCommands, 
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, 
                aspect);
        }
        attach.layout = targetLayout;
    };

    transitionGBufferForGBuffer(gbuffer.albedo, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor, false);
    transitionGBufferForGBuffer(gbuffer.normal, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor, false);
    transitionGBufferForGBuffer(gbuffer.material, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor, false);
    transitionGBufferForGBuffer(gbuffer.depth, vk::ImageLayout::eDepthAttachmentOptimal,
        {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth, true);

    // Transition post scene buffer for writing
    transition_image_layout(postBuffers[imageIndex].color.textureImage,
        postBuffers[imageIndex].colorLayout, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    postBuffers[imageIndex].colorLayout = vk::ImageLayout::eColorAttachmentOptimal;

    std::array<vk::RenderingAttachmentInfo, 3> gbufferColorAttachments = {
        vk::RenderingAttachmentInfo{ .imageView = gbuffer.albedo.texture.textureImageView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) },
        vk::RenderingAttachmentInfo{ .imageView = gbuffer.normal.texture.textureImageView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = vk::ClearColorValue(std::array<float, 4>{0.5f, 0.5f, 1.0f, 1.0f}) },
        vk::RenderingAttachmentInfo{ .imageView = gbuffer.material.texture.textureImageView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.6f, 1.0f, 0.0f}) }
    };

    vk::RenderingAttachmentInfo gbufferDepthAttachment{
        .imageView = gbuffer.depth.texture.textureImageView,
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
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *gbufferPipelineLayout, 0, *instanceBufferResources.descriptorSets[imageIndex], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(sphereMesh.indices.size()), instanceCount, 0, 0, 0);
    commandBuffer.endRendering();
}

void PostFXRenderer::recordDeferredPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    // GBuffer: shader-read transitions (layout only, no swapchain access)
    auto transitionForShaderRead = [&](GBufferAttachment& attach, vk::ImageAspectFlags aspect, bool isDepth)
    {
        if (isDepth)
        {
            transition_image_layout(attach.texture.textureImage, attach.layout, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eFragmentShader, aspect);
        }
        else
        {
            transition_image_layout(attach.texture.textureImage, attach.layout, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader, aspect);
        }
        attach.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    };

    transitionForShaderRead(gbuffer.albedo, vk::ImageAspectFlagBits::eColor, false);
    transitionForShaderRead(gbuffer.normal, vk::ImageAspectFlagBits::eColor, false);
    transitionForShaderRead(gbuffer.material, vk::ImageAspectFlagBits::eColor, false);
    transitionForShaderRead(gbuffer.depth, vk::ImageAspectFlagBits::eDepth, true);

    vk::RenderingAttachmentInfo sceneAttachment{
        .imageView = postBuffers[imageIndex].color.textureImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(std::array<float, 4>{0.03f, 0.03f, 0.03f, 1.0f})
    };

    vk::RenderingInfo lightingRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &sceneAttachment
    };

    commandBuffer.beginRendering(lightingRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *deferredPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *deferredPipelineLayout, 0, *lightUboResources.descriptorSets[imageIndex], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    postBuffers[imageIndex].colorLayout = vk::ImageLayout::eColorAttachmentOptimal;
}

void PostFXRenderer::recordBloomExtractPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    // Transition post scene buffer to shader read
    transition_image_layout(postBuffers[imageIndex].color.textureImage,
        postBuffers[imageIndex].colorLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    postBuffers[imageIndex].colorLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    // Transition bloom buffer for writing
    transition_image_layout(postBuffers[imageIndex].bloom.textureImage,
        postBuffers[imageIndex].bloomLayout, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    postBuffers[imageIndex].bloomLayout = vk::ImageLayout::eColorAttachmentOptimal;

    uint32_t bw = (swapChainExtent.width + 3) / 4;
    uint32_t bh = (swapChainExtent.height + 3) / 4;

    vk::RenderingAttachmentInfo bloomExtractAttachment{
        .imageView = postBuffers[imageIndex].bloom.textureImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f})
    };

    vk::RenderingInfo bloomExtractRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = vk::Extent2D{bw, bh} },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &bloomExtractAttachment
    };

    commandBuffer.beginRendering(bloomExtractRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(bw), static_cast<float>(bh), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D{bw, bh}));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomExtractPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *bloomExtractPipelineLayout, 0, *bloomExtractDescriptorSets[imageIndex], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();
}

void PostFXRenderer::recordBlurPasses(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    uint32_t bw = (swapChainExtent.width + 3) / 4;
    uint32_t bh = (swapChainExtent.height + 3) / 4;
    int blurIterations = static_cast<int>(std::round(bloomRadius));
    if (blurIterations <= 0) return;

    // Fixed roles: blurBuffers.horizontal = H-blur render target,
    // blurBuffers.vertical = V-blur render target.
    // After each HV pair, vertical result is copied back to bloom.

    // Layout of bloom texture
    vk::ImageLayout bloomLayout = postBuffers[imageIndex].bloomLayout;
    // Layout of blurH texture
    vk::ImageLayout hLayout = blurBuffers[imageIndex].hLayout;
    // Layout of blurV texture
    vk::ImageLayout vLayout = blurBuffers[imageIndex].vLayout;

    // For the next H-blur: bloom must be in SRO (will be transitioned from bloomLayout)
    vk::ImageLayout bloomSrcLayout = bloomLayout;

    for (int iter = 0; iter < blurIterations; ++iter)
    {
        // H-blur: bloom -> blurH
        // bloom: TDO→SRO needs srcAccess=TransferWrite since it was last written as a transfer dest
        transition_image_layout(postBuffers[imageIndex].bloom.textureImage, bloomSrcLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor);
        transition_image_layout(blurBuffers[imageIndex].horizontal.textureImage, hLayout, vk::ImageLayout::eColorAttachmentOptimal,
            {}, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

        vk::RenderingAttachmentInfo blurHAttachment{
            .imageView = blurBuffers[imageIndex].horizontal.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f})
        };
        vk::RenderingInfo blurHRenderingInfo{
            .renderArea = { .offset = {0, 0}, .extent = vk::Extent2D{bw, bh} },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &blurHAttachment
        };

        commandBuffer.beginRendering(blurHRenderingInfo);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(bw), static_cast<float>(bh), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D{bw, bh}));
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *blurHPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blurHPipelineLayout, 0, *blurHDescriptorSets[imageIndex], nullptr);

        struct BlurPushConsts { glm::vec2 texelSize; glm::vec3 padding; };
        BlurPushConsts blurConstH{ glm::vec2(4.0f / static_cast<float>(bw), 0.0f), glm::vec3(0.0f) };
        commandBuffer.pushConstants(*blurHPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, vk::ArrayProxy<const BlurPushConsts>(1, &blurConstH));
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRendering();

        // Sync local layout vars immediately after render ends:
        // bloom is now SRO (H-blur fragment shader read), hLayout tracks blurH's post-barrier state.
        bloomSrcLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        blurBuffers[imageIndex].hLayout = vk::ImageLayout::eColorAttachmentOptimal;
        hLayout = vk::ImageLayout::eColorAttachmentOptimal;

        // V-blur: blurH -> blurV
        transition_image_layout(blurBuffers[imageIndex].horizontal.textureImage, hLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor);
        transition_image_layout(blurBuffers[imageIndex].vertical.textureImage, vLayout, vk::ImageLayout::eColorAttachmentOptimal,
            {}, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

        vk::RenderingAttachmentInfo blurVAttachment{
            .imageView = blurBuffers[imageIndex].vertical.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f})
        };
        vk::RenderingInfo blurVRenderingInfo{
            .renderArea = { .offset = {0, 0}, .extent = vk::Extent2D{bw, bh} },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &blurVAttachment
        };

        commandBuffer.beginRendering(blurVRenderingInfo);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(bw), static_cast<float>(bh), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D{bw, bh}));
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *blurVPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blurVPipelineLayout, 0, *blurDescriptorSets[imageIndex], nullptr);

        BlurPushConsts blurConstV{ glm::vec2(0.0f, 4.0f / static_cast<float>(bh)), glm::vec3(0.0f) };
        commandBuffer.pushConstants(*blurVPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, vk::ArrayProxy<const BlurPushConsts>(1, &blurConstV));
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRendering();

        // Update tracked layouts after V-blur render
        blurBuffers[imageIndex].hLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        blurBuffers[imageIndex].vLayout = vk::ImageLayout::eColorAttachmentOptimal;
        hLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        vLayout = vk::ImageLayout::eColorAttachmentOptimal;

        // Copy blurV result back to bloom
        transition_image_layout(blurBuffers[imageIndex].vertical.textureImage, vLayout, vk::ImageLayout::eTransferSrcOptimal,
            {}, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageAspectFlagBits::eColor);
        transition_image_layout(postBuffers[imageIndex].bloom.textureImage, bloomSrcLayout, vk::ImageLayout::eTransferDstOptimal,
            {}, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageAspectFlagBits::eColor);

        vk::ImageCopy copyRegion{
            .srcSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .dstSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .extent = { bw, bh, 1 }
        };
        commandBuffer.copyImage(*blurBuffers[imageIndex].vertical.textureImage, vk::ImageLayout::eTransferSrcOptimal,
            *postBuffers[imageIndex].bloom.textureImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

        // bloom = copy dst; blurV = copy src (stays in TSO).
        // hLayout is already SRO (V-blur barrier endpoint), ready for next V-blur barrier.
        bloomSrcLayout = vk::ImageLayout::eTransferDstOptimal;
        vLayout = vk::ImageLayout::eTransferSrcOptimal;
    }

    // Update tracked layouts for recordBloomCompositePass
    // blurV is the copy src (→ TransferSrcOptimal), bloom is the copy dst (→ TransferDstOptimal)
    blurBuffers[imageIndex].hLayout = hLayout;
    blurBuffers[imageIndex].vLayout = vLayout;
    postBuffers[imageIndex].bloomLayout = vk::ImageLayout::eTransferDstOptimal;
}

void PostFXRenderer::recordBloomCompositePass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    // Transition scene buffer and bloom for composite shader read
    transition_image_layout(postBuffers[imageIndex].color.textureImage,
        postBuffers[imageIndex].colorLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    postBuffers[imageIndex].colorLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(postBuffers[imageIndex].bloom.textureImage,
        postBuffers[imageIndex].bloomLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    postBuffers[imageIndex].bloomLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    // Transition swapchain for final output
    transition_image_layout(swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    vk::RenderingAttachmentInfo compositeAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(std::array<float, 4>{0.03f, 0.03f, 0.03f, 1.0f})
    };

    vk::RenderingInfo compositeRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &compositeAttachment
    };

    commandBuffer.beginRendering(compositeRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomCompositePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *bloomCompositePipelineLayout, 0, *bloomCompositeDescriptorSets[imageIndex], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();
}

void PostFXRenderer::recordUIPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    // Transition swapchain for UI render
    transition_image_layout(swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

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
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUI(commandBuffer, imageIndex);
    commandBuffer.endRendering();

    // Transition to present
    transition_image_layout(swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex], vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite, {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;
}

void PostFXRenderer::render()
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
        if (!recreateGBufferSizedResources()) throw std::runtime_error("failed to recreate gbuffer resources");
        if (!recreatePostSizedResources()) throw std::runtime_error("failed to recreate post resources");
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    updateUIFrame(imageIndex);
    updateDeferredBuffers(imageIndex);
    updatePostSettingsBuffers(imageIndex);

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
        if (!recreateGBufferSizedResources()) throw std::runtime_error("failed to recreate gbuffer resources");
        if (!recreatePostSizedResources()) throw std::runtime_error("failed to recreate post resources");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

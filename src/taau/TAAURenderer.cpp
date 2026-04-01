#include "TAAURenderer.h"

#include <Base/Mesh.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <imgui.h>

struct ShadowInstanceData
{
    glm::mat4 model;
    glm::vec4 color;
};

struct SceneUBO
{
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec3 camPos;
};

#include <glm/gtc/matrix_transform.hpp>

void TAAURenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool TAAURenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 0.0f, 5.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 5_taau");
}

bool TAAURenderer::prepareResource()
{
    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    createShadowBuffers();

    if (!createShadowDescriptorSetLayout()) return false;
    if (!createShadowDescriptorPool()) return false;
    if (!createShadowMapResources()) return false;
    if (!createShadowPipelines()) return false;
    createShadowDescriptorSets();

    if (!createTAAUResources()) return false;
    if (!createTAAUDescriptorSetLayout()) return false;
    if (!createTAAUDescriptorPool()) return false;
    createTAAUDescriptorSets();
    if (!createTAAUPipeline()) return false;

    if (!initUI()) return false;

    return true;
}

bool TAAURenderer::createShadowDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 2, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 4, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };

        shadowDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create shadow descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool TAAURenderer::createShadowDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        };

        shadowDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create shadow descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void TAAURenderer::createShadowDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *shadowDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *shadowDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    shadowInstanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceBufferInfo{ .buffer = *shadowInstanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(ShadowInstanceData) * instanceCount };
        vk::DescriptorBufferInfo shadowBufferInfo{ .buffer = *shadowUboResources.Buffers[i], .offset = 0, .range = sizeof(ShadowUBO) };
        vk::DescriptorBufferInfo shadowParamsBufferInfo{ .buffer = *shadowParamsUboResources.Buffers[i], .offset = 0, .range = sizeof(ShadowParamsUBO) };
        vk::DescriptorImageInfo shadowMapInfo{ .sampler = shadowMapData.textureSampler, .imageView = shadowMapData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *shadowInstanceBufferResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *shadowInstanceBufferResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
            { .dstSet = *shadowInstanceBufferResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &shadowBufferInfo },
            { .dstSet = *shadowInstanceBufferResources.descriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowMapInfo },
            { .dstSet = *shadowInstanceBufferResources.descriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &shadowParamsBufferInfo },
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

void TAAURenderer::createShadowBuffers()
{
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(shadowUboResources, sizeof(ShadowUBO));
    createUniformBuffers(shadowParamsUboResources, sizeof(ShadowParamsUBO));

    // Scene instances: 1 floor + 8 cubes.
    instanceCount = 9;
    createStorageBuffers(shadowInstanceBufferResources, sizeof(ShadowInstanceData) * instanceCount);
}

bool TAAURenderer::createShadowMapResources()
{
    try
    {
        vk::Format shadowDepthFormat = findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD16Unorm },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
        );

        createImage(
            shadowMapExtent.width,
            shadowMapExtent.height,
            1,
            shadowDepthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            shadowMapData
        );
        shadowMapData.textureImageView = createImageView(shadowMapData.textureImage, shadowDepthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToBorder,
            .addressModeV = vk::SamplerAddressMode::eClampToBorder,
            .addressModeW = vk::SamplerAddressMode::eClampToBorder,
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
        shadowMapData.textureSampler = vk::raii::Sampler(device, samplerInfo);

        shadowMapLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create shadow map resources: " << e.what() << std::endl;
        return false;
    }
}

bool TAAURenderer::createShadowPipelines()
{
    try
    {
        shadowPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*shadowDescriptorSetLayout,
            .pushConstantRangeCount = 0
        });

        const auto bindingDescription = Vertex::getBindingDescription();
        const auto shadowDepthAttributes = Vertex::getPositionOnlyAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo shadowDepthVertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(shadowDepthAttributes.size()),
            .pVertexAttributeDescriptions = shadowDepthAttributes.data()
        };

        const auto shadowLitAttributes = Vertex::getPositionNormalAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo shadowLitVertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(shadowLitAttributes.size()),
            .pVertexAttributeDescriptions = shadowLitAttributes.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo shadowRasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::True,
            .depthBiasConstantFactor = 1.25f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 1.75f,
            .lineWidth = 1.0f
        };

        vk::PipelineRasterizationStateCreateInfo litRasterizer{
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
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLessOrEqual,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        const vk::Format shadowDepthFormat = findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD16Unorm },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
        );

        {
            vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "shadow_depth.spv"));
            vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
            vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
            vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

            vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 0, .pAttachments = nullptr };

            vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
                {
                    .stageCount = 2,
                    .pStages = shaderStages,
                    .pVertexInputState = &shadowDepthVertexInputInfo,
                    .pInputAssemblyState = &inputAssembly,
                    .pViewportState = &viewportState,
                    .pRasterizationState = &shadowRasterizer,
                    .pMultisampleState = &multisampling,
                    .pDepthStencilState = &depthStencil,
                    .pColorBlendState = &colorBlending,
                    .pDynamicState = &dynamicState,
                    .layout = shadowPipelineLayout,
                    .renderPass = nullptr
                },
                {
                    .colorAttachmentCount = 0,
                    .pColorAttachmentFormats = nullptr,
                    .depthAttachmentFormat = shadowDepthFormat
                }
            };

            shadowDepthPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        }

        {
            vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "shadow_lit.spv"));
            vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
            vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
            vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

            std::array<vk::PipelineColorBlendAttachmentState, 2> colorBlendAttachments = {
                vk::PipelineColorBlendAttachmentState{
                    .blendEnable = vk::False,
                    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                },
                vk::PipelineColorBlendAttachmentState{
                    .blendEnable = vk::False,
                    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                }
            };
            vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()), .pAttachments = colorBlendAttachments.data() };

            const vk::Format depthFormat = findDepthFormat();
            const std::array<vk::Format, 2> colorFormats{ swapChainImageFormat, vk::Format::eR16G16Sfloat };
            vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
                {
                    .stageCount = 2,
                    .pStages = shaderStages,
                    .pVertexInputState = &shadowLitVertexInputInfo,
                    .pInputAssemblyState = &inputAssembly,
                    .pViewportState = &viewportState,
                    .pRasterizationState = &litRasterizer,
                    .pMultisampleState = &multisampling,
                    .pDepthStencilState = &depthStencil,
                    .pColorBlendState = &colorBlending,
                    .pDynamicState = &dynamicState,
                    .layout = shadowPipelineLayout,
                    .renderPass = nullptr
                },
                {
                    .colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
                    .pColorAttachmentFormats = colorFormats.data(),
                    .depthAttachmentFormat = depthFormat
                }
            };

            shadowLitPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create shadow pipelines: " << e.what() << std::endl;
        return false;
    }
}

void TAAURenderer::updateShadowBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };

    sceneUbo.projection[2][0] += taauJitterCurrent.x * 2.0f;
    sceneUbo.projection[2][1] += taauJitterCurrent.y * 2.0f;
    sceneUbo.projection[1][1] *= -1;

    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    float deltaTime = platform ? platform->frameTimer : 0.0f;
    static float lightAngle = 0.0f;
    lightAngle += deltaTime * 0.35f;

    const glm::vec3 lightDir = glm::normalize(glm::vec3(std::cos(lightAngle) * 0.6f, -1.0f, std::sin(lightAngle) * 0.6f));
    const glm::vec3 lightPos = -lightDir * 12.0f;
    const glm::vec3 target(0.0f, -0.5f, 0.0f);

    const glm::mat4 lightView = glm::lookAt(lightPos, target, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProj = glm::ortho(-7.0f, 7.0f, -7.0f, 7.0f, 0.1f, 30.0f);
    lightProj[1][1] *= -1;

    const glm::mat4 currentViewProj = sceneUbo.projection * sceneUbo.view;

    ShadowUBO shadowUbo{
        .lightViewProj = lightProj * lightView,
        .prevViewProj = taauPrevViewProj,
        .dirLightDirIntensity = glm::vec4(lightDir, dirLightIntensity),
        .dirLightColor = glm::vec4(1.0f),
        .pointLightPosIntensity = glm::vec4(0.0f, 2.0f, 2.0f, pointLightIntensity),
        .pointLightColor = glm::vec4(1.0f),
        .areaLightPosIntensity = glm::vec4(-2.5f, 2.5f, -1.2f, areaLightIntensity),
        .areaLightColor = glm::vec4(1.0f, 0.95f, 0.85f, 1.0f),
        .areaLightU = glm::vec4(0.9f, 0.0f, 0.0f, 0.0f),
        .areaLightV = glm::vec4(0.0f, 0.0f, 0.9f, 0.0f),
    };
    std::memcpy(shadowUboResources.BuffersMapped[frameIndex], &shadowUbo, sizeof(shadowUbo));

    updateTAAUHistory(currentViewProj);

    ShadowParamsUBO shadowParams{
        .shadowFilterMode = shadowFilterMode,
        .pcfRadiusTexels = pcfRadiusTexels,
        .pcssLightSizeTexels = pcssLightSizeTexels,
        .shadowBiasMin = 0.0006f,
        .invShadowMapSize = glm::vec2(1.0f / float(shadowMapExtent.width), 1.0f / float(shadowMapExtent.height)),
        .padding0 = glm::vec2(0.0f)
    };
    std::memcpy(shadowParamsUboResources.BuffersMapped[frameIndex], &shadowParams, sizeof(shadowParams));

    std::vector<ShadowInstanceData> instances;
    instances.reserve(instanceCount);

    ShadowInstanceData floor{};
    floor.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -2.0f, 0.0f));
    floor.model = glm::scale(floor.model, glm::vec3(8.0f, 0.25f, 8.0f));
    floor.color = glm::vec4(0.55f, 0.55f, 0.60f, 1.0f);
    instances.push_back(floor);

    taauSceneTime += deltaTime;
    for (uint32_t i = 0; i < 8; ++i)
    {
        const float a = static_cast<float>(i) / 8.0f * 6.2831853f;

        glm::vec3 pos(std::cos(a) * 2.8f, -1.0f, std::sin(a) * 2.8f);
        float yaw = a;

        if (!taauFreezeHistory && i == 1)
        {
            float fastPhase = taauSceneTime * 2.4f;
            float zigzag = std::sin(fastPhase);
            float offset = std::sin(fastPhase * 1.5f) * 0.6f;
            pos.x = 2.5f + zigzag * 1.2f;
            pos.z = -1.6f + offset;
        }

        if (i == 5)
        {
            pos.x = 3.7f + std::sin(taauSceneTime * 0.6f) * 0.25f;
        }

        glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
        model = glm::rotate(model, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(0.65f));

        const glm::vec3 c = (i & 1u) ? glm::vec3(0.92f, 0.30f, 0.25f) : glm::vec3(0.25f, 0.65f, 0.92f);
        instances.push_back({ model, glm::vec4(c, 1.0f) });
    }

    std::memcpy(shadowInstanceBufferResources.BuffersMapped[frameIndex], instances.data(), sizeof(ShadowInstanceData) * instances.size());
}

float TAAURenderer::halton(uint32_t index, uint32_t base)
{
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0u)
    {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

bool TAAURenderer::createTAAUResources()
{
    try
    {
        const uint32_t scaledWidth = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * taauRenderScale));
        const uint32_t scaledHeight = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * taauRenderScale));

        taauInputColorData.mipLevels = 1;
        createImage(scaledWidth, scaledHeight, 1, swapChainImageFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal, taauInputColorData);
        taauInputColorData.textureImageView = createImageView(taauInputColorData.textureImage, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);

        taauVelocityData.mipLevels = 1;
        createImage(scaledWidth, scaledHeight, 1, vk::Format::eR16G16Sfloat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, taauVelocityData);
        taauVelocityData.textureImageView = createImageView(taauVelocityData.textureImage, vk::Format::eR16G16Sfloat, vk::ImageAspectFlagBits::eColor, 1);

        taauDepthData.mipLevels = 1;
        vk::Format taauDepthFormat = findDepthFormat();
        createImage(scaledWidth, scaledHeight, 1, taauDepthFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, taauDepthData);
        taauDepthData.textureImageView = createImageView(taauDepthData.textureImage, taauDepthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        for (int i = 0; i < 2; ++i)
        {
            taauHistoryColorData[i].mipLevels = 1;
            createImage(swapChainExtent.width, swapChainExtent.height, 1, swapChainImageFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal, taauHistoryColorData[i]);
            taauHistoryColorData[i].textureImageView = createImageView(taauHistoryColorData[i].textureImage, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
            taauHistoryLayouts[i] = vk::ImageLayout::eUndefined;
        }

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
        taauColorSampler = vk::raii::Sampler(device, samplerInfo);
        taauDepthSampler = vk::raii::Sampler(device, samplerInfo);

        createUniformBuffers(taauParamsUboResources, sizeof(TAAUParamsUBO));

        taauInputLayout = vk::ImageLayout::eUndefined;
        taauVelocityLayout = vk::ImageLayout::eUndefined;
        taauDepthLayout = vk::ImageLayout::eUndefined;
        taauHistoryReadIndex = 0;
        taauHistoryValid = false;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create TAAU resources: " << e.what() << std::endl;
        return false;
    }
}

bool TAAURenderer::createTAAUDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 4, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        taauDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create TAAU descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool TAAURenderer::createTAAUDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u},
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT}
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        taauDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create TAAU descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void TAAURenderer::createTAAUDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *taauDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *taauDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    taauDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        updateTAAUDescriptorSet(i, taauHistoryReadIndex);
    }
}

void TAAURenderer::updateTAAUDescriptorSet(uint32_t frameIndex, uint32_t historyReadIndex)
{
    vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *taauParamsUboResources.Buffers[frameIndex], .offset = 0, .range = sizeof(TAAUParamsUBO) };

    vk::DescriptorImageInfo inputInfo{
        .sampler = *taauColorSampler,
        .imageView = *taauInputColorData.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::DescriptorImageInfo historyInfo{
        .sampler = *taauColorSampler,
        .imageView = *taauHistoryColorData[historyReadIndex].textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::DescriptorImageInfo velocityInfo{
        .sampler = *taauColorSampler,
        .imageView = *taauVelocityData.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::DescriptorImageInfo depthInfo{
        .sampler = *taauDepthSampler,
        .imageView = *taauDepthData.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    std::vector<vk::WriteDescriptorSet> writes = {
        {.dstSet = *taauDescriptorSets[frameIndex], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &inputInfo},
        {.dstSet = *taauDescriptorSets[frameIndex], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &historyInfo},
        {.dstSet = *taauDescriptorSets[frameIndex], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &velocityInfo},
        {.dstSet = *taauDescriptorSets[frameIndex], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
        {.dstSet = *taauDescriptorSets[frameIndex], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo},
    };

    device.updateDescriptorSets(writes, nullptr);
}

bool TAAURenderer::createTAAUPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "taau_resolve.spv"));
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount = 0, .vertexAttributeDescriptionCount = 0 };
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
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*taauDescriptorSetLayout, .pushConstantRangeCount = 0 };
        taauPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
                .layout = taauPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = vk::Format::eUndefined
            }
        };

        taauPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create TAAU pipeline: " << e.what() << std::endl;
        return false;
    }
}

void TAAURenderer::updateTAAUBuffers(uint32_t currentImage)
{
    taauJitterPrev = taauJitterCurrent;
    const uint32_t sampleIndex = static_cast<uint32_t>((taauFrameCounter % 8u) + 1u);
    const float hx = halton(sampleIndex, 2u) - 0.5f;
    const float hy = halton(sampleIndex, 3u) - 0.5f;
    const float fullWidth = std::max(1.0f, static_cast<float>(swapChainExtent.width));
    const float fullHeight = std::max(1.0f, static_cast<float>(swapChainExtent.height));
    const float jitterScale = 0.45f;
    taauJitterCurrent = glm::vec2(hx / fullWidth, hy / fullHeight) * jitterScale;
    taauFrameCounter++;

    TAAUParamsUBO params{
        .blendFactor = taauParams.blendFactor,
        .reactiveClamp = taauParams.reactiveClamp,
        .antiFlicker = taauParams.antiFlicker,
        .velocityScale = taauParams.velocityScale,
        .historyClampGamma = taauParams.historyClampGamma,
        .historyRejectThreshold = taauParams.historyRejectThreshold,
        .pad0 = 0.0f,
        .pad1 = 0.0f,
    };

    std::memcpy(taauParamsUboResources.BuffersMapped[currentImage], &params, sizeof(params));
}

void TAAURenderer::updateTAAUHistory(const glm::mat4& currentViewProj)
{
    taauPrevViewProj = currentViewProj;
}

void TAAURenderer::recordTAAU(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    if (!taauEnabled)
    {
        transition_image_layout(taauInputColorData.textureImage, taauInputLayout, vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eTransfer, vk::ImageAspectFlagBits::eColor);
        taauInputLayout = vk::ImageLayout::eTransferSrcOptimal;

        transition_image_layout(swapChainImages[imageIndex], swapChainImageLayouts[imageIndex], vk::ImageLayout::eTransferDstOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eTransfer, vk::ImageAspectFlagBits::eColor);
        swapChainImageLayouts[imageIndex] = vk::ImageLayout::eTransferDstOptimal;

        const vk::Extent2D taauExtent{
            std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * taauRenderScale)),
            std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * taauRenderScale))
        };
        vk::ImageBlit blitRegion{
            .srcSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .srcOffsets = std::array<vk::Offset3D, 2>{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{static_cast<int32_t>(taauExtent.width), static_cast<int32_t>(taauExtent.height), 1}
            },
            .dstSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .dstOffsets = std::array<vk::Offset3D, 2>{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{static_cast<int32_t>(swapChainExtent.width), static_cast<int32_t>(swapChainExtent.height), 1}
            }
        };
        commandBuffer.blitImage(taauInputColorData.textureImage, vk::ImageLayout::eTransferSrcOptimal,
            swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal, blitRegion, vk::Filter::eLinear);

        transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
            vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
        swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;
        return;
    }

    const uint32_t historyRead = taauHistoryReadIndex;
    const uint32_t historyWrite = (historyRead + 1u) % 2u;

    transition_image_layout(taauInputColorData.textureImage, taauInputLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
    taauInputLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(taauVelocityData.textureImage, taauVelocityLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
    taauVelocityLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(taauDepthData.textureImage, taauDepthLayout, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eDepth);
    taauDepthLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    if (!taauHistoryValid)
    {
        transition_image_layout(taauInputColorData.textureImage, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits2::eShaderSampledRead, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::PipelineStageFlagBits2::eTransfer, vk::ImageAspectFlagBits::eColor);

        transition_image_layout(taauHistoryColorData[historyRead].textureImage, taauHistoryLayouts[historyRead], vk::ImageLayout::eTransferDstOptimal,
            {}, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eTransfer, vk::ImageAspectFlagBits::eColor);

        const vk::Extent2D taauExtent{
            std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * taauRenderScale)),
            std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * taauRenderScale))
        };
        vk::ImageBlit initBlit{
            .srcSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .srcOffsets = std::array<vk::Offset3D, 2>{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{static_cast<int32_t>(taauExtent.width), static_cast<int32_t>(taauExtent.height), 1}
            },
            .dstSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .dstOffsets = std::array<vk::Offset3D, 2>{
                vk::Offset3D{0, 0, 0},
                vk::Offset3D{static_cast<int32_t>(swapChainExtent.width), static_cast<int32_t>(swapChainExtent.height), 1}
            }
        };
        commandBuffer.blitImage(taauInputColorData.textureImage, vk::ImageLayout::eTransferSrcOptimal,
            taauHistoryColorData[historyRead].textureImage, vk::ImageLayout::eTransferDstOptimal, initBlit, vk::Filter::eLinear);

        transition_image_layout(taauHistoryColorData[historyRead].textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
        taauHistoryLayouts[historyRead] = vk::ImageLayout::eShaderReadOnlyOptimal;

        transition_image_layout(taauInputColorData.textureImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
    }

    if (taauHistoryLayouts[historyRead] != vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        transition_image_layout(taauHistoryColorData[historyRead].textureImage, taauHistoryLayouts[historyRead], vk::ImageLayout::eShaderReadOnlyOptimal,
            {}, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
        taauHistoryLayouts[historyRead] = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    updateTAAUDescriptorSet(currentFrame, historyRead);

    vk::RenderingAttachmentInfo resolveAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
    };
    vk::RenderingInfo resolveRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &resolveAttachmentInfo
    };

    commandBuffer.beginRendering(resolveRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *taauPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *taauPipelineLayout, 0, *taauDescriptorSets[currentFrame], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eTransfer, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eTransferSrcOptimal;

    transition_image_layout(taauHistoryColorData[historyWrite].textureImage, taauHistoryLayouts[historyWrite], vk::ImageLayout::eTransferDstOptimal,
        {}, vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eTransfer, vk::ImageAspectFlagBits::eColor);

    vk::ImageBlit historyBlit{
        .srcSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .srcOffsets = std::array<vk::Offset3D, 2>{
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int32_t>(swapChainExtent.width), static_cast<int32_t>(swapChainExtent.height), 1}
        },
        .dstSubresource = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .dstOffsets = std::array<vk::Offset3D, 2>{
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int32_t>(swapChainExtent.width), static_cast<int32_t>(swapChainExtent.height), 1}
        }
    };
    commandBuffer.blitImage(swapChainImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal,
        taauHistoryColorData[historyWrite].textureImage, vk::ImageLayout::eTransferDstOptimal,
        historyBlit, vk::Filter::eLinear);

    transition_image_layout(taauHistoryColorData[historyWrite].textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eColor);
    taauHistoryLayouts[historyWrite] = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    taauHistoryReadIndex = historyWrite;
    taauHistoryValid = true;
}

void TAAURenderer::updateTAAUUI()
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGui::Begin("TAAU Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("TAA stage-3 (jitter + rejection)");
    ImGui::BulletText("Current + reprojected history blend");
    ImGui::BulletText("Halton jitter enabled");
    ImGui::BulletText("Neighborhood clamp + history rejection");

    ImGui::Separator();
    ImGui::Checkbox("Enable TAA Resolve", &taauEnabled);
    ImGui::SliderFloat("BlendFactor", &taauParams.blendFactor, 0.2f, 0.98f);
    ImGui::SliderFloat("ReactiveClamp", &taauParams.reactiveClamp, 0.2f, 1.2f);
    ImGui::SliderFloat("AntiFlicker", &taauParams.antiFlicker, 0.0f, 1.0f);
    ImGui::SliderFloat("VelocityScale", &taauParams.velocityScale, 0.2f, 2.5f);
    ImGui::SliderFloat("ClampGamma", &taauParams.historyClampGamma, 0.5f, 2.5f);
    ImGui::SliderFloat("RejectThreshold", &taauParams.historyRejectThreshold, 0.01f, 0.5f);

    if (ImGui::SliderFloat("RenderScale", &taauRenderScale, 0.5f, 1.0f, "%.2f"))
    {
        taauHistoryValid = false;
        waitIdle();
        createTAAUResources();
        createTAAUDescriptorSets();
    }

    ImGui::Checkbox("FreezeHistory", &taauFreezeHistory);
    if (ImGui::Button("Reset History"))
    {
        taauHistoryValid = false;
    }
    ImGui::End();
}

bool TAAURenderer::initUI()
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

void TAAURenderer::updateUIFrame()
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
    updateTAAUUI();
    ImGui::Render();
}

void TAAURenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
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
        std::memcpy(vtxDst, cmdList->VtxBuffer.Data, static_cast<size_t>(cmdList->VtxBuffer.Size) * sizeof(ImDrawVert));
        std::memcpy(idxDst, cmdList->IdxBuffer.Data, static_cast<size_t>(cmdList->IdxBuffer.Size) * sizeof(ImDrawIdx));
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

void TAAURenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    transition_image_layout(swapChainImages[imageIndex], swapChainImageLayouts[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(depthData.textureImage, depthImageLayout, vk::ImageLayout::eDepthAttachmentOptimal,
        {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests, vk::ImageAspectFlagBits::eDepth);
    depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    transition_image_layout(shadowMapData.textureImage, shadowMapLayout, vk::ImageLayout::eDepthAttachmentOptimal,
        {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests, vk::ImageAspectFlagBits::eDepth);
    shadowMapLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    vk::ClearValue shadowClearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    vk::RenderingAttachmentInfo shadowDepthAttachmentInfo = {
        .imageView = shadowMapData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = shadowClearDepth
    };
    vk::RenderingInfo shadowRenderingInfo = {
        .renderArea = {.offset = {0, 0}, .extent = shadowMapExtent},
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &shadowDepthAttachmentInfo
    };

    auto drawShadowCubes = [&]() {
        if (instanceCount == 0) {
            return;
        }
        commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
        commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
        commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), instanceCount, 0, 0, 0);
    };

    commandBuffer.beginRendering(shadowRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(shadowMapExtent.width), static_cast<float>(shadowMapExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), shadowMapExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowDepthPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
    drawShadowCubes();
    commandBuffer.endRendering();

    transition_image_layout(shadowMapData.textureImage, vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests, vk::PipelineStageFlagBits2::eFragmentShader, vk::ImageAspectFlagBits::eDepth);
    shadowMapLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(taauInputColorData.textureImage, taauInputLayout, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    taauInputLayout = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(taauVelocityData.textureImage, taauVelocityLayout, vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);
    taauVelocityLayout = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(taauDepthData.textureImage, taauDepthLayout, vk::ImageLayout::eDepthAttachmentOptimal,
        {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests, vk::ImageAspectFlagBits::eDepth);
    taauDepthLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    const vk::Extent2D taauExtent{
        std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * taauRenderScale)),
        std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * taauRenderScale))
    };

    std::array<vk::RenderingAttachmentInfo, 2> taauColorAttachments = {
        vk::RenderingAttachmentInfo{
            .imageView = taauInputColorData.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
        },
        vk::RenderingAttachmentInfo{
            .imageView = taauVelocityData.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
        }
    };

    vk::RenderingAttachmentInfo taauDepthAttachmentInfo = {
        .imageView = taauDepthData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
    };

    vk::RenderingInfo taauRenderingInfo = {
        .renderArea = {.offset = {0, 0}, .extent = taauExtent},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(taauColorAttachments.size()),
        .pColorAttachments = taauColorAttachments.data(),
        .pDepthAttachment = &taauDepthAttachmentInfo
    };

    commandBuffer.beginRendering(taauRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(taauExtent.width), static_cast<float>(taauExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), taauExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowLitPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
    drawShadowCubes();
    commandBuffer.endRendering();

    recordTAAU(commandBuffer, imageIndex);

    vk::RenderingAttachmentInfo uiAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingInfo uiRenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachmentInfo
    };

    commandBuffer.beginRendering(uiRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUI(commandBuffer);
    commandBuffer.endRendering();

    transition_image_layout(swapChainImages[imageIndex], swapChainImageLayouts[imageIndex], vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite, {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eBottomOfPipe, vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;

    commandBuffer.end();
}

void TAAURenderer::render()
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
    updateTAAUBuffers(currentFrame);
    updateShadowBuffers(currentFrame);
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
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void TAAURenderer::cleanup()
{
    shutdownUI();
}

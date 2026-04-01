#include "ShadowRenderer.h"

#include <Base/Mesh.h>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cmath>
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

void ShadowRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool ShadowRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 1.5f, 9.5f));
    return VulkanBase::initVulkan("VulkanRenderer - 4_shadow");
}

bool ShadowRenderer::prepareResource()
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

    if (!initUI()) return false;

    return true;
}

void ShadowRenderer::createShadowBuffers()
{
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(shadowUboResources, sizeof(ShadowUBO));
    createUniformBuffers(shadowParamsUboResources, sizeof(ShadowParamsUBO));

    // Scene instances: 1 floor + 8 cubes.
    instanceCount = 9;
    createStorageBuffers(shadowInstanceBufferResources, sizeof(ShadowInstanceData) * instanceCount);
}

bool ShadowRenderer::createShadowDescriptorSetLayout()
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

bool ShadowRenderer::createShadowDescriptorPool()
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

void ShadowRenderer::createShadowDescriptorSets()
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

bool ShadowRenderer::createShadowMapResources()
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

bool ShadowRenderer::createShadowPipelines()
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

        // Depth-only pipeline (shadow map)
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

        // Lit pipeline (onscreen)
        {
            vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "shadow_lit.spv"));
            vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
            vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
            vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

            vk::PipelineColorBlendAttachmentState colorBlendAttachment{
                .blendEnable = vk::False,
                .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
            };
            vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

            const vk::Format depthFormat = findDepthFormat();
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
                    .colorAttachmentCount = 1,
                    .pColorAttachmentFormats = &swapChainImageFormat,
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

void ShadowRenderer::updateShadowBuffers(uint32_t frameIndex)
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

    float deltaTime = platform ? platform->frameTimer : 0.0f;
    static float lightAngle = 0.0f;
    lightAngle += deltaTime * 0.35f;

    const glm::vec3 lightDir = glm::normalize(glm::vec3(std::cos(lightAngle) * 0.6f, -1.0f, std::sin(lightAngle) * 0.6f));
    const glm::vec3 lightPos = -lightDir * 12.0f;
    const glm::vec3 target(0.0f, -0.5f, 0.0f);

    const glm::mat4 lightView = glm::lookAt(lightPos, target, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProj = glm::ortho(-7.0f, 7.0f, -7.0f, 7.0f, 0.1f, 30.0f);
    lightProj[1][1] *= -1;

    ShadowUBO shadowUbo{
        .lightViewProj = lightProj * lightView,
        .prevViewProj = glm::mat4(1.0f),
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

    // Floor
    {
        glm::mat4 model(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -2.0f, 0.0f));
        model = glm::scale(model, glm::vec3(8.0f, 0.25f, 8.0f));
        instances.push_back(ShadowInstanceData{ model, glm::vec4(0.55f, 0.55f, 0.60f, 1.0f) });
    }

    // Cubes around origin
    const glm::vec3 cubeColor0(0.92f, 0.30f, 0.25f);
    const glm::vec3 cubeColor1(0.25f, 0.65f, 0.92f);
    for (int i = 0; i < 8; ++i)
    {
        const float a = static_cast<float>(i) / 8.0f * 6.2831853f;
        glm::mat4 model(1.0f);
        model = glm::translate(model, glm::vec3(std::cos(a) * 2.8f, -1.0f, std::sin(a) * 2.8f));
        model = glm::rotate(model, a + lightAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(0.65f));
        const glm::vec3 c = (i & 1) ? cubeColor0 : cubeColor1;
        instances.push_back(ShadowInstanceData{ model, glm::vec4(c, 1.0f) });
    }

    std::memcpy(shadowInstanceBufferResources.BuffersMapped[frameIndex], instances.data(), sizeof(ShadowInstanceData) * instances.size());
}

bool ShadowRenderer::initUI()
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

void ShadowRenderer::updateShadowUI()
{
    ImGui::Begin("Shadows & Lights");
    const char* modes[] = { "Hard", "PCF", "PCSS" };
    ImGui::Combo("Shadow Filter", &shadowFilterMode, modes, 3);

    if (shadowFilterMode == 1)
    {
        ImGui::SliderFloat("PCF Radius (texels)", &pcfRadiusTexels, 0.0f, 6.0f);
    }
    if (shadowFilterMode == 2)
    {
        ImGui::SliderFloat("Light Size (texels)", &pcssLightSizeTexels, 1.0f, 80.0f);
    }

    ImGui::Separator();
    ImGui::SliderFloat("Dir Light Intensity", &dirLightIntensity, 0.0f, 20.0f);
    ImGui::SliderFloat("Point Light Intensity", &pointLightIntensity, 0.0f, 80.0f);
    ImGui::SliderFloat("Area Light Intensity", &areaLightIntensity, 0.0f, 80.0f);
    ImGui::End();
}

void ShadowRenderer::updateUIFrame()
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
    updateShadowUI();
    ImGui::Render();
}

void ShadowRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
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

void ShadowRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    // ---- Shadow depth pass ----
    transition_image_layout(
        shadowMapData.textureImage,
        shadowMapLayout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    shadowMapLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    const vk::ClearValue clearShadowDepth = vk::ClearDepthStencilValue(1.0f, 0);
    const vk::RenderingAttachmentInfo shadowDepthAttachment{
        .imageView = shadowMapData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearShadowDepth
    };
    const vk::RenderingInfo shadowRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = shadowMapExtent },
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &shadowDepthAttachment
    };

    commandBuffer.beginRendering(shadowRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, float(shadowMapExtent.width), float(shadowMapExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), shadowMapExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowDepthPipeline);
    commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), instanceCount, 0, 0, 0);
    commandBuffer.endRendering();

    // Make shadow map readable in the lit pass.
    transition_image_layout(
        shadowMapData.textureImage,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth);
    shadowMapLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    // ---- Lit pass ----
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

    const vk::ClearValue clearColor = vk::ClearColorValue(0.07f, 0.07f, 0.09f, 1.0f);
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
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, float(swapChainExtent.width), float(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowLitPipeline);
    commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), instanceCount, 0, 0, 0);
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

void ShadowRenderer::cleanup()
{
    shutdownUI();
}

void ShadowRenderer::render()
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
    commandBuffers[currentFrame].reset();

    updateUIFrame();
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



#include "CsmRenderer.h"

#include <Base/Mesh.h>
#include <Base/VulkanBase_UI.h>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <imgui.h>

// =============================================================================
// Initialization
// =============================================================================

void CsmRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool CsmRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 1.5f, 9.5f));
    return VulkanBase::initVulkan("VulkanRenderer - Cascaded Shadow Maps");
}

bool CsmRenderer::prepareResource()
{
    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    generateCube(planeMesh);
    createVertexBuffer(planeMesh);
    createIndexBuffer(planeMesh);

    createCsmBuffers();

    if (!createCsmDescriptorSetLayout()) return false;
    if (!createCsmDescriptorPool()) return false;
    if (!createCsmMapResources()) return false;
    createCsmDescriptorSets();
    if (!createCsmPipelines()) return false;

    if (!initUI()) return false;

    return true;
}

bool CsmRenderer::initUI()
{
    return initVulkanUI();
}

// =============================================================================
// Buffers
// =============================================================================

void CsmRenderer::createCsmBuffers()
{
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(csmUboResources, sizeof(CsmUBO));
    createUniformBuffers(shadowParamsUboResources, sizeof(ShadowParamsUBO));

    // 1 floor + 9 cubes
    instanceCount = 10;
    createStorageBuffers(instanceBufferResources, sizeof(InstanceData) * instanceCount);
}

// =============================================================================
// Descriptor Set Layout
// =============================================================================

bool CsmRenderer::createCsmDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            // binding 0: SceneUBO (camera VP)
            {.binding = 0,
              .descriptorType = vk::DescriptorType::eUniformBuffer,
              .descriptorCount = 1,
              .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
              // binding 1: InstanceData[] SSBO
              {.binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
                // binding 2: CsmUBO (4 cascade VPs + split depths)
                {.binding = 2,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
                  // binding 3: shadowMapArray (texture2DArray)
                  {.binding = 3,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eFragment },
                    // binding 4: ShadowParamsUBO
                    {.binding = 4,
                      .descriptorType = vk::DescriptorType::eUniformBuffer,
                      .descriptorCount = 1,
                      .stageFlags = vk::ShaderStageFlagBits::eFragment },
        };

        csmDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings = bindings.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create CSM descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool CsmRenderer::createCsmDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer,        .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
            {.type = vk::DescriptorType::eStorageBuffer,         .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            {.type = vk::DescriptorType::eCombinedImageSampler,  .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        };

        csmDescriptorPool = vk::raii::DescriptorPool(device,
            vk::DescriptorPoolCreateInfo{
                .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                .maxSets = MAX_FRAMES_IN_FLIGHT,
                .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                .pPoolSizes = poolSizes.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create CSM descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void CsmRenderer::createCsmDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *csmDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *csmDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };
    csmDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo sceneBuf{
            .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instBuf{
            .buffer = *instanceBufferResources.Buffers[i], .offset = 0,
            .range = sizeof(InstanceData) * instanceCount };
        vk::DescriptorBufferInfo csmBuf{
            .buffer = *csmUboResources.Buffers[i], .offset = 0, .range = sizeof(CsmUBO) };
        vk::DescriptorBufferInfo shadowParamsBuf{
            .buffer = *shadowParamsUboResources.Buffers[i], .offset = 0,
            .range = sizeof(ShadowParamsUBO) };
        vk::DescriptorImageInfo shadowMapImg{
            .sampler = *csmSampler,
            .imageView = *csmArrayView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writes = {
            {.dstSet = *csmDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBuf },
            {.dstSet = *csmDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instBuf },
            {.dstSet = *csmDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &csmBuf },
            {.dstSet = *csmDescriptorSets[i], .dstBinding = 3, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowMapImg },
            {.dstSet = *csmDescriptorSets[i], .dstBinding = 4, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &shadowParamsBuf },
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

// =============================================================================
// CSM Texture Array Resources
// =============================================================================

bool CsmRenderer::createCsmMapResources()
{
    try
    {
        vk::Format depthFormat = findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD16Unorm },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
        );

        // Create 2D texture array with CASCADE_COUNT layers
        // Note: VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT is only for 3D images
        createImage(
            SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1,
            CASCADE_COUNT,
            {},  // no flags needed for 2D array
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            csmTextureArray
        );

        // Per-layer image views (for depth-only rendering to each cascade)
        csmLayerViews.clear();
        csmLayerViews.reserve(CASCADE_COUNT);
        for (uint32_t layer = 0; layer < CASCADE_COUNT; ++layer)
        {
            vk::ImageViewCreateInfo viewInfo{
                .image = *csmTextureArray.textureImage,
                .viewType = vk::ImageViewType::e2DArray,
                .format = depthFormat,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0, .levelCount = 1,
                    .baseArrayLayer = layer, .layerCount = 1
                }
            };
            csmLayerViews.emplace_back(device, viewInfo);
        }

        // Full-array image view (for shader sampling in lit pass)
        {
            vk::ImageViewCreateInfo viewInfo{
                .image = *csmTextureArray.textureImage,
                .viewType = vk::ImageViewType::e2DArray,
                .format = depthFormat,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0, .levelCount = 1,
                    .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
                }
            };
            csmArrayView = vk::raii::ImageView(device, viewInfo);
        }

        // Sampler for shadow map array
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
            .minLod = 0.0f, .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite,
            .unnormalizedCoordinates = vk::False
        };
        csmSampler = vk::raii::Sampler(device, samplerInfo);

        csmArrayLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create CSM map resources: " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// Pipelines
// =============================================================================

bool CsmRenderer::createCsmPipelines()
{
    try
    {
        // --- Pipeline Layouts ---

        // Depth pipeline layout: 1 descriptor set + push constant for cascadeIndex
        vk::PushConstantRange pushRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(uint32_t)
        };
        csmDepthPipelineLayout = vk::raii::PipelineLayout(device,
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount = 1,
                .pSetLayouts = &*csmDescriptorSetLayout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pushRange
            });

        // Lit pipeline layout: 1 descriptor set, no push constants
        csmLitPipelineLayout = vk::raii::PipelineLayout(device,
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount = 1,
                .pSetLayouts = &*csmDescriptorSetLayout
            });

        // --- Common State ---
        const auto bindingDescription = Vertex::getBindingDescription();
        const auto posOnlyAttrs = Vertex::getPositionOnlyAttributeDescriptions();
        const auto posNormalAttrs = Vertex::getPositionNormalAttributeDescriptions();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLessOrEqual,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        const vk::Format depthFormat = findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD16Unorm },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
        );

        // --- Depth Pipeline (shadow map) ---
        {
            vk::raii::ShaderModule shader = createShaderModule(
                readFile(std::string(VK_SHADERS_DIR) + "csm_depth.spv"));
            vk::PipelineShaderStageCreateInfo vertStage{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *shader, .pName = "vertMain" };
            vk::PipelineShaderStageCreateInfo fragStage{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *shader, .pName = "fragMain" };
            vk::PipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

            vk::PipelineVertexInputStateCreateInfo vertexInput{
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &bindingDescription,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(posOnlyAttrs.size()),
                .pVertexAttributeDescriptions = posOnlyAttrs.data()
            };

            vk::PipelineRasterizationStateCreateInfo rasterizer{
                .depthClampEnable = vk::False,
                .rasterizerDiscardEnable = vk::False,
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eBack,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .depthBiasEnable = vk::True,
                .depthBiasConstantFactor = 1.25f,
                .depthBiasClamp = 0.0f,
                .depthBiasSlopeFactor = 1.75f,
                .lineWidth = 1.0f
            };

            vk::PipelineColorBlendStateCreateInfo colorBlend{
                .logicOpEnable = vk::False,
                .attachmentCount = 0,
                .pAttachments = nullptr
            };

            vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
                {
                    .stageCount = 2, .pStages = stages,
                    .pVertexInputState = &vertexInput,
                    .pInputAssemblyState = &inputAssembly,
                    .pViewportState = &viewportState,
                    .pRasterizationState = &rasterizer,
                    .pMultisampleState = &multisampling,
                    .pDepthStencilState = &depthStencil,
                    .pColorBlendState = &colorBlend,
                    .pDynamicState = &dynamicState,
                    .layout = *csmDepthPipelineLayout,
                    .renderPass = nullptr
                },
                {
                    .colorAttachmentCount = 0,
                    .pColorAttachmentFormats = nullptr,
                    .depthAttachmentFormat = depthFormat
                }
            };

            csmDepthPipeline = vk::raii::Pipeline(device, nullptr,
                chain.get<vk::GraphicsPipelineCreateInfo>());
        }

        // --- Lit Pipeline (onscreen) ---
        {
            vk::raii::ShaderModule shader = createShaderModule(
                readFile(std::string(VK_SHADERS_DIR) + "csm_lit.spv"));
            vk::PipelineShaderStageCreateInfo vertStage{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *shader, .pName = "vertMain" };
            vk::PipelineShaderStageCreateInfo fragStage{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *shader, .pName = "fragMain" };
            vk::PipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

            vk::PipelineVertexInputStateCreateInfo vertexInput{
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &bindingDescription,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(posNormalAttrs.size()),
                .pVertexAttributeDescriptions = posNormalAttrs.data()
            };

            vk::PipelineRasterizationStateCreateInfo rasterizer{
                .depthClampEnable = vk::False,
                .rasterizerDiscardEnable = vk::False,
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .depthBiasEnable = vk::False,
                .lineWidth = 1.0f
            };

            vk::PipelineColorBlendAttachmentState colorBlendAttachment{
                .blendEnable = vk::False,
                .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                                | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
            };
            vk::PipelineColorBlendStateCreateInfo colorBlend{
                .logicOpEnable = vk::False,
                .attachmentCount = 1,
                .pAttachments = &colorBlendAttachment
            };

            const vk::Format sceneDepthFormat = findDepthFormat();
            vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
                {
                    .stageCount = 2, .pStages = stages,
                    .pVertexInputState = &vertexInput,
                    .pInputAssemblyState = &inputAssembly,
                    .pViewportState = &viewportState,
                    .pRasterizationState = &rasterizer,
                    .pMultisampleState = &multisampling,
                    .pDepthStencilState = &depthStencil,
                    .pColorBlendState = &colorBlend,
                    .pDynamicState = &dynamicState,
                    .layout = *csmLitPipelineLayout,
                    .renderPass = nullptr
                },
                {
                    .colorAttachmentCount = 1,
                    .pColorAttachmentFormats = &swapChainImageFormat,
                    .depthAttachmentFormat = sceneDepthFormat
                }
            };

            csmLitPipeline = vk::raii::Pipeline(device, nullptr,
                chain.get<vk::GraphicsPipelineCreateInfo>());
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create CSM pipelines: " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// Cascade Split Calculation
// Practical Split Scheme:
//   split_i = λ * near * (far/near)^(i/N) + (1-λ) * (near + (far-near) * i/N)
// where λ = splitLambda, N = CASCADE_COUNT, i = 1..N
// =============================================================================

void CsmRenderer::calculateCascadeSplits()
{
    const float nearPlane = 0.1f;
    const float farPlane = 100.0f;

    cascadeSplitDepths[0] = nearPlane;
    cascadeSplitDepths[CASCADE_COUNT] = farPlane;

    for (uint32_t i = 1; i < CASCADE_COUNT; ++i)
    {
        float ratio = static_cast<float>(i) / static_cast<float>(CASCADE_COUNT);
        float logSplit = nearPlane * std::pow(farPlane / nearPlane, ratio);
        float linSplit = nearPlane + (farPlane - nearPlane) * ratio;
        cascadeSplitDepths[i] = splitLambda * logSplit + (1.0f - splitLambda) * linSplit;
    }
}

// =============================================================================
// Compute Light View-Projection for a Cascade
// Extracts the 8 corners of the sub-frustum, transforms to light space,
// computes the AABB, and builds an orthographic projection.
// =============================================================================

void CsmRenderer::computeCascadeViewProj(uint32_t cascadeIndex, const glm::vec3& lightDir)
{
    const float nearZ = cascadeSplitDepths[cascadeIndex];
    const float farZ = cascadeSplitDepths[cascadeIndex + 1];

    const float aspect = static_cast<float>(swapChainExtent.width)
        / static_cast<float>(swapChainExtent.height);
    const float fov = glm::radians(camera.Zoom);

    // Compute sub-frustum corners in view space
    const float tanHalfFov = std::tan(fov * 0.5f);
    const float nearH = nearZ * tanHalfFov;
    const float nearW = nearH * aspect;
    const float farH = farZ * tanHalfFov;
    const float farW = farH * aspect;

    // 8 corners of the sub-frustum in view space
    std::array<glm::vec4, 8> cornersVS = {
        glm::vec4(-nearW,  nearH, nearZ, 1.0f),
        glm::vec4(nearW,  nearH, nearZ, 1.0f),
        glm::vec4(nearW, -nearH, nearZ, 1.0f),
        glm::vec4(-nearW, -nearH, nearZ, 1.0f),
        glm::vec4(-farW,   farH, farZ,  1.0f),
        glm::vec4(farW,   farH, farZ,  1.0f),
        glm::vec4(farW,  -farH, farZ,  1.0f),
        glm::vec4(-farW,  -farH, farZ,  1.0f),
    };

    // Transform to world space
    const glm::mat4 invView = glm::inverse(camera.GetViewMatrix());
    std::array<glm::vec3, 8> cornersWS;
    for (int i = 0; i < 8; ++i)
    {
        glm::vec4 ws = invView * cornersVS[i];
        cornersWS[i] = glm::vec3(ws) / ws.w;
    }

    // Build light view matrix
    const glm::vec3 lightPos = glm::vec3(0.0f) - lightDir * 50.0f;
    const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), up);

    // Transform corners to light view space to compute AABB
    glm::vec3 minAABB(std::numeric_limits<float>::max());
    glm::vec3 maxAABB(std::numeric_limits<float>::lowest());
    for (const auto& corner : cornersWS)
    {
        glm::vec4 ls = lightView * glm::vec4(corner, 1.0f);
        minAABB = glm::min(minAABB, glm::vec3(ls));
        maxAABB = glm::max(maxAABB, glm::vec3(ls));
    }

    // Expand Z range to capture potential occluders behind the frustum
    const float zPadding = 30.0f;
    minAABB.z -= zPadding;
    maxAABB.z += zPadding;

    // Build orthographic projection from AABB
    glm::mat4 lightProj = glm::ortho(minAABB.x, maxAABB.x,
        minAABB.y, maxAABB.y,
        -maxAABB.z, -minAABB.z);
    lightProj[1][1] *= -1;  // Vulkan NDC flip

    cascadeViewProj[cascadeIndex] = lightProj * lightView;
}

// =============================================================================
// Update Buffers
// =============================================================================

void CsmRenderer::updateCsmBuffers(uint32_t frameIndex)
{
    // --- Scene UBO ---
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1;
    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    // --- Light direction ---
    float deltaTime = platform ? platform->frameTimer : 0.0f;
    lightAngle += deltaTime * 0.35f;
    const glm::vec3 lightDir = glm::normalize(
        glm::vec3(std::cos(lightAngle) * 0.6f, -1.0f, std::sin(lightAngle) * 0.6f));

    // --- Cascade splits & VPs ---
    calculateCascadeSplits();
    for (uint32_t i = 0; i < CASCADE_COUNT; ++i)
    {
        computeCascadeViewProj(i, lightDir);
    }

    // --- CSM UBO ---
    CsmUBO csmUbo{};
    for (uint32_t i = 0; i < CASCADE_COUNT; ++i)
    {
        csmUbo.cascadeViewProj[i] = cascadeViewProj[i];
    }
    csmUbo.cascadeSplitDepths = glm::vec4(
        cascadeSplitDepths[1], cascadeSplitDepths[2],
        cascadeSplitDepths[3], cascadeSplitDepths[4]);
    std::memcpy(csmUboResources.BuffersMapped[frameIndex], &csmUbo, sizeof(csmUbo));

    // --- Shadow Params UBO ---
    ShadowParamsUBO shadowParams{
        .shadowFilterMode = shadowFilterMode,
        .pcfRadiusTexels = pcfRadiusTexels,
        .pcssLightSizeTexels = pcssLightSizeTexels,
        .shadowBiasMin = 0.0006f,
        .invShadowMapSize = glm::vec2(1.0f / float(SHADOW_MAP_SIZE), 1.0f / float(SHADOW_MAP_SIZE)),
        .padding0 = glm::vec2(0.0f)
    };
    std::memcpy(shadowParamsUboResources.BuffersMapped[frameIndex], &shadowParams, sizeof(shadowParams));

    // --- Instance Data ---
    auto* instances = static_cast<InstanceData*>(instanceBufferResources.BuffersMapped[frameIndex]);

    // Floor (instance 0)
    {
        glm::mat4 model(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, -2.0f, 0.0f));
        model = glm::scale(model, glm::vec3(8.0f, 0.25f, 8.0f));
        instances[0] = InstanceData{ model, glm::vec4(0.55f, 0.55f, 0.60f, 1.0f) };
    }

    // Cubes around origin (instances 1-9)
    const glm::vec3 cubeColor0(0.92f, 0.30f, 0.25f);
    const glm::vec3 cubeColor1(0.25f, 0.65f, 0.92f);
    for (int i = 1; i < static_cast<int>(instanceCount); ++i)
    {
        const float a = static_cast<float>(i - 1) / static_cast<float>(instanceCount - 1) * 6.2831853f;
        glm::mat4 model(1.0f);
        model = glm::translate(model, glm::vec3(std::cos(a) * 2.8f, -1.0f, std::sin(a) * 2.8f));
        model = glm::rotate(model, a + lightAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(0.65f));
        const glm::vec3 c = (i & 1) ? cubeColor0 : cubeColor1;
        instances[i] = InstanceData{ model, glm::vec4(c, 1.0f) };
    }
}

// =============================================================================
// Render Loop
// =============================================================================

void CsmRenderer::render()
{
    auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
        throw std::runtime_error("waitForFences failed");

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX,
        *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR) { recreateSwapChain(); return; }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("acquireNextImage failed");

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    updateUIFrame();
    updateCsmBuffers(currentFrame);
    recordCommandBuffer(imageIndex);

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

    try { result = presentQueue.presentKHR(presentInfo); }
    catch (const vk::OutOfDateKHRError&) { framebufferResized = false; recreateSwapChain(); return; }

    if (result == vk::Result::eSuboptimalKHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// =============================================================================
// Command Buffer Recording
// =============================================================================

void CsmRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& cmd = commandBuffers[currentFrame];
    cmd.begin({});

    // =====================================================================
    // Pass 1: CSM Depth Pass (one per cascade)
    // =====================================================================

    // Transition entire CSM array (all layers) for depth writing.
    // Base class transition_image_layout only transitions layer 0, so we use
    // a custom barrier with layerCount = CASCADE_COUNT.
    {
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests
                          | vk::PipelineStageFlagBits2::eLateFragmentTests,
            .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout = csmArrayLayout,
            .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *csmTextureArray.textureImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
            }
        };
        vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
        cmd.pipelineBarrier2(depInfo);
    }
    csmArrayLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    // Render to each cascade layer
    for (uint32_t cascadeIdx = 0; cascadeIdx < CASCADE_COUNT; ++cascadeIdx)
    {
        vk::RenderingAttachmentInfo depthAttachment{
            .imageView = *csmLayerViews[cascadeIdx],
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
        };

        vk::RenderingInfo renderInfo{
            .renderArea = {.offset = {0, 0}, .extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} },
            .layerCount = 1,
            .colorAttachmentCount = 0,
            .pDepthAttachment = &depthAttachment
        };

        cmd.beginRendering(renderInfo);
        cmd.setViewport(0, vk::Viewport(0.0f, 0.0f,
            static_cast<float>(SHADOW_MAP_SIZE), static_cast<float>(SHADOW_MAP_SIZE), 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({ 0, 0 }, { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE }));

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *csmDepthPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *csmDepthPipelineLayout, 0, *csmDescriptorSets[currentFrame], nullptr);

        // Push cascade index
        cmd.pushConstants<uint32_t>(*csmDepthPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, cascadeIdx);

        // Draw floor (instance 0)
        cmd.bindVertexBuffers(0, *planeMesh.vertexBuffer, { 0 });
        cmd.bindIndexBuffer(*planeMesh.indexBuffer, 0, vk::IndexType::eUint16);
        cmd.drawIndexed(static_cast<uint32_t>(planeMesh.indices.size()), 1, 0, 0, 0);

        // Draw cubes (instances 1..instanceCount-1)
        cmd.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
        cmd.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexType::eUint16);
        cmd.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()),
            instanceCount - 1, 0, 0, 1);

        cmd.endRendering();
    }

    // Transition CSM array (all layers) for shader reading
    {
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *csmTextureArray.textureImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
            }
        };
        vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
        cmd.pipelineBarrier2(depInfo);
    }
    csmArrayLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    // =====================================================================
    // Pass 2: Lit Pass
    // =====================================================================

    // Transition swapchain for color output
    transition_image_layout(cmd,
        swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    // Transition scene depth
    transition_image_layout(cmd,
        *depthData.textureImage,
        depthImageLayout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    vk::RenderingAttachmentInfo colorAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.07f, 0.07f, 0.09f, 1.0f)
    };

    vk::RenderingAttachmentInfo depthAttachment{
        .imageView = depthData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
    };

    vk::RenderingInfo renderInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment
    };

    cmd.beginRendering(renderInfo);
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f,
        static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({ 0, 0 }, swapChainExtent));

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *csmLitPipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *csmLitPipelineLayout, 0, *csmDescriptorSets[currentFrame], nullptr);

    // Draw floor
    cmd.bindVertexBuffers(0, *planeMesh.vertexBuffer, { 0 });
    cmd.bindIndexBuffer(*planeMesh.indexBuffer, 0, vk::IndexType::eUint16);
    cmd.drawIndexed(static_cast<uint32_t>(planeMesh.indices.size()), 1, 0, 0, 0);

    // Draw cubes
    cmd.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
    cmd.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexType::eUint16);
    cmd.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), instanceCount - 1, 0, 0, 1);

    cmd.endRendering();

    // =====================================================================
    // Pass 3: UI
    // =====================================================================

    vk::RenderingAttachmentInfo uiAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingInfo uiRenderInfo{
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachment
    };

    cmd.beginRendering(uiRenderInfo);
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f,
        static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({ 0, 0 }, swapChainExtent));
    recordUICmdBuffer(cmd, currentFrame);
    cmd.endRendering();

    // Transition for presentation
    transition_image_layout(cmd,
        swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;

    cmd.end();
}

// =============================================================================
// UI Panel
// =============================================================================

void CsmRenderer::updateUIPanel()
{
    ImGui::Begin("Cascaded Shadow Maps");

    if (ImGui::CollapsingHeader("Cascade Splits", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Split Lambda", &splitLambda, 0.0f, 1.0f,
            "%.2f (0=linear, 1=log)");
        ImGui::Checkbox("Visualize Cascades", &visualizeCascades);

        ImGui::Separator();
        ImGui::Text("Split Depths:");
        for (uint32_t i = 1; i <= CASCADE_COUNT; ++i)
        {
            ImGui::Text("  Cascade %u: %.2f", i, cascadeSplitDepths[i]);
        }
    }

    if (ImGui::CollapsingHeader("Shadow Filter"))
    {
        const char* modes[] = { "Hard", "PCF", "PCSS" };
        ImGui::Combo("Filter Mode", &shadowFilterMode, modes, 3);

        if (shadowFilterMode == 1)
        {
            ImGui::SliderFloat("PCF Radius (texels)", &pcfRadiusTexels, 0.0f, 6.0f);
        }
        if (shadowFilterMode == 2)
        {
            ImGui::SliderFloat("Light Size (texels)", &pcssLightSizeTexels, 1.0f, 80.0f);
        }
    }

    ImGui::Separator();
    ImGui::SliderFloat("Light Intensity", &dirLightIntensity, 0.0f, 5.0f);

    ImGui::End();
}

// =============================================================================
// Cleanup
// =============================================================================

void CsmRenderer::cleanup()
{
    device.waitIdle();
    shutdownVulkanUI();
}

void CsmRenderer::recreateSwapChain()
{
    VulkanBase::recreateSwapChain();
}

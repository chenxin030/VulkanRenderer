#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <array>

#if RENDERING_LEVEL == 6

struct ShadowInstanceData
{
    glm::mat4 model;
    glm::vec4 color;
};

struct TAAUParams
{
    float blendFactor;
    float reactiveClamp;
    float antiFlicker;
    float velocityScale;
    float historyClampGamma;
    float historyRejectThreshold;
};

struct TAAUParamsUBO
{
    float blendFactor;
    float reactiveClamp;
    float antiFlicker;
    float velocityScale;
    float historyClampGamma;
    float historyRejectThreshold;
    float pad0;
    float pad1;
};

struct TAAUSceneState
{
    glm::mat4 prevViewProj;
    glm::vec3 fastMoveVelocity;
    float time = 0.0f;
    bool freezeHistory = false;
};

static TAAUSceneState gTaauSceneState{};
static TAAUParams gTaauParams{ 0.90f, 0.55f, 0.88f, 1.00f, 1.15f, 0.22f };

static float halton(uint32_t index, uint32_t base)
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

bool Renderer::createTAAUResources()
{
    try
    {
        const uint32_t scaledWidth = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * taauRenderScale));
        const uint32_t scaledHeight = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * taauRenderScale));

        taauInputColorData.mipLevels = 1;
        createImage(
            scaledWidth,
            scaledHeight,
            1,
            swapChainImageFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            taauInputColorData
        );
        taauInputColorData.textureImageView = createImageView(taauInputColorData.textureImage, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);

        taauVelocityData.mipLevels = 1;
        createImage(
            scaledWidth,
            scaledHeight,
            1,
            vk::Format::eR16G16Sfloat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            taauVelocityData
        );
        taauVelocityData.textureImageView = createImageView(taauVelocityData.textureImage, vk::Format::eR16G16Sfloat, vk::ImageAspectFlagBits::eColor, 1);

        taauDepthData.mipLevels = 1;
        vk::Format taauDepthFormat = findDepthFormat();
        createImage(
            scaledWidth,
            scaledHeight,
            1,
            taauDepthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            taauDepthData
        );
        taauDepthData.textureImageView = createImageView(taauDepthData.textureImage, taauDepthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        for (int i = 0; i < 2; ++i)
        {
            taauHistoryColorData[i].mipLevels = 1;
            createImage(
                swapChainExtent.width,
                swapChainExtent.height,
                1,
                swapChainImageFormat,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                taauHistoryColorData[i]
            );
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

bool Renderer::createTAAUDescriptorSetLayout()
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

bool Renderer::createTAAUDescriptorPool()
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

void Renderer::createTAAUDescriptorSets()
{
    if (taauDescriptorSets.size() != MAX_FRAMES_IN_FLIGHT)
    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *taauDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *taauDescriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        taauDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        updateTAAUDescriptorSet(i, taauHistoryReadIndex);
    }
}

void Renderer::updateTAAUDescriptorSet(uint32_t frameIndex, uint32_t historyReadIndex)
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

bool Renderer::createTAAUPipeline()
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

void Renderer::updateTAAUBuffers(uint32_t currentImage)
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
        .blendFactor = gTaauParams.blendFactor,
        .reactiveClamp = gTaauParams.reactiveClamp,
        .antiFlicker = gTaauParams.antiFlicker,
        .velocityScale = gTaauParams.velocityScale,
        .historyClampGamma = gTaauParams.historyClampGamma,
        .historyRejectThreshold = gTaauParams.historyRejectThreshold,
        .pad0 = 0.0f,
        .pad1 = 0.0f,
    };

    memcpy(taauParamsUboResources.BuffersMapped[currentImage], &params, sizeof(params));
}

void Renderer::recordTAAU(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    if (!taauEnabled)
    {
        transition_image_layout(
            taauInputColorData.textureImage,
            taauInputLayout,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageAspectFlagBits::eColor
        );
        taauInputLayout = vk::ImageLayout::eTransferSrcOptimal;

        transition_image_layout(
            swapChainImages[imageIndex],
            swapChainImageLayouts[imageIndex],
            vk::ImageLayout::eTransferDstOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageAspectFlagBits::eColor
        );
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
        commandBuffer.blitImage(
            taauInputColorData.textureImage, vk::ImageLayout::eTransferSrcOptimal,
            swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
            blitRegion,
            vk::Filter::eLinear
        );

        transition_image_layout(
            swapChainImages[imageIndex],
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor
        );
        swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;
        return;
    }

    const uint32_t historyRead = taauHistoryReadIndex;
    const uint32_t historyWrite = (historyRead + 1u) % 2u;

    transition_image_layout(
        taauInputColorData.textureImage,
        taauInputLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
    );
    taauInputLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(
        taauVelocityData.textureImage,
        taauVelocityLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
    );
    taauVelocityLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(
        taauDepthData.textureImage,
        taauDepthLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth
    );
    taauDepthLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    // TAAU 冷启动 | 历史重置后的恢复
    if (!taauHistoryValid)
    {
        transition_image_layout(
            taauInputColorData.textureImage,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageAspectFlagBits::eColor
        );

        transition_image_layout(
            taauHistoryColorData[historyRead].textureImage,
            taauHistoryLayouts[historyRead],
            vk::ImageLayout::eTransferDstOptimal,
            {},
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::ImageAspectFlagBits::eColor
        );

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
        commandBuffer.blitImage(
            taauInputColorData.textureImage, vk::ImageLayout::eTransferSrcOptimal,
            taauHistoryColorData[historyRead].textureImage, vk::ImageLayout::eTransferDstOptimal,
            initBlit,
            vk::Filter::eLinear
        );

        transition_image_layout(
            taauHistoryColorData[historyRead].textureImage,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor
        );
        taauHistoryLayouts[historyRead] = vk::ImageLayout::eShaderReadOnlyOptimal;

        transition_image_layout(
            taauInputColorData.textureImage,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eTransferRead,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor
        );
    }

    if (taauHistoryLayouts[historyRead] != vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        transition_image_layout(
            taauHistoryColorData[historyRead].textureImage,
            taauHistoryLayouts[historyRead],
            vk::ImageLayout::eShaderReadOnlyOptimal,
            {},
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor
        );
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

    transition_image_layout(
        swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor
    );
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eTransferSrcOptimal;

    transition_image_layout(
        taauHistoryColorData[historyWrite].textureImage,
        taauHistoryLayouts[historyWrite],
        vk::ImageLayout::eTransferDstOptimal,
        {},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor
    );

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
    commandBuffer.blitImage(
        swapChainImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal,
        taauHistoryColorData[historyWrite].textureImage, vk::ImageLayout::eTransferDstOptimal,
        historyBlit,
        vk::Filter::eLinear
    );

    transition_image_layout(
        taauHistoryColorData[historyWrite].textureImage,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
    );
    taauHistoryLayouts[historyWrite] = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(
        swapChainImages[imageIndex],
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferRead,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
    );
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    taauHistoryReadIndex = historyWrite;
    taauHistoryValid = true;
}

void Renderer::updateTAAUScene(float deltaTime)
{
    gTaauSceneState.time += deltaTime;

    if (scene != nullptr)
    {
        if (!gTaauSceneState.freezeHistory && scene->taauMovingProbe != 0)
        {
            float fastPhase = gTaauSceneState.time * 2.4f;
            float zigzag = std::sin(fastPhase);
            float offset = std::sin(fastPhase * 1.5f) * 0.6f;
            Transform* moving = scene->world.getTransform(scene->taauMovingProbe);
            if (moving)
            {
                moving->position.x = 2.5f + zigzag * 1.2f;
                moving->position.z = -1.6f + offset;
            }
        }

        if (scene->taauEdgeProbe != 0)
        {
            Transform* edgeProbe = scene->world.getTransform(scene->taauEdgeProbe);
            if (edgeProbe)
            {
                edgeProbe->position.x = 3.7f + std::sin(gTaauSceneState.time * 0.6f) * 0.25f;
            }
        }
    }

    gTaauSceneState.fastMoveVelocity = glm::vec3(
        std::cos(gTaauSceneState.time * 2.4f),
        0.0f,
        std::sin(gTaauSceneState.time * 1.5f)
    ) * gTaauParams.velocityScale;
}

void Renderer::updateTAAUHistory(const glm::mat4& currentViewProj)
{
    gTaauSceneState.prevViewProj = taauPrevViewProj;
    taauPrevViewProj = currentViewProj;
}

void Renderer::updateTAAUUI()
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
    ImGui::SliderFloat("BlendFactor", &gTaauParams.blendFactor, 0.2f, 0.98f);
    ImGui::SliderFloat("ReactiveClamp", &gTaauParams.reactiveClamp, 0.2f, 1.2f);
    ImGui::SliderFloat("AntiFlicker", &gTaauParams.antiFlicker, 0.0f, 1.0f);
    ImGui::SliderFloat("VelocityScale", &gTaauParams.velocityScale, 0.2f, 2.5f);
    ImGui::SliderFloat("ClampGamma", &gTaauParams.historyClampGamma, 0.5f, 2.5f);
    ImGui::SliderFloat("RejectThreshold", &gTaauParams.historyRejectThreshold, 0.01f, 0.5f);
    if (ImGui::SliderFloat("RenderScale", &taauRenderScale, 0.5f, 1.0f, "%.2f"))
    {
        taauHistoryValid = false;
        waitIdle();
        createTAAUResources();
        createTAAUDescriptorSets();
    }
    ImGui::Checkbox("FreezeHistory", &gTaauSceneState.freezeHistory);
    if (ImGui::Button("Reset History"))
    {
        taauHistoryValid = false;
    }
    ImGui::End();
}

#endif
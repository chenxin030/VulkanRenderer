#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#if RENDERING_LEVEL == 7

struct SSRSceneUBO
{
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 invProjection;
    glm::vec4 cameraPosNear;
    glm::vec4 cameraFarPadding;
};

struct SSRParams
{
    float maxRayDistance;
    float thickness;
    float stride;
    float intensity;

    glm::vec2 invResolution;
    int debugMode; // 0=off,1=hit mask,2=steps,3=depth
    int maxSteps;
    float padding0;
};

bool Renderer::createSSRResources()
{
    try
    {
        ssrColorData.mipLevels = 1;
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            swapChainImageFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            ssrColorData
        );
        ssrColorData.textureImageView = createImageView(ssrColorData.textureImage, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);

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
        ssrColorSampler = vk::raii::Sampler(device, samplerInfo);

        vk::SamplerCreateInfo depthSamplerInfo = samplerInfo;
        depthSamplerInfo.magFilter = vk::Filter::eNearest;
        depthSamplerInfo.minFilter = vk::Filter::eNearest;
        ssrDepthSampler = vk::raii::Sampler(device, depthSamplerInfo);

        ssrColorLayout = vk::ImageLayout::eUndefined;

        createUniformBuffers(ssrSceneUboResources, sizeof(SSRSceneUBO));
        createUniformBuffers(ssrParamsUboResources, sizeof(SSRParams));
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create SSR resources: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createSSRDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        ssrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create SSR descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createSSRDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u}
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        ssrDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create SSR descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createSSRDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *ssrDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *ssrDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    ssrDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *ssrSceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SSRSceneUBO) };
        vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *ssrParamsUboResources.Buffers[i], .offset = 0, .range = sizeof(SSRParams) };

        vk::DescriptorImageInfo depthInfo{
            .sampler = *ssrDepthSampler,
            .imageView = *depthData.textureImageView,
            .imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal
        };

        vk::DescriptorImageInfo colorInfo{
            .sampler = *ssrColorSampler,
            .imageView = *ssrColorData.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        std::vector<vk::WriteDescriptorSet> writes = {
            {.dstSet = *ssrDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo},
            {.dstSet = *ssrDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
            {.dstSet = *ssrDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &colorInfo},
            {.dstSet = *ssrDescriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo},
        };

        device.updateDescriptorSets(writes, nullptr);
    }
}

bool Renderer::createSSRPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "ssr.spv"));
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
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
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

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*ssrDescriptorSetLayout, .pushConstantRangeCount = 0 };
        ssrPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
                .layout = ssrPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        ssrPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create SSR pipeline: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::updateSSRBuffers(uint32_t currentImage)
{
    constexpr float nearPlane = 0.1f;
    constexpr float farPlane = 100.0f;

    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom),
        static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
        nearPlane, farPlane);
    projection[1][1] *= -1;

    SSRSceneUBO sceneUbo{
        .projection = projection,
        .view = camera.GetViewMatrix(),
        .invProjection = glm::inverse(projection),
        .cameraPosNear = glm::vec4(camera.Position, nearPlane),
        .cameraFarPadding = glm::vec4(farPlane, 0.0f, 0.0f, 0.0f)
    };
    memcpy(ssrSceneUboResources.BuffersMapped[currentImage], &sceneUbo, sizeof(sceneUbo));

    SSRParams params{
        .maxRayDistance = ssrMaxRayDistance,
        .thickness = ssrThickness,
        .stride = ssrStride,
        .intensity = ssrIntensity,
        .invResolution = glm::vec2(1.0f / float(swapChainExtent.width), 1.0f / float(swapChainExtent.height)),
        .debugMode = 0,
        .maxSteps = ssrMaxSteps,
        .padding0 = 0.0f
    };

    memcpy(ssrParamsUboResources.BuffersMapped[currentImage], &params, sizeof(params));
}

void Renderer::recordSSR(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    if (!ssrEnabled)
    {
        return;
    }

    transition_image_layout(
        swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex],
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor
    );
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eTransferSrcOptimal;

    transition_image_layout(
        ssrColorData.textureImage,
        ssrColorLayout,
        vk::ImageLayout::eTransferDstOptimal,
        {},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor
    );
    ssrColorLayout = vk::ImageLayout::eTransferDstOptimal;

    vk::ImageBlit blitRegion{
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
        ssrColorData.textureImage, vk::ImageLayout::eTransferDstOptimal,
        blitRegion,
        vk::Filter::eLinear
    );

    transition_image_layout(
        ssrColorData.textureImage,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
    );
    ssrColorLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

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

    transition_image_layout(
        depthData.textureImage,
        depthImageLayout,
        vk::ImageLayout::eDepthReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth
    );
    depthImageLayout = vk::ImageLayout::eDepthReadOnlyOptimal;

    vk::RenderingAttachmentInfo compositeAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingInfo compositeRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &compositeAttachmentInfo
    };

    commandBuffer.beginRendering(compositeRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *ssrPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *ssrPipelineLayout, 0, *ssrDescriptorSets[currentFrame], nullptr);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    transition_image_layout(
        depthData.textureImage,
        vk::ImageLayout::eDepthReadOnlyOptimal,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth
    );
    depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
}

void Renderer::updateSSRUI()
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGui::Begin("SSR", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Enable", &ssrEnabled);
    ImGui::SliderFloat("Intensity", &ssrIntensity, 0.0f, 1.5f);
    ImGui::SliderFloat("MaxDistance", &ssrMaxRayDistance, 1.0f, 50.0f);
    ImGui::SliderFloat("Thickness", &ssrThickness, 0.02f, 0.6f);
    ImGui::SliderFloat("Stride", &ssrStride, 0.05f, 1.0f);
    ImGui::SliderInt("MaxSteps", &ssrMaxSteps, 8, 128);
    ImGui::End();
}

#endif

#include "VulkanBase_UI.h"
#include "VulkanBase.h"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <cstring>

void vkrReallocateUIMemory(VulkanBase* self, UiFrameBuffers& fb, size_t vtxBytes, size_t idxBytes)
{
    if (fb.vertexBuffer == vk::raii::Buffer(nullptr) || fb.vertexBufferSize < vtxBytes)
    {
        if (fb.vertexMapped != nullptr)
        {
            fb.vertexBufferMemory.unmapMemory();
            fb.vertexMapped = nullptr;
        }
        fb.vertexBuffer = vk::raii::Buffer(nullptr);
        fb.vertexBufferMemory = vk::raii::DeviceMemory(nullptr);
        self->createBuffer(vtxBytes, vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            fb.vertexBuffer, fb.vertexBufferMemory);
        fb.vertexMapped = fb.vertexBufferMemory.mapMemory(0, vtxBytes);
        fb.vertexBufferSize = vtxBytes;
    }

    if (fb.indexBuffer == vk::raii::Buffer(nullptr) || fb.indexBufferSize < idxBytes)
    {
        if (fb.indexMapped != nullptr)
        {
            fb.indexBufferMemory.unmapMemory();
            fb.indexMapped = nullptr;
        }
        fb.indexBuffer = vk::raii::Buffer(nullptr);
        fb.indexBufferMemory = vk::raii::DeviceMemory(nullptr);
        self->createBuffer(idxBytes, vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            fb.indexBuffer, fb.indexBufferMemory);
        fb.indexMapped = fb.indexBufferMemory.mapMemory(0, idxBytes);
        fb.indexBufferSize = idxBytes;
    }
}

bool initVulkanUI(VulkanBase* self)
{
    if (!self->uiEnabled)
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
    self->createBuffer(uploadSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);

    void* mapped = stagingBufferMemory.mapMemory(0, uploadSize);
    memcpy(mapped, pixels, static_cast<size_t>(uploadSize));
    stagingBufferMemory.unmapMemory();

    self->uiFontTexture.mipLevels = 1;
    self->createImage(static_cast<uint32_t>(fontWidth), static_cast<uint32_t>(fontHeight), 1,
        vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, self->uiFontTexture);

    self->transitionImageLayout(self->uiFontTexture.textureImage, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, 1);
    self->copyBufferToImage(stagingBuffer, self->uiFontTexture.textureImage,
        static_cast<uint32_t>(fontWidth), static_cast<uint32_t>(fontHeight));
    self->transitionImageLayout(self->uiFontTexture.textureImage, vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    self->uiFontTexture.textureImageView = self->createImageView(self->uiFontTexture.textureImage,
        vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor, 1);

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
    self->uiFontTexture.textureSampler = vk::raii::Sampler(self->device, samplerInfo);

    std::vector<vk::DescriptorSetLayoutBinding> uiBindings = {
        {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment}
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(uiBindings.size()),
        .pBindings = uiBindings.data()
    };
    self->uiDescriptorSetLayout = vk::raii::DescriptorSetLayout(self->device, layoutInfo);

    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1}
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    self->uiDescriptorPool = vk::raii::DescriptorPool(self->device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *self->uiDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*self->uiDescriptorSetLayout
    };
    self->uiDescriptorSets = vk::raii::DescriptorSets(self->device, allocInfo);

    vk::DescriptorImageInfo fontInfo{
        .sampler = self->uiFontTexture.textureSampler,
        .imageView = self->uiFontTexture.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };
    vk::WriteDescriptorSet write{
        .dstSet = *self->uiDescriptorSets[0],
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &fontInfo
    };
    self->device.updateDescriptorSets({write}, nullptr);

    vk::PushConstantRange pushConstRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = 16u
    };
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*self->uiDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstRange
    };
    self->uiPipelineLayout = vk::raii::PipelineLayout(self->device, pipelineLayoutInfo);

    vk::raii::ShaderModule shaderModule = self->createShaderModule(
        self->readFile(std::string(VK_SHADERS_DIR) + "imgui.spv"));
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName = "vertMain"
    };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule,
        .pName = "fragMain"
    };
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(ImDrawVert),
        .inputRate = vk::VertexInputRate::eVertex
    };
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

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False
    };
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1
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

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False
    };

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
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                         vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };

    vk::Format depthFormat = self->findDepthFormat();
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
            .layout = self->uiPipelineLayout,
            .renderPass = nullptr
        },
        {
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &self->swapChainImageFormat,
            .depthAttachmentFormat = depthFormat
        }
    };
    self->uiPipeline = vk::raii::Pipeline(self->device, nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

    self->uiFrameBuffers.resize(self->swapChainImages.size());
    return true;
}

void vkrUpdateUIFrame(VulkanBase* self)
{
    if (!self->uiEnabled || ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(self->swapChainExtent.width),
                             static_cast<float>(self->swapChainExtent.height));
    io.DeltaTime = self->platform->frameTimer > 0.0f ? self->platform->frameTimer : (1.0f / 60.0f);

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(self->platform->window, &mouseX, &mouseY);
    io.MousePos = ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
    io.MouseDown[0] = glfwGetMouseButton(self->platform->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    io.MouseDown[1] = glfwGetMouseButton(self->platform->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    io.MouseDown[2] = glfwGetMouseButton(self->platform->window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    ImGui::NewFrame();
    vkrUpdateUIPanel(self);
    ImGui::Render();
}

void vkrRecordUICmdBuffer(VulkanBase* self, vk::raii::CommandBuffer& cmdBuffer)
{
    vkrRecordUICmdBuffer(self, cmdBuffer, self->currentFrame);
}

void vkrRecordUICmdBuffer(VulkanBase* self, vk::raii::CommandBuffer& cmdBuffer, uint32_t frameIndex)
{
    if (!self->uiEnabled || ImGui::GetCurrentContext() == nullptr ||
        self->uiPipeline == vk::raii::Pipeline(nullptr))
    {
        return;
    }

    ImDrawData* drawData = ImGui::GetDrawData();
    if (drawData == nullptr || drawData->TotalVtxCount <= 0)
    {
        return;
    }

    auto& fb = self->uiFrameBuffers[frameIndex];
    const size_t vertexBytes = static_cast<size_t>(drawData->TotalVtxCount) * sizeof(ImDrawVert);
    const size_t indexBytes = static_cast<size_t>(drawData->TotalIdxCount) * sizeof(ImDrawIdx);

    vkrReallocateUIMemory(self, fb, vertexBytes, indexBytes);

    ImDrawVert* vtxDst = reinterpret_cast<ImDrawVert*>(fb.vertexMapped);
    ImDrawIdx* idxDst = reinterpret_cast<ImDrawIdx*>(fb.indexMapped);
    for (int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        memcpy(vtxDst, cmdList->VtxBuffer.Data,
               static_cast<size_t>(cmdList->VtxBuffer.Size) * sizeof(ImDrawVert));
        memcpy(idxDst, cmdList->IdxBuffer.Data,
               static_cast<size_t>(cmdList->IdxBuffer.Size) * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
    }

    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *self->uiPipeline);
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *self->uiPipelineLayout, 0, *self->uiDescriptorSets[0], nullptr);
    cmdBuffer.bindVertexBuffers(0, *fb.vertexBuffer, {0});
    cmdBuffer.bindIndexBuffer(*fb.indexBuffer, 0,
        sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

    UiPushConsts pc;
    pc.scale = glm::vec2(2.0f / float(drawData->DisplaySize.x),
                          2.0f / float(drawData->DisplaySize.y));
    pc.translate = glm::vec2(-1.0f, -1.0f);
    cmdBuffer.pushConstants(*self->uiPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
        vk::ArrayProxy<const UiPushConsts>(1, &pc));

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

            if (clipRect.x < float(self->swapChainExtent.width) &&
                clipRect.y < float(self->swapChainExtent.height) &&
                clipRect.z >= 0.0f && clipRect.w >= 0.0f)
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
                cmdBuffer.setScissor(0, scissor);
                cmdBuffer.drawIndexed(cmd->ElemCount, 1,
                    globalIndexOffset + indexOffset, globalVertexOffset, 0);
            }

            indexOffset += cmd->ElemCount;
        }
        globalIndexOffset += static_cast<uint32_t>(cmdList->IdxBuffer.Size);
        globalVertexOffset += cmdList->VtxBuffer.Size;
    }
}

void vkrShutdownUI(VulkanBase* self)
{
    if (ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    self->device.waitIdle();

    ImGui::DestroyContext();
}

void vkrUpdateUIPanel(VulkanBase* self)
{
    self->updateUIPanel();
}

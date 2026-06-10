#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <cstdint>

// ====================================================================
// VkrPipelineBuilder — Eliminates repeated Vulkan struct boilerplate
// across all pipeline creation sites.
// Uses designated initializers (compatible with VULKAN_HPP_NO_STRUCT_CONSTRUCTORS).
// ====================================================================

struct GraphicsPipelineBuilder
{
    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    vk::PipelineVertexInputStateCreateInfo vertexInput{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False,
    };

    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1, .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
    };

    vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA,
    };

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    std::vector<vk::Format> colorFormats;
    vk::Format depthFormat = vk::Format::eUndefined;
    std::vector<vk::PushConstantRange> pushConstants;

    GraphicsPipelineBuilder& addShader(vk::ShaderStageFlagBits stage,
                                       vk::ShaderModule module,
                                       const char* entry)
    {
        stages.push_back({.stage = stage, .module = module, .pName = entry});
        return *this;
    }

    template<typename BindingArray, typename AttrArray>
    GraphicsPipelineBuilder& setVertexInput(const BindingArray& bindings,
                                            const AttrArray& attributes)
    {
        m_bindingStorage.assign(bindings.begin(), bindings.end());
        m_attrStorage.assign(attributes.begin(), attributes.end());
        vertexInput = vk::PipelineVertexInputStateCreateInfo{
            .vertexBindingDescriptionCount = static_cast<uint32_t>(m_bindingStorage.size()),
            .pVertexBindingDescriptions = m_bindingStorage.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(m_attrStorage.size()),
            .pVertexAttributeDescriptions = m_attrStorage.data(),
        };
        return *this;
    }

    GraphicsPipelineBuilder& setVertexInput(
        const vk::VertexInputBindingDescription& binding,
        const vk::VertexInputAttributeDescription& attr)
    {
        m_bindingStorage = { binding };
        m_attrStorage = { attr };
        vertexInput = vk::PipelineVertexInputStateCreateInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = m_bindingStorage.data(),
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = m_attrStorage.data(),
        };
        return *this;
    }

    GraphicsPipelineBuilder& setTopology(vk::PrimitiveTopology topo) {
        inputAssembly.topology = topo; return *this;
    }
    GraphicsPipelineBuilder& setCullMode(vk::CullModeFlags mode) {
        rasterizer.cullMode = mode; return *this;
    }
    GraphicsPipelineBuilder& setPolygonMode(vk::PolygonMode mode) {
        rasterizer.polygonMode = mode; return *this;
    }
    GraphicsPipelineBuilder& setFrontFace(vk::FrontFace ff) {
        rasterizer.frontFace = ff; return *this;
    }
    GraphicsPipelineBuilder& setDepthTest(bool enable) {
        depthStencil.depthTestEnable = enable ? vk::True : vk::False; return *this;
    }
    GraphicsPipelineBuilder& setDepthWrite(bool enable) {
        depthStencil.depthWriteEnable = enable ? vk::True : vk::False; return *this;
    }
    GraphicsPipelineBuilder& setDepthCompareOp(vk::CompareOp op) {
        depthStencil.depthCompareOp = op; return *this;
    }
    GraphicsPipelineBuilder& setDepthBias(float cf, float clamp, float sf) {
        rasterizer.depthBiasEnable = vk::True;
        rasterizer.depthBiasConstantFactor = cf;
        rasterizer.depthBiasClamp = clamp;
        rasterizer.depthBiasSlopeFactor = sf;
        return *this;
    }
    GraphicsPipelineBuilder& setBlendEnable(bool enable) {
        blendAttachment.blendEnable = enable ? vk::True : vk::False; return *this;
    }
    GraphicsPipelineBuilder& setLineWidth(float w) {
        rasterizer.lineWidth = w; return *this;
    }
    GraphicsPipelineBuilder& addColorAttachment(vk::Format format) {
        colorFormats.push_back(format); return *this;
    }
    GraphicsPipelineBuilder& setColorFormats(std::initializer_list<vk::Format> fmts) {
        colorFormats.assign(fmts); return *this;
    }
    GraphicsPipelineBuilder& setDepthFormat(vk::Format fmt) {
        depthFormat = fmt; return *this;
    }
    GraphicsPipelineBuilder& addPushConstant(vk::ShaderStageFlags stages, uint32_t offset, uint32_t size) {
        pushConstants.push_back({.stageFlags = stages, .offset = offset, .size = size});
        return *this;
    }

    vk::PipelineColorBlendStateCreateInfo colorBlending() const {
        if (colorFormats.empty())
            return {.attachmentCount = 0, .pAttachments = nullptr};
        return {
            .attachmentCount = static_cast<uint32_t>(colorFormats.size()),
            .pAttachments = &blendAttachment,
        };
    }
    vk::PipelineDynamicStateCreateInfo dynamicState() const {
        return {
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };
    }
    vk::PipelineRenderingCreateInfo renderingInfo() const {
        return {
            .colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
            .pColorAttachmentFormats = colorFormats.empty() ? nullptr : colorFormats.data(),
            .depthAttachmentFormat = depthFormat,
        };
    }

    vk::raii::Pipeline build(vk::raii::Device& dev,
                             vk::PipelineLayout layout,
                             vk::RenderPass renderPass = nullptr) const
    {
        auto cb = colorBlending();
        auto ds = dynamicState();
        auto ri = renderingInfo();

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
            vk::GraphicsPipelineCreateInfo{
                .stageCount = static_cast<uint32_t>(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertexInput,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &cb,
                .pDynamicState = &ds,
                .layout = layout,
                .renderPass = renderPass,
            },
            ri,
        };
        return vk::raii::Pipeline(dev, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
    }

private:
    std::vector<vk::VertexInputBindingDescription> m_bindingStorage;
    std::vector<vk::VertexInputAttributeDescription> m_attrStorage;
};

// ====================================================================
// ComputePipelineBuilder
// ====================================================================
struct ComputePipelineBuilder
{
    vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .pName = "compMain",
    };

    ComputePipelineBuilder& setShader(vk::ShaderModule module,
                                       const char* entry = "compMain")
    {
        stage.module = module;
        stage.pName = entry;
        return *this;
    }

    vk::raii::Pipeline build(vk::raii::Device& dev, vk::PipelineLayout layout) const
    {
        return vk::raii::Pipeline(dev, nullptr,
            vk::ComputePipelineCreateInfo{ .stage = stage, .layout = layout });
    }
};

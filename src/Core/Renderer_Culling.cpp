#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <chrono>
#include <algorithm>
#include <cmath>

#if RENDERING_LEVEL == 8

namespace
{
    struct CullingInstanceData
    {
        glm::mat4 model;
        glm::vec4 color;
    };

    struct DrawCommand
    {
        uint32_t indexCount;
        uint32_t instanceCount;
        uint32_t firstIndex;
        int32_t vertexOffset;
        uint32_t firstInstance;
    };

    struct CullingStats
    {
        uint32_t totalCount;
        uint32_t visibleCount;
        float gpuMs;
        float frameMs;
    };

    static constexpr uint32_t kWorkgroupSize = 64u;
static_assert(sizeof(CullingParamsUBO) % 16 == 0, "CullingParamsUBO must be 16-byte aligned");
}

bool Renderer::createCullingBuffers()
{
    try
    {
        maxInstances = scene ? scene->getMaxInstances() : 0;

        createUniformBuffers(cullingGlobalUboResources, sizeof(SceneUBO));
        createStorageBuffers(cullingInstanceBufferResources, sizeof(CullingInstanceData) * maxInstances);

        // indirectBuffer 既作为 compute 的写入目标，也要被 draw indirect 使用。
        createStorageBuffers(cullingIndirectBufferResources, sizeof(DrawCommand),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
        createStorageBuffers(cullingVisibleBufferResources, sizeof(uint32_t) * maxInstances);
        // statsBuffer 由 compute 写入，随后需要拷贝到 CPU 读回，所以标记 TransferSrc。
        createStorageBuffers(cullingStatsBufferResources, sizeof(CullingStats),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        createUniformBuffers(cullingParamsBufferResources, sizeof(CullingParamsUBO));

        // 可见数量在 GPU 上累加（storage），并通过拷贝更新/读取，所以需要 TransferDst/TransferSrc。
        // 创建在 DeviceLocal（GPU 本地内存），这块不能直接给 CPU map。
        // 每帧 compute 写完后，先做 barrier（ShaderWrite -> TransferRead），再拷贝：见下面945行commandBuffer.copyBuffer
        createBuffer(sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal, cullingVisibleCountBuffer, cullingVisibleCountMemory);

        // statsReadbackBuffer 仅作为 GPU->CPU 的拷贝目标，所以只要 TransferDst + 可映射内存。
        // 因为是 HostVisible | HostCoherent，所以能 map 到 cullingVisibleCountMapped。
        createBuffer(sizeof(CullingStats), vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            cullingStatsReadbackBuffer, cullingStatsReadbackMemory);
        cullingStatsReadbackMapped = cullingStatsReadbackMemory.mapMemory(0, sizeof(CullingStats));

        // visibleReadbackBuffer 仅作为 GPU->CPU 的拷贝目标，所以只要 TransferDst + 可映射内存。
        createBuffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            cullingVisibleReadbackBuffer, cullingVisibleReadbackMemory);
        cullingVisibleCountMapped = cullingVisibleReadbackMemory.mapMemory(0, sizeof(uint32_t));

        // Timestamp QueryPool 用于统计每帧 culling 的 GPU 耗时。
        // 每个 in-flight frame 占用 2 个槽位：
        // - slot(frame*2+0): 在 culling 命令开头写入（TopOfPipe）
        // - slot(frame*2+1): 在 culling 命令结尾写入（BottomOfPipe）
        // CPU 读取两者差值并乘以 timestampPeriod，换算为毫秒。
        // 所以 queryCount = MAX_FRAMES_IN_FLIGHT * 2。
        vk::QueryPoolCreateInfo queryInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = MAX_FRAMES_IN_FLIGHT * 2u
        };
        cullingTimestampQueryPool = vk::raii::QueryPool(device, queryInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling buffers: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingDescriptorSetLayouts()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> depthBindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex}
        };
        vk::DescriptorSetLayoutCreateInfo depthLayoutInfo{
            .bindingCount = static_cast<uint32_t>(depthBindings.size()),
            .pBindings = depthBindings.data()
        };
        cullingDepthDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, depthLayoutInfo);

        std::vector<vk::DescriptorSetLayoutBinding> drawBindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex},
            {.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex}
        };
        vk::DescriptorSetLayoutCreateInfo drawLayoutInfo{
            .bindingCount = static_cast<uint32_t>(drawBindings.size()),
            .pBindings = drawBindings.data()
        };
        cullingDrawDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, drawLayoutInfo);

        std::vector<vk::DescriptorSetLayoutBinding> cullingBindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 5, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 7, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 8, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(cullingBindings.size()),
            .pBindings = cullingBindings.data()
        };
        cullingDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling descriptor set layouts: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingDescriptorPools()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> cullingPoolSizes = {
            // binding0(SceneUBO) + binding6(CullingParamsUBO) = 2 个/帧。
            // 这里配成 *4，预留一倍冗余，避免后续扩展时池子不够，下同
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u},
            // binding1(instance) + binding2(indirect) + binding3(visible) + binding4(stats) + binding5(visibleCount) = 5 个/帧。
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 10u},
            // binding7(Hi-Z 纹理) + binding8(采样器兼容位) = 2 个/帧。
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u}
        };
        vk::DescriptorPoolCreateInfo cullingPoolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(cullingPoolSizes.size()),
            .pPoolSizes = cullingPoolSizes.data()
        };
        cullingDescriptorPool = vk::raii::DescriptorPool(device, cullingPoolInfo);

        std::vector<vk::DescriptorPoolSize> depthPoolSizes = {
            // binding0(SceneUBO) = 1 个 UBO/帧。
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u},
            // binding1(instanceBuffer) = 1 个 SSBO/帧。
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u}
        };
        vk::DescriptorPoolCreateInfo depthPoolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(depthPoolSizes.size()),
            .pPoolSizes = depthPoolSizes.data()
        };
        cullingDepthDescriptorPool = vk::raii::DescriptorPool(device, depthPoolInfo);

        std::vector<vk::DescriptorPoolSize> drawPoolSizes = {
            // binding0(SceneUBO) = 1 个 UBO/帧。
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u},
            // binding1(instanceBuffer) + binding2(visibleIndices) = 2 个 SSBO/帧；
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u}
        };
        vk::DescriptorPoolCreateInfo drawPoolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(drawPoolSizes.size()),
            .pPoolSizes = drawPoolSizes.data()
        };
        cullingDrawDescriptorPool = vk::raii::DescriptorPool(device, drawPoolInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling descriptor pools: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createCullingDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> depthLayouts(MAX_FRAMES_IN_FLIGHT, *cullingDepthDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo depthAllocInfo{
        .descriptorPool = *cullingDepthDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(depthLayouts.size()),
        .pSetLayouts = depthLayouts.data()
    };
    cullingDepthDescriptorSets = vk::raii::DescriptorSets(device, depthAllocInfo);

    std::vector<vk::DescriptorSetLayout> cullingLayouts(MAX_FRAMES_IN_FLIGHT, *cullingDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo cullingAllocInfo{
        .descriptorPool = *cullingDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(cullingLayouts.size()),
        .pSetLayouts = cullingLayouts.data()
    };
    cullingDescriptorSets = vk::raii::DescriptorSets(device, cullingAllocInfo);

    std::vector<vk::DescriptorSetLayout> drawLayouts(MAX_FRAMES_IN_FLIGHT, *cullingDrawDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo drawAllocInfo{
        .descriptorPool = *cullingDrawDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(drawLayouts.size()),
        .pSetLayouts = drawLayouts.data()
    };
    cullingDrawDescriptorSets = vk::raii::DescriptorSets(device, drawAllocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneInfo{ .buffer = *cullingGlobalUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceInfo{ .buffer = *cullingInstanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(CullingInstanceData) * maxInstances };
        vk::DescriptorBufferInfo drawInfo{ .buffer = *cullingIndirectBufferResources.Buffers[i], .offset = 0, .range = sizeof(DrawCommand) };
        vk::DescriptorBufferInfo visibleInfo{ .buffer = *cullingVisibleBufferResources.Buffers[i], .offset = 0, .range = sizeof(uint32_t) * maxInstances };
        vk::DescriptorBufferInfo statsInfo{ .buffer = *cullingStatsBufferResources.Buffers[i], .offset = 0, .range = sizeof(CullingStats) };
        vk::DescriptorBufferInfo visibleCountInfo{ .buffer = cullingVisibleCountBuffer, .offset = 0, .range = sizeof(uint32_t) };
        vk::DescriptorBufferInfo paramsInfo{ .buffer = *cullingParamsBufferResources.Buffers[i], .offset = 0, .range = sizeof(CullingParamsUBO) };

        vk::DescriptorImageInfo depthInfo{ .sampler = cullingHiZTexture.textureSampler, .imageView = cullingHiZTexture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo depthInfo2{ .sampler = cullingHiZTexture.textureSampler, .imageView = cullingHiZTexture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> depthWrites = {
            {.dstSet = *cullingDepthDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo},
            {.dstSet = *cullingDepthDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceInfo}
        };
        device.updateDescriptorSets(depthWrites, nullptr);

        std::vector<vk::WriteDescriptorSet> cullingWrites = {
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &drawInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &statsInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleCountInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo },
            {.dstSet = *cullingDescriptorSets[i], .dstBinding = 8, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo2 }
        };
        device.updateDescriptorSets(cullingWrites, nullptr);

        vk::DescriptorBufferInfo drawVisibleInfo{ .buffer = *cullingVisibleBufferResources.Buffers[i], .offset = 0, .range = sizeof(uint32_t) * maxInstances };
        std::vector<vk::WriteDescriptorSet> drawWrites = {
            {.dstSet = *cullingDrawDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo},
            {.dstSet = *cullingDrawDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceInfo},
            {.dstSet = *cullingDrawDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &drawVisibleInfo}
        };
        device.updateDescriptorSets(drawWrites, nullptr);
    }
}

bool Renderer::createCullingPipelines()
{
    try
    {
        vk::raii::ShaderModule cullingModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_comp.spv"));
        vk::PipelineShaderStageCreateInfo shaderStage{ .stage = vk::ShaderStageFlagBits::eCompute, .module = cullingModule, .pName = "compMain" };
        vk::PipelineLayoutCreateInfo layoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*cullingDescriptorSetLayout, .pushConstantRangeCount = 0 };
        cullingPipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

        vk::ComputePipelineCreateInfo pipelineInfo{ .stage = shaderStage, .layout = cullingPipelineLayout };
        cullingPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);

        vk::raii::ShaderModule depthModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_depth.spv"));
        vk::PipelineShaderStageCreateInfo depthStage{ .stage = vk::ShaderStageFlagBits::eVertex, .module = depthModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo depthStages[] = { depthStage };

        auto bindingDescription = Vertex::getBindingDescription();
        auto positionOnlyAttributes = Vertex::getPositionOnlyAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo depthVertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(positionOnlyAttributes.size()),
            .pVertexAttributeDescriptions = positionOnlyAttributes.data()
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
            .depthCompareOp = vk::CompareOp::eLessOrEqual,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 0, .pAttachments = nullptr };
        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo depthLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*cullingDepthDescriptorSetLayout, .pushConstantRangeCount = 0 };
        cullingDepthPipelineLayout = vk::raii::PipelineLayout(device, depthLayoutInfo);

        vk::Format depthFormat = findDepthFormat();
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> depthPipelineInfo = {
            {
                .stageCount = 1,
                .pStages = depthStages,
                .pVertexInputState = &depthVertexInput,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = cullingDepthPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 0,
                .pColorAttachmentFormats = nullptr,
                .depthAttachmentFormat = depthFormat
            }
        };
        cullingDepthPipeline = vk::raii::Pipeline(device, nullptr, depthPipelineInfo.get<vk::GraphicsPipelineCreateInfo>());

        vk::raii::ShaderModule drawModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_draw.spv"));
        vk::PipelineShaderStageCreateInfo drawVertStage{ .stage = vk::ShaderStageFlagBits::eVertex, .module = drawModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo drawFragStage{ .stage = vk::ShaderStageFlagBits::eFragment, .module = drawModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo drawStages[] = { drawVertStage, drawFragStage };

        auto drawAttributes = Vertex::getPositionNormalAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo drawVertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(drawAttributes.size()),
            .pVertexAttributeDescriptions = drawAttributes.data()
        };
        vk::PipelineRasterizationStateCreateInfo drawRasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = vk::False,
                                                                   .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo drawColorBlend{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };
        vk::PipelineLayoutCreateInfo drawLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*cullingDrawDescriptorSetLayout, .pushConstantRangeCount = 0 };
        cullingDrawPipelineLayout = vk::raii::PipelineLayout(device, drawLayoutInfo);

        vk::Format swapDepthFormat = findDepthFormat();
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> drawPipelineInfo = {
            {
                .stageCount = 2,
                .pStages = drawStages,
                .pVertexInputState = &drawVertexInput,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &drawRasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &drawColorBlend,
                .pDynamicState = &dynamicState,
                .layout = cullingDrawPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = swapDepthFormat
            }
        };
        cullingDrawPipeline = vk::raii::Pipeline(device, nullptr, drawPipelineInfo.get<vk::GraphicsPipelineCreateInfo>());

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling pipelines: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingDepthResources()
{
    try
    {
        cullingDepthExtent = swapChainExtent;
        vk::Format depthFormat = findDepthFormat();
        createImage(cullingDepthExtent.width, cullingDepthExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, cullingDepthTexture);

        cullingDepthTexture.textureImageView = createImageView(cullingDepthTexture.textureImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
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
            .maxLod = 0.0f
        };
        cullingDepthTexture.textureSampler = vk::raii::Sampler(device, samplerInfo);

        cullingDepthLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling depth resources: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingHiZResources()
{
    try
    {
        cullingHiZMipCount = static_cast<uint32_t>(std::floor(std::log2(std::max(cullingDepthExtent.width, cullingDepthExtent.height)))) + 1u;
        createImage(
            cullingDepthExtent.width,
            cullingDepthExtent.height,
            cullingHiZMipCount,
            vk::Format::eR32Sfloat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            cullingHiZTexture);

        cullingHiZTexture.textureImageView = createImageView(cullingHiZTexture.textureImage, vk::Format::eR32Sfloat, vk::ImageAspectFlagBits::eColor, cullingHiZMipCount);

        cullingHiZMipViews.clear();
        cullingHiZMipViews.reserve(cullingHiZMipCount);
        for (uint32_t mip = 0; mip < cullingHiZMipCount; ++mip)
        {
            vk::ImageViewCreateInfo viewInfo{
                .image = cullingHiZTexture.textureImage,
                .viewType = vk::ImageViewType::e2D,
                .format = vk::Format::eR32Sfloat,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1}
            };
            cullingHiZMipViews.emplace_back(device, viewInfo);
        }

        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .minLod = 0.0f,
            .maxLod = static_cast<float>(cullingHiZMipCount - 1)
        };
        cullingHiZTexture.textureSampler = vk::raii::Sampler(device, samplerInfo);

        cullingHiZLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z resources: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingHiZDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };
        cullingHiZDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingHiZDescriptorPool()
{
    try
    {
        const uint32_t setCount = MAX_FRAMES_IN_FLIGHT * cullingHiZMipCount;
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eStorageImage, .descriptorCount = setCount * 2u},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = setCount * 2u}
        };
        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = setCount,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        cullingHiZDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createCullingHiZDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT * cullingHiZMipCount, *cullingHiZDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *cullingHiZDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };
    cullingHiZDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);
    updateCullingHiZDescriptorSets();
}

void Renderer::updateCullingHiZDescriptorSets()
{
    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame)
    {
        for (uint32_t mip = 0; mip < cullingHiZMipCount; ++mip)
        {
            const uint32_t setIndex = frame * cullingHiZMipCount + mip;
            vk::DescriptorImageInfo srcInfo{};
            if (mip == 0)
            {
                srcInfo = {.sampler = cullingDepthTexture.textureSampler, .imageView = cullingDepthTexture.textureImageView, .imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal};
            }
            else
            {
                srcInfo = {.sampler = cullingHiZTexture.textureSampler, .imageView = cullingHiZMipViews[mip - 1], .imageLayout = vk::ImageLayout::eGeneral};
            }
            vk::DescriptorImageInfo dstInfo{ .sampler = nullptr, .imageView = cullingHiZMipViews[mip], .imageLayout = vk::ImageLayout::eGeneral };
            vk::DescriptorImageInfo srcCopyInfo = srcInfo;
            vk::DescriptorImageInfo srcSamplerInfo = srcInfo;

            std::vector<vk::WriteDescriptorSet> writes = {
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dstInfo},
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dstInfo},
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &srcCopyInfo},
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &srcSamplerInfo}
            };
            device.updateDescriptorSets(writes, nullptr);
        }
    }
}

bool Renderer::createCullingHiZPipeline()
{
    try
    {
        vk::raii::ShaderModule module = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_hiz_build.spv"));
        vk::PipelineShaderStageCreateInfo stage{ .stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "compMain" };
        vk::PipelineLayoutCreateInfo layoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*cullingHiZDescriptorSetLayout, .pushConstantRangeCount = 0 };
        cullingHiZPipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);
        vk::ComputePipelineCreateInfo pipelineInfo{ .stage = stage, .layout = cullingHiZPipelineLayout };
        cullingHiZPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z pipeline: " << e.what() << std::endl;
        return false;
    }
}

// 用compute来实现cullingHiZTexture的mipmap分级（下采样的方式）
void Renderer::recordCullingHiZ(vk::raii::CommandBuffer& commandBuffer)
{
    vk::ImageMemoryBarrier2 hiZInitBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = {},
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = cullingHiZLayout,
        .newLayout = vk::ImageLayout::eGeneral,         // 允许 compute 写
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cullingHiZTexture.textureImage,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, cullingHiZMipCount, 0, 1}
    };
    vk::DependencyInfo hiZInitDep{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hiZInitBarrier };
    commandBuffer.pipelineBarrier2(hiZInitDep);
    cullingHiZLayout = vk::ImageLayout::eGeneral;

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *cullingHiZPipeline);
    for (uint32_t mip = 0; mip < cullingHiZMipCount; ++mip)
    {
        const uint32_t setIndex = currentFrame * cullingHiZMipCount + mip;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *cullingHiZPipelineLayout, 0, *cullingHiZDescriptorSets[setIndex], nullptr);

        const uint32_t mipWidth = std::max(1u, cullingDepthExtent.width >> mip);
        const uint32_t mipHeight = std::max(1u, cullingDepthExtent.height >> mip);
        const uint32_t groupX = (mipWidth + 7u) / 8u;
        const uint32_t groupY = (mipHeight + 7u) / 8u;
        commandBuffer.dispatch(groupX, groupY, 1);

        vk::MemoryBarrier2 mipBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
        };
        vk::DependencyInfo dep{ .memoryBarrierCount = 1, .pMemoryBarriers = &mipBarrier };
        commandBuffer.pipelineBarrier2(dep);
    }

    vk::ImageMemoryBarrier2 hiZToReadBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cullingHiZTexture.textureImage,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, cullingHiZMipCount, 0, 1}
    };
    vk::DependencyInfo hiZToReadDep{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hiZToReadBarrier };
    commandBuffer.pipelineBarrier2(hiZToReadDep);
    cullingHiZLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

bool Renderer::createCullingCommandPool()
{
    try
    {
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.computeFamily.value()
        };
        computeCommandPool = vk::raii::CommandPool(device, poolInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling command pool: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingCommandBuffers()
{
    try
    {
        computeCommandBuffers.clear();
        computeCommandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *computeCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };

        computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling command buffers: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createCullingSyncObjects()
{
    try
    {
        cullingCompleteSemaphores.clear();
        cullingCompleteSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semaphoreInfo{};
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            cullingCompleteSemaphores.emplace_back(device, semaphoreInfo);
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling sync objects: " << e.what() << std::endl;
        return false;
    }
}

static void extractFrustumPlanes(const glm::mat4& matrix, glm::vec4* planesOut)
{
    glm::mat4 m = matrix;

    planesOut[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]);
    planesOut[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]);
    planesOut[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]);
    planesOut[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]);
    planesOut[4] = glm::vec4(m[0][2], m[1][2], m[2][2], m[3][2]);
    planesOut[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]);

    for (int i = 0; i < 6; ++i)
    {
        float len = glm::length(glm::vec3(planesOut[i]));
        if (len > 0.0f)
        {
            planesOut[i] /= len;
        }
    }
}

static void fillDrawCommand(DrawCommand& drawCmd, const Mesh& mesh, uint32_t instanceCount)
{
    drawCmd.indexCount = static_cast<uint32_t>(mesh.indices.size());
    drawCmd.instanceCount = instanceCount;
    drawCmd.firstIndex = 0;
    drawCmd.vertexOffset = 0;
    drawCmd.firstInstance = 0;
}

void Renderer::updateCullingBuffers(uint32_t currentImage)
{
    if (!scene)
    {
        return;
    }

    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1;
    memcpy(cullingGlobalUboResources.BuffersMapped[currentImage], &sceneUbo, sizeof(sceneUbo));

    static auto lastFrame = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    cullingFrameMs = std::chrono::duration<float, std::milli>(now - lastFrame).count();
    lastFrame = now;

    std::vector<RenderInstance> renderInstances;
    scene->world.collectRenderInstances(MeshTag::Cube, renderInstances, maxInstances);
    cullingTotalCountCpu = static_cast<uint32_t>(renderInstances.size());

    std::vector<CullingInstanceData> instanceData;
    instanceData.reserve(renderInstances.size());
    for (const auto& instance : renderInstances)
    {
        instanceData.push_back(CullingInstanceData{ instance.model, instance.color });
    }

    if (!instanceData.empty())
    {
        memcpy(cullingInstanceBufferResources.BuffersMapped[currentImage], instanceData.data(), sizeof(CullingInstanceData) * instanceData.size());
    }

    DrawCommand drawCmd{};
    fillDrawCommand(drawCmd, resourceManager->meshes[scene->cubeMeshIndex], cullingEnabled ? 0u : cullingTotalCountCpu);
    memcpy(cullingIndirectBufferResources.BuffersMapped[currentImage], &drawCmd, sizeof(drawCmd));

    CullingParamsUBO params{};
    extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, params.frustumPlanes);
    params.aabbMin = glm::vec4(-0.5f, -0.5f, -0.5f, 0.0f);
    params.aabbMax = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
    params.hiZInfo = glm::vec4(
        static_cast<float>(cullingDepthExtent.width),
        static_cast<float>(cullingDepthExtent.height),
        static_cast<float>(cullingHiZMipCount),
        0.0015f);
    params.totalInstances = cullingTotalCountCpu;
    params.useCulling = cullingEnabled ? 1u : 0u;
    memcpy(cullingParamsBufferResources.BuffersMapped[currentImage], &params, sizeof(CullingParamsUBO));
}

void Renderer::recordCullingCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = computeCommandBuffers[currentFrame];
    commandBuffer.begin({});

    commandBuffer.resetQueryPool(*cullingTimestampQueryPool, currentFrame * 2u, 2u);
    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *cullingTimestampQueryPool, currentFrame * 2u);

    vk::ImageMemoryBarrier2 toDepthAttachBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = {},
        .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .oldLayout = cullingDepthLayout,
        .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cullingDepthTexture.textureImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 }
    };
    vk::DependencyInfo toDepthAttachDepInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDepthAttachBarrier };
    commandBuffer.pipelineBarrier2(toDepthAttachDepInfo);
    cullingDepthLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    vk::RenderingAttachmentInfo depthAttachmentInfo{
        .imageView = cullingDepthTexture.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearDepth
    };
    vk::RenderingInfo depthRenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = cullingDepthExtent},
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(depthRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(cullingDepthExtent.width), static_cast<float>(cullingDepthExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), cullingDepthExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *cullingDepthPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *cullingDepthPipelineLayout, 0, *cullingDepthDescriptorSets[currentFrame], nullptr);

    auto& mesh = resourceManager->meshes[scene->cubeMeshIndex];
    commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
    commandBuffer.drawIndexed(static_cast<uint32_t>(mesh.indices.size()), cullingTotalCountCpu, 0, 0, 0);
    commandBuffer.endRendering();

    // 深度 prepass 完成后，把深度图从 DepthAttachment 切为 DepthReadOnly，
    // 供后续 Hi-Z 构建与 culling compute 进行采样读取。
    vk::ImageMemoryBarrier2 toDepthReadBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .newLayout = vk::ImageLayout::eDepthReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cullingDepthTexture.textureImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 }
    };
    vk::DependencyInfo toDepthReadDepInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDepthReadBarrier };
    commandBuffer.pipelineBarrier2(toDepthReadDepInfo);
    cullingDepthLayout = vk::ImageLayout::eDepthReadOnlyOptimal;

    // 构建 Hi-Z 金字塔（逐 mip 下采样），供遮挡测试按 lod 采样。
    recordCullingHiZ(commandBuffer);

    // 每帧在 compute 前清零可见计数器（visibleCountBuffer[0]），
    // 避免沿用上一帧结果导致 instanceCount 累加错误。
    commandBuffer.fillBuffer(cullingVisibleCountBuffer, 0, sizeof(uint32_t), 0u);
    vk::BufferMemoryBarrier visibleCountResetBarrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = cullingVisibleCountBuffer,
        .offset = 0,
        .size = sizeof(uint32_t)
    };
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, { visibleCountResetBarrier }, {}
    );

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *cullingPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *cullingPipelineLayout, 0, *cullingDescriptorSets[currentFrame], nullptr);

    // 线程组数量 = ceil(totalInstances / kWorkgroupSize)
    // kWorkgroupSize 与 shader 的 [numthreads] 保持一致（当前 64）。
    uint32_t dispatchCount = (cullingTotalCountCpu + kWorkgroupSize - 1u) / kWorkgroupSize;
    commandBuffer.dispatch(dispatchCount, 1, 1);

    // compute 写完 stats/count 后，转到 transfer 阶段做 copy readback，
    // 需要 barrier 保证 ShaderWrite -> TransferRead 可见。
    vk::BufferMemoryBarrier statsBarrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = *cullingStatsBufferResources.Buffers[currentFrame],
        .offset = 0,
        .size = sizeof(CullingStats)
    };
    vk::BufferMemoryBarrier countBarrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = cullingVisibleCountBuffer,
        .offset = 0,
        .size = sizeof(uint32_t)
    };
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, { statsBarrier, countBarrier }, {}
    );

    // 把 GPU 侧统计数据拷贝到 HostVisible 的 readback buffer，
    // 下一阶段 CPU 在 updateCullingStats() 中直接读取 mapped 内存。
    commandBuffer.copyBuffer(*cullingStatsBufferResources.Buffers[currentFrame], *cullingStatsReadbackBuffer, vk::BufferCopy(0, 0, sizeof(CullingStats)));
    commandBuffer.copyBuffer(cullingVisibleCountBuffer, cullingVisibleReadbackBuffer, vk::BufferCopy(0, 0, sizeof(uint32_t)));

    // 写结束时间戳：与开头时间戳做差得到本帧 culling GPU 耗时。
    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *cullingTimestampQueryPool, currentFrame * 2u + 1u);
    commandBuffer.end();
}

void Renderer::recordCullingDrawCommands(vk::raii::CommandBuffer& commandBuffer)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *cullingDrawPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *cullingDrawPipelineLayout, 0, *cullingDrawDescriptorSets[currentFrame], nullptr);

    auto& mesh = resourceManager->meshes[scene->cubeMeshIndex];
    commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
    // 间接绘制：GPU 直接读取 culling 结果，不经 CPU 回传实例列表。
    // 只传一个“命令缓冲地址+偏移”，实际参数在 buffer 里（VkDrawIndexedIndirectCommand）
    // compute 写了这个 buffer，用来直接绘制
    // CPU 不需要回读 DrawCommand 再发 draw 命令
    commandBuffer.drawIndexedIndirect(*cullingIndirectBufferResources.Buffers[currentFrame], 0, 1, sizeof(DrawCommand));
}

void Renderer::updateCullingUI()
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGui::Begin("Culling", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Enable Culling", &cullingEnabled);
    ImGui::Checkbox("Show Stats", &cullingShowStats);
    ImGui::Text("Instances: %u", cullingTotalCountCpu);
    if (cullingShowStats)
    {
        ImGui::Text("Visible: %u", cullingVisibleCountCpu);
        ImGui::Text("Frame: %.3f ms", cullingFrameMs);
        ImGui::Text("Culling GPU: %.3f ms", cullingGpuMs);
    }
    ImGui::End();
}

void Renderer::updateCullingStats()
{
    uint64_t timestamps[2] = {};
    vk::Device deviceHandle = *device;
    vk::Result res = deviceHandle.getQueryPoolResults(*cullingTimestampQueryPool, currentFrame * 2u, 2u, sizeof(timestamps), timestamps, sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (res == vk::Result::eSuccess)
    {
        vk::PhysicalDeviceProperties props = physicalDevice.getProperties();
        double timestampPeriod = props.limits.timestampPeriod;
        cullingGpuMs = static_cast<float>((timestamps[1] - timestamps[0]) * timestampPeriod * 1e-6);
    }

    if (cullingStatsReadbackMapped != nullptr)
    {
        auto* stats = reinterpret_cast<CullingStats*>(cullingStatsReadbackMapped);
        cullingVisibleCountCpu = stats->visibleCount;
        cullingTotalCountCpu = stats->totalCount;
    }

    if (cullingVisibleCountMapped != nullptr)
    {
        cullingVisibleCountCpu = *reinterpret_cast<uint32_t*>(cullingVisibleCountMapped);
    }
}

#endif

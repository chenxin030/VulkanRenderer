#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#if RENDERING_LEVEL == 3

bool Renderer::createPBRDescriptorSetLayout() {
    try {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {
                .binding = 0,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            },
            {
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            },
            {
                .binding = 2,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment
            }
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        pbrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create PBR descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createPBRDescriptorPool() {
    try {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2 }, // Scene + Light
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        pbrDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create PBR descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createPBRDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pbrDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *pbrDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    pbrInstanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo sceneBufferInfo{
            .buffer = *sceneUboResources.Buffers[i],
            .offset = 0,
            .range = sizeof(SceneUBO)
        };

        vk::DescriptorBufferInfo instanceBufferInfo{
            .buffer = *pbrInstanceBufferResources.Buffers[i],
            .offset = 0,
            .range = sizeof(PBRInstanceData) * MAX_OBJECTS
        };

        vk::DescriptorBufferInfo lightBufferInfo{
            .buffer = *lightUboResources.Buffers[i],
            .offset = 0,
            .range = sizeof(LightUBO)
        };

        std::vector<vk::WriteDescriptorSet> descriptorWrites = {
            {
                .dstSet = *pbrInstanceBufferResources.descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &sceneBufferInfo
            },
            {
                .dstSet = *pbrInstanceBufferResources.descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &instanceBufferInfo
            },
            {
                .dstSet = *pbrInstanceBufferResources.descriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &lightBufferInfo
            }
        };

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }
}

bool Renderer::createPBRPipeline() {
    try {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "pbr.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = vk::False,
                                                                   .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*pbrDescriptorSetLayout, .pushConstantRangeCount = 0 };

        pbrPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
                .layout = pbrPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        pbrPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create PBR graphics pipeline: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createPBRBuffers() {
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(lightUboResources, sizeof(LightUBO));
    createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * MAX_OBJECTS);
}

void Renderer::updatePBRInstanceBuffers(uint32_t currentImage) {
    // Scene UBO
    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1;
    memcpy(sceneUboResources.BuffersMapped[currentImage], &sceneUbo, sizeof(sceneUbo));

    // Instance Data (7*7 grid)
    std::vector<PBRInstanceData> instanceData(MAX_OBJECTS);
    uint32_t gridSize = sqrt(MAX_OBJECTS);
    for (uint32_t y = 0; y < gridSize; ++y) {
        for (uint32_t x = 0; x < gridSize; ++x) {
            uint32_t index = y * gridSize + x;
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(float(x - (gridSize / 2.0f)) * 1.5f, float(y - (gridSize / 2.0f)) * 1.5f, 0.0f));
            model = glm::scale(model, glm::vec3(0.7f));
            instanceData[index].model = model;
            instanceData[index].metallic = glm::clamp((float)x / (float)(gridSize - 1), 0.1f, 1.0f);
            instanceData[index].roughness = glm::clamp((float)y / (float)(gridSize - 1), 0.05f, 1.0f);
            instanceData[index].color = glm::vec3(1.0f, 0.765557f, 0.336057f);  // gold
        }
    }
    memcpy(pbrInstanceBufferResources.BuffersMapped[currentImage], instanceData.data(), sizeof(PBRInstanceData) * MAX_OBJECTS);

    // Light Animation
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    LightUBO lightUbo;
    lightUbo.lights[0] = { .position = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 400) };

    lightUbo.lights[1] = { .position = glm::vec4(-20.0f, -10.0f, 10.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 50.0f) };

    // Moving along X axis
    lightUbo.lights[2] = { .position = glm::vec4(sin(time * 0.5f) * 12.0f, 5.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 150.0f) };
    // Moving along Y axis
    lightUbo.lights[3] = { .position = glm::vec4(0.0f, cos(time * 0.5f) * 12.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 150.0f) };

    memcpy(lightUboResources.BuffersMapped[currentImage], &lightUbo, sizeof(lightUbo));
}

#endif

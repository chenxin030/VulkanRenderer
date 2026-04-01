#include "ClusteredRenderer.h"

#include <Base/Mesh.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <random>

static void generatePlane(Mesh& mesh, float width, float height)
{
    float w = width / 2.0f;
    float h = height / 2.0f;

    mesh.vertices = {
        Vertex{ .pos = { -w, 0.0f, -h }, .normal = { 0.0f, 1.0f, 0.0f }, .texCoord = { 0.0f, 0.0f } },
        Vertex{ .pos = {  w, 0.0f, -h }, .normal = { 0.0f, 1.0f, 0.0f }, .texCoord = { 1.0f, 0.0f } },
        Vertex{ .pos = {  w, 0.0f,  h }, .normal = { 0.0f, 1.0f, 0.0f }, .texCoord = { 1.0f, 1.0f } },
        Vertex{ .pos = { -w, 0.0f,  h }, .normal = { 0.0f, 1.0f, 0.0f }, .texCoord = { 0.0f, 1.0f } },
    };

    mesh.indices = {
        0, 1, 2,
        2, 3, 0,
    };
}

void ClusteredRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool ClusteredRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 5.0f, 25.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 12_clustered");
}

bool ClusteredRenderer::prepareResource()
{
    generateSphere(sphereMesh, 0.3f, 16);
    createVertexBuffer(sphereMesh);
    createIndexBuffer(sphereMesh);

    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    generatePlane(planeMesh, 50.0f, 50.0f);
    createVertexBuffer(planeMesh);
    createIndexBuffer(planeMesh);

    generateSceneLights();

    if (!createClusterBuffers()) return false;
    if (!createClusteredDescriptorSetLayout()) return false;
    if (!createClusteredDescriptorPool()) return false;
    createClusteredDescriptorSets();
    if (!createClusteredPipeline()) return false;

    if (!createComputeDescriptorSetLayout()) return false;
    if (!createComputeDescriptorPool()) return false;
    createComputeDescriptorSets();
    if (!createComputePipeline()) return false;

    if (!createClusterCommandPool()) return false;
    if (!createComputeSyncObjects()) return false;

    if (!initUI()) return false;

    return true;
}

void ClusteredRenderer::generateSceneLights()
{
    sceneLights.clear();
    sceneLights.reserve(MAX_LIGHTS);

    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> posX(-20.0f, 20.0f);
    std::uniform_real_distribution<float> posY(0.5f, 8.0f);
    std::uniform_real_distribution<float> posZ(-20.0f, 20.0f);
    std::uniform_real_distribution<float> intensity(5.0f, 50.0f);
    std::uniform_real_distribution<float> hue(0.0f, 1.0f);

    for (uint32_t i = 0; i < MAX_LIGHTS; ++i)
    {
        float h = hue(rng);
        float s = 0.8f;
        float v = 1.0f;

        float c = v * s;
        float x = c * std::abs(6.0f * std::fmod(h, 1.0f) - 3.0f - 1.0f);
        float m = v - c;

        float r, g, b;
        if (h < 1.0f / 6.0f) { r = c; g = x; b = 0.0f; }
        else if (h < 2.0f / 6.0f) { r = x; g = c; b = 0.0f; }
        else if (h < 3.0f / 6.0f) { r = 0.0f; g = c; b = x; }
        else if (h < 4.0f / 6.0f) { r = 0.0f; g = x; b = c; }
        else if (h < 5.0f / 6.0f) { r = x; g = 0.0f; b = c; }
        else { r = c; g = 0.0f; b = x; }

        PointLight light;
        light.position = glm::vec4(posX(rng), posY(rng), posZ(rng), 1.0f);
        light.color = glm::vec4(r + m, g + m, b + m, intensity(rng));
        sceneLights.push_back(light);
    }

    lightCount = static_cast<uint32_t>(sceneLights.size());
}

bool ClusteredRenderer::createClusterBuffers()
{
    try
    {
        createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
        createUniformBuffers(clusterParamsResources, sizeof(ClusterParamsUBO));
        createUniformBuffers(groundUboResources, sizeof(GroundUBO));

        createStorageBuffers(lightBufferResources, sizeof(PointLight) * MAX_LIGHTS);

        const uint32_t totalClusters = getTotalClusters();

        createBuffer(
            totalClusters * sizeof(uint32_t) * 2,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            lightGridBuffer,
            lightGridMemory);
        lightGridMapped = lightGridMemory.mapMemory(0, totalClusters * sizeof(uint32_t) * 2);

        createBuffer(
            getLightIndexBufferSize() * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            lightIndexBuffer,
            lightIndexMemory);
        lightIndexMapped = lightIndexMemory.mapMemory(0, getLightIndexBufferSize() * sizeof(uint32_t));

        createBuffer(
            getTotalClusters() * sizeof(uint32_t) * 2,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            lightGridReadbackBuffer,
            lightGridReadbackMemory);
        lightGridReadbackMapped = lightGridReadbackMemory.mapMemory(0, getTotalClusters() * sizeof(uint32_t) * 2);

        computeFence = vk::raii::Fence(device, vk::FenceCreateInfo{});

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create cluster buffers: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createClusteredDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 3, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
            { .binding = 4, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment }
        };

        clusteredDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create clustered descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createClusteredDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u }
        };

        clusteredDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create clustered descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void ClusteredRenderer::createClusteredDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *clusteredDescriptorSetLayout);
    clusteredDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *clusteredDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo lightInfo{ .buffer = *lightBufferResources.Buffers[i], .offset = 0, .range = sizeof(PointLight) * MAX_LIGHTS };
        vk::DescriptorBufferInfo gridInfo{ .buffer = lightGridBuffer, .offset = 0, .range = getTotalClusters() * sizeof(uint32_t) * 2 };
        vk::DescriptorBufferInfo indexInfo{ .buffer = lightIndexBuffer, .offset = 0, .range = getLightIndexBufferSize() * sizeof(uint32_t) };
        vk::DescriptorBufferInfo paramsInfo{ .buffer = *clusterParamsResources.Buffers[i], .offset = 0, .range = sizeof(ClusterParamsUBO) };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *clusteredDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
            { .dstSet = *clusteredDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &lightInfo },
            { .dstSet = *clusteredDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &gridInfo },
            { .dstSet = *clusteredDescriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo },
            { .dstSet = *clusteredDescriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indexInfo }
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool ClusteredRenderer::createClusteredPipeline()
{
    try
    {
        clusteredPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*clusteredDescriptorSetLayout });

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "cluster_lighting.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        const auto bindingDescription = Vertex::getBindingDescription();
        const auto attributeDescriptions = Vertex::getAttributeDescriptions();
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

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        const vk::Format depthFormat = findDepthFormat();
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
                .layout = clusteredPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        clusteredPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create clustered graphics pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createComputeDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 3, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 4, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute }
        };

        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createComputeDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u }
        };

        computeDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void ClusteredRenderer::createComputeDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);
    computeDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *computeDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo lightInfo{ .buffer = *lightBufferResources.Buffers[i], .offset = 0, .range = sizeof(PointLight) * MAX_LIGHTS };
        vk::DescriptorBufferInfo gridInfo{ .buffer = lightGridBuffer, .offset = 0, .range = getTotalClusters() * sizeof(uint32_t) * 2 };
        vk::DescriptorBufferInfo indexInfo{ .buffer = lightIndexBuffer, .offset = 0, .range = getLightIndexBufferSize() * sizeof(uint32_t) };
        vk::DescriptorBufferInfo sceneInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo paramsInfo{ .buffer = *clusterParamsResources.Buffers[i], .offset = 0, .range = sizeof(ClusterParamsUBO) };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *computeDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &lightInfo },
            { .dstSet = *computeDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &gridInfo },
            { .dstSet = *computeDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indexInfo },
            { .dstSet = *computeDescriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
            { .dstSet = *computeDescriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo }
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool ClusteredRenderer::createComputePipeline()
{
    try
    {
        computePipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*computeDescriptorSetLayout });

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "cluster_comp.spv"));
        vk::PipelineShaderStageCreateInfo computeStage{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "compMain" };

        computePipeline = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{ .stage = computeStage, .layout = computePipelineLayout });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createClusterCommandPool()
{
    try
    {
        computeCommandPool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.computeFamily.value() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create cluster command pool: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createClusterCommandBuffers()
{
    try
    {
        computeCommandBuffers = vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
            .commandPool = *computeCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create cluster command buffers: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createComputeSyncObjects()
{
    try
    {
        computeCompleteSemaphores.clear();
        computeCompleteSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            computeCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create compute sync objects: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createSyncObjects()
{
    try
    {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();

        const auto count = static_cast<uint32_t>(swapChainImages.size());
        presentCompleteSemaphores.reserve(count);
        renderFinishedSemaphores.reserve(count);
        inFlightFences.reserve(count);

        vk::SemaphoreCreateInfo semaphoreInfo{};
        for (uint32_t i = 0; i < count; i++)
        {
            presentCompleteSemaphores.emplace_back(device, semaphoreInfo);
            renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
        }

        vk::FenceCreateInfo fenceInfo{ .flags = vk::FenceCreateFlagBits::eSignaled };
        for (uint32_t i = 0; i < count; i++)
        {
            inFlightFences.emplace_back(device, fenceInfo);
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create sync objects: " << e.what() << std::endl;
        return false;
    }
}

bool ClusteredRenderer::createCommandBuffers()
{
    try
    {
        commandBuffers.clear();
        commandBuffers.reserve(swapChainImages.size());

        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(swapChainImages.size())
        };

        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
        computeCommandBuffers.clear();
        computeCommandBuffers.reserve(swapChainImages.size());
        vk::CommandBufferAllocateInfo computeAllocInfo{
            .commandPool = *computeCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(swapChainImages.size())
        };
        computeCommandBuffers = vk::raii::CommandBuffers(device, computeAllocInfo);
        uiFrameBuffers.resize(swapChainImages.size());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create command buffers: " << e.what() << std::endl;
        return false;
    }
}

void ClusteredRenderer::updateClusterStats()
{
    auto cmdBuffer = beginSingleTimeCommands();
    vk::DeviceSize copySize = static_cast<vk::DeviceSize>(getTotalClusters()) * sizeof(uint32_t) * 2;
    cmdBuffer->copyBuffer(*lightGridBuffer, *lightGridReadbackBuffer, {vk::BufferCopy{0, 0, copySize}});
    endSingleTimeCommands(*cmdBuffer);

    auto* gridData = static_cast<uint32_t*>(lightGridReadbackMapped);
    uint64_t totalLights = 0;
    uint32_t nonEmptyClusters = 0;
    const uint32_t totalClusters = getTotalClusters();
    for (uint32_t i = 0; i < totalClusters; ++i)
    {
        uint32_t count = gridData[i * 2 + 1];
        totalLights += count;
        if (count > 0) nonEmptyClusters++;
    }
    avgLightsPerCluster = nonEmptyClusters > 0 ? static_cast<uint32_t>(totalLights / nonEmptyClusters) : 0;
}

void ClusteredRenderer::updateClusterBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(
            glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            CLUSTER_Z_NEAR,
            CLUSTER_Z_FAR),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position,
        .nearZ = CLUSTER_Z_NEAR
    };
    sceneUbo.projection[1][1] *= -1.0f;
    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    static auto lastFrame = std::chrono::high_resolution_clock::now();
    const auto now = std::chrono::high_resolution_clock::now();
    frameMs = std::chrono::duration<float, std::milli>(now - lastFrame).count();
    lastFrame = now;
    fps = (frameMs > 0.0f) ? (1000.0f / frameMs) : 0.0f;

    const float tileX = 2.0f / static_cast<float>(clusterX);
    const float tileY = 2.0f / static_cast<float>(clusterY);

    float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
    float fovY = glm::radians(camera.Zoom);
    float tanHalfFovY = std::tan(fovY * 0.5f);
    float tanHalfFovX = tanHalfFovY * aspect;

    ClusterParamsUBO params{};
    params.clusterX = clusterX;
    params.clusterY = clusterY;
    params.clusterZ = clusterZ;
    params.totalClusters = getTotalClusters();
    params.tileSize = glm::vec3(tileX, tileY, 0.0f);
    params.cameraPos = camera.Position;
    params.farZ = CLUSTER_Z_FAR;

    float logFn = std::log(CLUSTER_Z_FAR / CLUSTER_Z_NEAR);
    params.zCount = static_cast<float>(clusterZ);
    params.zMin = CLUSTER_Z_NEAR;
    params.zMax = CLUSTER_Z_FAR;
    std::memcpy(clusterParamsResources.BuffersMapped[frameIndex], &params, sizeof(params));

    static auto animStart = std::chrono::high_resolution_clock::now();
    const float animTime = std::chrono::duration<float>(now - animStart).count();

    for (size_t i = 0; i < sceneLights.size(); ++i)
    {
        auto& light = sceneLights[i];
        float phase = animTime * 0.5f + static_cast<float>(i) * 0.1f;
        light.position.x = std::sin(phase + static_cast<float>(i) * 0.3f) * 15.0f;
        light.position.y = 0.5f + std::abs(std::sin(phase * 0.7f + static_cast<float>(i) * 0.2f)) * 6.0f;
        light.position.z = std::cos(phase + static_cast<float>(i) * 0.3f) * 15.0f;
    }
    std::memcpy(lightBufferResources.BuffersMapped[frameIndex], sceneLights.data(), sizeof(PointLight) * sceneLights.size());

    GroundUBO groundUbo{};
    groundUbo.model = glm::mat4(1.0f);
    groundUbo.model = glm::translate(groundUbo.model, glm::vec3(0.0f, -0.01f, 0.0f));
    groundUbo.color = glm::vec4(0.15f, 0.15f, 0.18f, 1.0f);
    std::memcpy(groundUboResources.BuffersMapped[frameIndex], &groundUbo, sizeof(groundUbo));
}

void ClusteredRenderer::recordComputeCommandBuffer()
{
    auto& commandBuffer = computeCommandBuffers[currentFrame];
    commandBuffer.begin({});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *computeDescriptorSets[currentFrame], nullptr);

    uint32_t dispatchX = (clusterX + 7u) / 8u;
    uint32_t dispatchY = (clusterY + 7u) / 8u;
    uint32_t dispatchZ = (clusterZ + 7u) / 8u;
    commandBuffer.dispatch(dispatchX, dispatchY, dispatchZ);

    commandBuffer.end();
}

bool ClusteredRenderer::initUI()
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
    std::memcpy(mapped, pixels, static_cast<size_t>(uploadSize));
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

void ClusteredRenderer::updateClusteredUI()
{
    ImGui::Begin("Clustered Forward Shading", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Enable Clustered Shading", &clusteredShadingEnabled);

    ImGui::Separator();
    ImGui::Text("Cluster Resolution");
    ImGui::SliderInt("Cluster X", reinterpret_cast<int*>(&clusterX), 4, 32);
    ImGui::SliderInt("Cluster Y", reinterpret_cast<int*>(&clusterY), 4, 18);
    ImGui::SliderInt("Cluster Z", reinterpret_cast<int*>(&clusterZ), 4, 64);

    ImGui::Separator();
    ImGui::Text("Scene Statistics");
    ImGui::Text("Lights: %u", lightCount);
    ImGui::Text("Total Clusters: %u", getTotalClusters());
    ImGui::Text("Lights/Cluster: %u (avg)", avgLightsPerCluster);
    ImGui::Text("Max Lights/Cluster: %u", MAX_LIGHTS_PER_CLUSTER);

    ImGui::Separator();
    ImGui::Text("Performance");
    ImGui::Text("Frame: %.3f ms", frameMs);
    ImGui::Text("FPS: %.1f", fps);

    ImGui::End();
}

void ClusteredRenderer::updateUIFrame()
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
    updateClusteredUI();
    ImGui::Render();
}

void ClusteredRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
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
        globalVertexOffset += cmdList->IdxBuffer.Size;
    }
}

void ClusteredRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

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

    const vk::ClearValue clearColor = vk::ClearColorValue(0.02f, 0.02f, 0.03f, 1.0f);
    const vk::RenderingAttachmentInfo attachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };
    const vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    const vk::RenderingAttachmentInfo depthAttachmentInfo{
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
        .pColorAttachments = &attachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *clusteredPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *clusteredPipelineLayout, 0, *clusteredDescriptorSets[currentFrame], nullptr);

    commandBuffer.bindVertexBuffers(0, *planeMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*planeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(planeMesh.indices)::value_type>::value);
    commandBuffer.drawIndexed(static_cast<uint32_t>(planeMesh.indices.size()), 1, 0, 0, 0);

    commandBuffer.bindVertexBuffers(0, *sphereMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*sphereMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(sphereMesh.indices)::value_type>::value);
    commandBuffer.drawIndexed(static_cast<uint32_t>(sphereMesh.indices.size()), lightCount, 0, 0, 0);

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

void ClusteredRenderer::render()
{
    uint32_t imageIndex = 0;
    auto [result, acquiredIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[imageIndex], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }
    imageIndex = acquiredIndex;

    const auto fenceResult = device.waitForFences(*inFlightFences[imageIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to wait for fence!");
    }

    device.resetFences(*inFlightFences[imageIndex]);
    commandBuffers[imageIndex].reset();
    computeCommandBuffers[imageIndex].reset();

    updateClusterBuffers(imageIndex);
    updateUIFrame();

    if (clusteredShadingEnabled)
    {
        device.resetFences(*computeFence);

        recordComputeCommandBuffer();
        vk::SubmitInfo cullSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*computeCommandBuffers[imageIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*computeCompleteSemaphores[imageIndex]
        };
        computeQueue.submit(cullSubmitInfo, *computeFence);

        (void)device.waitForFences(*computeFence, vk::True, UINT64_MAX);
        updateClusterStats();

        vk::Semaphore waitSemaphores[] = { *presentCompleteSemaphores[imageIndex], *computeCompleteSemaphores[imageIndex] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eComputeShader };
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 2,
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[imageIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };
        graphicsQueue.submit(submitInfo, *inFlightFences[imageIndex]);
    }
    else
    {
        recordCommandBuffer(imageIndex);
        const vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*presentCompleteSemaphores[imageIndex],
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[imageIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };
        graphicsQueue.submit(submitInfo, *inFlightFences[imageIndex]);
    }

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

void ClusteredRenderer::cleanup()
{
    shutdownUI();

    if (lightGridMapped != nullptr)
    {
        lightGridMemory.unmapMemory();
        lightGridMapped = nullptr;
    }
    if (lightIndexMapped != nullptr)
    {
        lightIndexMemory.unmapMemory();
        lightIndexMapped = nullptr;
    }
    if (lightGridReadbackMapped != nullptr)
    {
        lightGridReadbackMemory.unmapMemory();
        lightGridReadbackMapped = nullptr;
    }
}

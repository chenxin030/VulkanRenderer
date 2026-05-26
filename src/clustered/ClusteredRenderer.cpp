#include "ClusteredRenderer.h"

#include <Base/Mesh.h>
#include <Base/VulkanBase_UI.h>

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
    bool ok = VulkanBase::initVulkan("VulkanRenderer - 12_clustered");
    if (ok)
    {
        swapChainImageCount = static_cast<uint32_t>(swapChainImages.size());
    }
    return ok;
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
    if (!createClusterCommandBuffers()) return false;
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
        createUniformBuffers(sceneUboResources, sizeof(SceneUBO), swapChainImageCount);
        createUniformBuffers(clusterParamsResources, sizeof(ClusterParamsUBO), swapChainImageCount);
        createUniformBuffers(groundUboResources, sizeof(GroundUBO), swapChainImageCount);

        createStorageBuffers(lightBufferResources, sizeof(PointLight) * MAX_LIGHTS, vk::BufferUsageFlagBits::eStorageBuffer, swapChainImageCount);

        const uint32_t totalClusters = getTotalClusters();
        allocatedTotalClusters = totalClusters;

        createBuffer(
            totalClusters * sizeof(uint32_t) * 2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
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

        computeFences.clear();
        computeFences.reserve(swapChainImageCount);
        for (uint32_t i = 0; i < swapChainImageCount; ++i)
        {
            computeFences.emplace_back(device, vk::FenceCreateInfo{});
        }

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
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = swapChainImageCount * 3u },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = swapChainImageCount * 4u }
        };

        clusteredDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = swapChainImageCount * 2u,
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
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImageCount, *clusteredDescriptorSetLayout);
    clusteredDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *clusteredDescriptorPool,
        .descriptorSetCount = swapChainImageCount,
        .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < swapChainImageCount; ++i)
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
            .vertexAttributeDescriptionCount = 2,
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
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = swapChainImageCount * 3u },
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = swapChainImageCount * 2u }
        };

        computeDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = swapChainImageCount,
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
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImageCount, *computeDescriptorSetLayout);
    computeDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *computeDescriptorPool,
        .descriptorSetCount = swapChainImageCount,
        .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < swapChainImageCount; ++i)
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
            .commandBufferCount = swapChainImageCount });
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
        computeCompleteSemaphores.reserve(swapChainImageCount);
        for (uint32_t i = 0; i < swapChainImageCount; ++i)
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

void ClusteredRenderer::updateClusterStats()
{
    const uint32_t totalClusters = getTotalClusters();
    const uint32_t validClusters = std::min(totalClusters, allocatedTotalClusters);
    if (validClusters == 0)
    {
        avgLightsPerCluster = 0;
        return;
    }

    auto cmdBuffer = beginSingleTimeCommands();
    vk::DeviceSize copySize = static_cast<vk::DeviceSize>(validClusters) * sizeof(uint32_t) * 2;
    cmdBuffer->copyBuffer(*lightGridBuffer, *lightGridReadbackBuffer, {vk::BufferCopy{0, 0, copySize}});
    endSingleTimeCommands(*cmdBuffer);

    auto* gridData = static_cast<uint32_t*>(lightGridReadbackMapped);
    uint64_t totalLights = 0;
    uint32_t nonEmptyClusters = 0;
    for (uint32_t i = 0; i < validClusters; ++i)
    {
        uint32_t count = gridData[i * 2 + 1];
        totalLights += count;
        if (count > 0) nonEmptyClusters++;
    }
    avgLightsPerCluster = nonEmptyClusters > 0 ? static_cast<uint32_t>(totalLights / nonEmptyClusters) : 0;
}

void ClusteredRenderer::updateLightBuffer(uint32_t frameIndex)
{
    if (frameIndex >= lightBufferResources.BuffersMapped.size() || sceneLights.empty())
    {
        return;
    }

    std::memcpy(lightBufferResources.BuffersMapped[frameIndex], sceneLights.data(), sizeof(PointLight) * sceneLights.size());
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

    ClusterParamsUBO params{};
    params.clusterX = clusterX;
    params.clusterY = clusterY;
    params.clusterZ = clusterZ;
    params.totalClusters = getTotalClusters();
    params.tileSize = glm::vec3(tileX, tileY, 0.0f);
    params.cameraPos = camera.Position;
    params.farZ = CLUSTER_Z_FAR;
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
    updateLightBuffer(frameIndex);

    GroundUBO groundUbo{};
    groundUbo.model = glm::mat4(1.0f);
    groundUbo.model = glm::translate(groundUbo.model, glm::vec3(0.0f, -0.01f, 0.0f));
    groundUbo.color = glm::vec4(0.15f, 0.15f, 0.18f, 1.0f);
    std::memcpy(groundUboResources.BuffersMapped[frameIndex], &groundUbo, sizeof(groundUbo));
}

void ClusteredRenderer::recordComputeCommandBuffer(uint32_t frameIndex)
{
    auto& commandBuffer = computeCommandBuffers[frameIndex];
    commandBuffer.begin({});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *computeDescriptorSets[frameIndex], nullptr);

    uint32_t dispatchX = (clusterX + 7u) / 8u;
    uint32_t dispatchY = (clusterY + 7u) / 8u;
    uint32_t dispatchZ = (clusterZ + 7u) / 8u;
    commandBuffer.dispatch(dispatchX, dispatchY, dispatchZ);

    commandBuffer.end();
}

bool ClusteredRenderer::initUI()
{
    return initVulkanUI();
}

void ClusteredRenderer::updateUIPanel()
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
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *clusteredPipelineLayout, 0, *clusteredDescriptorSets[imageIndex], nullptr);

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
    recordUICmdBuffer(commandBuffer, imageIndex);
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
    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    updateClusterBuffers(imageIndex);
    updateUIFrame();

    if (clusteredShadingEnabled)
    {
        // device.waitForFences(*computeFences[imageIndex], vk::True, UINT64_MAX);
        // device.resetFences(*computeFences[imageIndex]);
        computeCommandBuffers[imageIndex].reset();
        recordComputeCommandBuffer(imageIndex);

        vk::SubmitInfo cullSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*computeCommandBuffers[imageIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*computeCompleteSemaphores[imageIndex]
        };
        computeQueue.submit(cullSubmitInfo, nullptr);

        updateClusterStats();

        device.resetFences(*inFlightFences[currentFrame]);
        commandBuffers[currentFrame].reset();
        recordCommandBuffer(imageIndex);

        vk::Semaphore waitSemaphores[] = { *presentCompleteSemaphores[currentFrame], *computeCompleteSemaphores[imageIndex] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader };
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 2,
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
        };
        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);
    }
    else
    {
        device.resetFences(*inFlightFences[currentFrame]);
        commandBuffers[currentFrame].reset();

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
    }

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };

    try
    {
        result = presentQueue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError&)
    {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
        return;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void ClusteredRenderer::recreateSwapChain()
{
    VulkanBase::recreateSwapChain();

    swapChainImageCount = static_cast<uint32_t>(swapChainImages.size());

    commandBuffers = vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = swapChainImageCount });

    computeFences.clear();
    for (uint32_t i = 0; i < swapChainImageCount; ++i)
    {
        computeFences.emplace_back(device, vk::FenceCreateInfo{});
    }
    computeCompleteSemaphores.clear();
    for (uint32_t i = 0; i < swapChainImageCount; ++i)
    {
        computeCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    computeCommandBuffers = vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
        .commandPool = *computeCommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = swapChainImageCount });

    sceneUboResources.clear();
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO), swapChainImageCount);
    clusterParamsResources.clear();
    createUniformBuffers(clusterParamsResources, sizeof(ClusterParamsUBO), swapChainImageCount);
    groundUboResources.clear();
    createUniformBuffers(groundUboResources, sizeof(GroundUBO), swapChainImageCount);
    lightBufferResources.clear();
    createStorageBuffers(lightBufferResources, sizeof(PointLight) * MAX_LIGHTS, vk::BufferUsageFlagBits::eStorageBuffer, swapChainImageCount);

    // Release old descriptor sets before recreating pools to avoid invalid frees.
    clusteredDescriptorSets = vk::raii::DescriptorSets(nullptr);
    computeDescriptorSets = vk::raii::DescriptorSets(nullptr);
    clusteredDescriptorPool = vk::raii::DescriptorPool(nullptr);
    computeDescriptorPool = vk::raii::DescriptorPool(nullptr);

    std::array<vk::DescriptorPoolSize, 2> clusteredPoolSizes = {{
        { vk::DescriptorType::eUniformBuffer, swapChainImageCount * 3u },
        { vk::DescriptorType::eStorageBuffer, swapChainImageCount * 4u }
    }};
    clusteredDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = swapChainImageCount * 2u,
        .poolSizeCount = 2u,
        .pPoolSizes = clusteredPoolSizes.data()
    });
    createClusteredDescriptorSets();

    std::array<vk::DescriptorPoolSize, 2> computePoolSizes = {{
        { vk::DescriptorType::eStorageBuffer, swapChainImageCount * 3u },
        { vk::DescriptorType::eUniformBuffer, swapChainImageCount * 2u }
    }};
    computeDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = swapChainImageCount,
        .poolSizeCount = 2u,
        .pPoolSizes = computePoolSizes.data()
    });
    createComputeDescriptorSets();
    uiFrameBuffers.resize(swapChainImageCount);
}

void ClusteredRenderer::cleanup()
{
    device.waitIdle();
    shutdownVulkanUI();

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

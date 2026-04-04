#include "CullingRenderer.h"

#include <Base/Mesh.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <random>

static_assert(sizeof(CullingRenderer::CullingParamsUBO) % 16 == 0, "CullingParamsUBO must be 16-byte aligned");

void CullingRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool CullingRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 0.0f, 40.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 7_culling");
}

bool CullingRenderer::prepareResource()
{
    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    buildStressScene();

    if (!createCullingBuffers()) return false;
    if (!createCullingDescriptorSetLayouts()) return false;
    if (!createCullingDescriptorPools()) return false;
    if (!createCullingDepthResources()) return false;
    if (!createCullingHiZResources()) return false;
    if (!createCullingHiZDescriptorSetLayout()) return false;
    if (!createCullingHiZDescriptorPool()) return false;
    createCullingDescriptorSets();
    createCullingHiZDescriptorSets();
    if (!createCullingPipelines()) return false;
    if (!createCullingHiZPipeline()) return false;

    if (!createCullingCommandPool()) return false;
    if (!createCullingCommandBuffers()) return false;
    if (!createCullingSyncObjects()) return false;

    if (!initUI()) return false;

    return true;
}

void CullingRenderer::buildStressScene()
{
    sceneInstances.clear();

    // Dense city-like grid with jittered positions and varied heights.
    constexpr int gridX = 180;
    constexpr int gridZ = 180;
    constexpr float spacing = 2.4f;

    std::mt19937 rng(1337u);
    std::uniform_real_distribution<float> jitter(-0.45f, 0.45f);
    std::uniform_real_distribution<float> scaleY(0.6f, 3.0f);
    std::uniform_real_distribution<float> rotY(0.0f, 6.2831853f);
    std::uniform_real_distribution<float> colorVar(0.75f, 1.15f);

    sceneInstances.reserve(static_cast<size_t>(gridX * gridZ));

    for (int z = 0; z < gridZ; ++z)
    {
        for (int x = 0; x < gridX; ++x)
        {
            const float px = (x - gridX * 0.5f) * spacing + jitter(rng);
            const float pz = (z - gridZ * 0.5f) * spacing + jitter(rng);
            const float sy = scaleY(rng);
            const float ry = rotY(rng);

            glm::mat4 model(1.0f);
            model = glm::translate(model, glm::vec3(px, sy * 0.5f - 1.25f, pz));
            model = glm::rotate(model, ry, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(0.8f, sy, 0.8f));

            float lane = ((x % 12) < 2 || (z % 12) < 2) ? 0.7f : 1.0f;
            float cv = colorVar(rng) * lane;
            glm::vec4 color(0.24f * cv, 0.62f * cv, 0.95f * cv, 1.0f);

            sceneInstances.push_back(CullingInstanceData{ model, color });
        }
    }

    totalInstanceCount = static_cast<uint32_t>(sceneInstances.size());
    visibleCountCpu = totalInstanceCount;
}

bool CullingRenderer::rebuildSwapchainDependentResources()
{
    cullingDepthTexture.textureSampler = vk::raii::Sampler(nullptr);
    cullingDepthTexture.textureImageView = vk::raii::ImageView(nullptr);
    cullingDepthTexture.textureImage = vk::raii::Image(nullptr);
    cullingDepthTexture.textureImageMemory = vk::raii::DeviceMemory(nullptr);

    cullingHiZMipViews.clear();
    cullingHiZTexture.textureSampler = vk::raii::Sampler(nullptr);
    cullingHiZTexture.textureImageView = vk::raii::ImageView(nullptr);
    cullingHiZTexture.textureImage = vk::raii::Image(nullptr);
    cullingHiZTexture.textureImageMemory = vk::raii::DeviceMemory(nullptr);

    cullingHiZDescriptorSets = vk::raii::DescriptorSets(nullptr);
    cullingHiZDescriptorPool = vk::raii::DescriptorPool(nullptr);

    cullingDepthLayout = vk::ImageLayout::eUndefined;
    cullingHiZLayout = vk::ImageLayout::eUndefined;

    if (!createCullingDepthResources()) return false;
    if (!createCullingHiZResources()) return false;
    if (!createCullingHiZDescriptorPool()) return false;
    createCullingHiZDescriptorSets();
    createCullingDescriptorSets();

    return true;
}

bool CullingRenderer::createCullingBuffers()
{
    try
    {
        createUniformBuffers(cullingGlobalUboResources, sizeof(SceneUBO));
        createStorageBuffers(cullingInstanceBufferResources, sizeof(CullingInstanceData) * totalInstanceCount);

        createStorageBuffers(
            cullingIndirectBufferResources,
            sizeof(DrawCommand),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
        createStorageBuffers(cullingVisibleBufferResources, sizeof(uint32_t) * totalInstanceCount);
        createStorageBuffers(
            cullingStatsBufferResources,
            sizeof(CullingStats),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        createUniformBuffers(cullingParamsBufferResources, sizeof(CullingParamsUBO));

        createBuffer(
            sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            cullingVisibleCountBuffer,
            cullingVisibleCountMemory);

        createBuffer(
            sizeof(CullingStats),
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            cullingStatsReadbackBuffer,
            cullingStatsReadbackMemory);
        cullingStatsReadbackMapped = cullingStatsReadbackMemory.mapMemory(0, sizeof(CullingStats));

        createBuffer(
            sizeof(uint32_t),
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            cullingVisibleReadbackBuffer,
            cullingVisibleReadbackMemory);
        cullingVisibleCountMapped = cullingVisibleReadbackMemory.mapMemory(0, sizeof(uint32_t));

        vk::QueryPoolCreateInfo queryInfo{ .queryType = vk::QueryType::eTimestamp, .queryCount = MAX_FRAMES_IN_FLIGHT * 2u };
        cullingTimestampQueryPool = vk::raii::QueryPool(device, queryInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling buffers: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingDescriptorSetLayouts()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> depthBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex }
        };
        cullingDepthDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(depthBindings.size()),
            .pBindings = depthBindings.data() });

        std::vector<vk::DescriptorSetLayoutBinding> drawBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex },
            { .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex }
        };
        cullingDrawDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(drawBindings.size()),
            .pBindings = drawBindings.data() });

        std::vector<vk::DescriptorSetLayoutBinding> cullingBindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 3, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 4, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 5, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 7, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute },
            { .binding = 8, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute }
        };
        cullingDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(cullingBindings.size()),
            .pBindings = cullingBindings.data() });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling descriptor set layouts: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingDescriptorPools()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> cullingPoolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 10u },
            { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u }
        };
        cullingDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(cullingPoolSizes.size()),
            .pPoolSizes = cullingPoolSizes.data() });

        std::vector<vk::DescriptorPoolSize> depthPoolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u }
        };
        cullingDepthDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(depthPoolSizes.size()),
            .pPoolSizes = depthPoolSizes.data() });

        std::vector<vk::DescriptorPoolSize> drawPoolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u },
            { .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4u }
        };
        cullingDrawDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT * 2u,
            .poolSizeCount = static_cast<uint32_t>(drawPoolSizes.size()),
            .pPoolSizes = drawPoolSizes.data() });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling descriptor pools: " << e.what() << std::endl;
        return false;
    }
}

void CullingRenderer::createCullingDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> depthLayouts(MAX_FRAMES_IN_FLIGHT, *cullingDepthDescriptorSetLayout);
    cullingDepthDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *cullingDepthDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(depthLayouts.size()),
        .pSetLayouts = depthLayouts.data() });

    std::vector<vk::DescriptorSetLayout> cullingLayouts(MAX_FRAMES_IN_FLIGHT, *cullingDescriptorSetLayout);
    cullingDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *cullingDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(cullingLayouts.size()),
        .pSetLayouts = cullingLayouts.data() });

    std::vector<vk::DescriptorSetLayout> drawLayouts(MAX_FRAMES_IN_FLIGHT, *cullingDrawDescriptorSetLayout);
    cullingDrawDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *cullingDrawDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(drawLayouts.size()),
        .pSetLayouts = drawLayouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneInfo{ .buffer = *cullingGlobalUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo instanceInfo{ .buffer = *cullingInstanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(CullingInstanceData) * totalInstanceCount };
        vk::DescriptorBufferInfo drawInfo{ .buffer = *cullingIndirectBufferResources.Buffers[i], .offset = 0, .range = sizeof(DrawCommand) };
        vk::DescriptorBufferInfo visibleInfo{ .buffer = *cullingVisibleBufferResources.Buffers[i], .offset = 0, .range = sizeof(uint32_t) * totalInstanceCount };
        vk::DescriptorBufferInfo statsInfo{ .buffer = *cullingStatsBufferResources.Buffers[i], .offset = 0, .range = sizeof(CullingStats) };
        vk::DescriptorBufferInfo visibleCountInfo{ .buffer = cullingVisibleCountBuffer, .offset = 0, .range = sizeof(uint32_t) };
        vk::DescriptorBufferInfo paramsInfo{ .buffer = *cullingParamsBufferResources.Buffers[i], .offset = 0, .range = sizeof(CullingParamsUBO) };

        vk::DescriptorImageInfo hiZInfo{ .sampler = cullingHiZTexture.textureSampler, .imageView = cullingHiZTexture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> depthWrites = {
            { .dstSet = *cullingDepthDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
            { .dstSet = *cullingDepthDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceInfo }
        };
        device.updateDescriptorSets(depthWrites, nullptr);

        std::vector<vk::WriteDescriptorSet> cullingWrites = {
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &drawInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &statsInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleCountInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hiZInfo },
            { .dstSet = *cullingDescriptorSets[i], .dstBinding = 8, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hiZInfo }
        };
        device.updateDescriptorSets(cullingWrites, nullptr);

        std::vector<vk::WriteDescriptorSet> drawWrites = {
            { .dstSet = *cullingDrawDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneInfo },
            { .dstSet = *cullingDrawDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceInfo },
            { .dstSet = *cullingDrawDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleInfo }
        };
        device.updateDescriptorSets(drawWrites, nullptr);
    }
}

bool CullingRenderer::createCullingDepthResources()
{
    try
    {
        cullingDepthExtent = swapChainExtent;
        const vk::Format depthFormat = findDepthFormat();

        createImage(
            cullingDepthExtent.width,
            cullingDepthExtent.height,
            1,
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            cullingDepthTexture);

        cullingDepthTexture.textureImageView = createImageView(cullingDepthTexture.textureImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);

        cullingDepthTexture.textureSampler = vk::raii::Sampler(device, vk::SamplerCreateInfo{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .maxAnisotropy = 1.0f,
            .compareOp = vk::CompareOp::eAlways,
            .maxLod = 0.0f });

        cullingDepthLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling depth resources: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingHiZResources()
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
                .subresourceRange = { vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1 }
            };
            cullingHiZMipViews.emplace_back(device, viewInfo);
        }

        cullingHiZTexture.textureSampler = vk::raii::Sampler(device, vk::SamplerCreateInfo{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .maxAnisotropy = 1.0f,
            .maxLod = static_cast<float>(cullingHiZMipCount - 1) });

        cullingHiZLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z resources: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingHiZDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}
        };
        cullingHiZDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingHiZDescriptorPool()
{
    try
    {
        const uint32_t setCount = MAX_FRAMES_IN_FLIGHT * cullingHiZMipCount;
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eStorageImage, .descriptorCount = setCount * 2u},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = setCount * 2u}
        };
        cullingHiZDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = setCount,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void CullingRenderer::createCullingHiZDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT * cullingHiZMipCount, *cullingHiZDescriptorSetLayout);
    cullingHiZDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *cullingHiZDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data() });

    updateCullingHiZDescriptorSets();
}

void CullingRenderer::updateCullingHiZDescriptorSets()
{
    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame)
    {
        for (uint32_t mip = 0; mip < cullingHiZMipCount; ++mip)
        {
            const uint32_t setIndex = frame * cullingHiZMipCount + mip;
            vk::DescriptorImageInfo srcInfo{};
            if (mip == 0)
            {
                srcInfo = { .sampler = cullingDepthTexture.textureSampler, .imageView = cullingDepthTexture.textureImageView, .imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal };
            }
            else
            {
                srcInfo = { .sampler = cullingHiZTexture.textureSampler, .imageView = cullingHiZMipViews[mip - 1], .imageLayout = vk::ImageLayout::eGeneral };
            }

            vk::DescriptorImageInfo dstInfo{ .sampler = nullptr, .imageView = cullingHiZMipViews[mip], .imageLayout = vk::ImageLayout::eGeneral };
            std::vector<vk::WriteDescriptorSet> writes = {
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dstInfo},
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dstInfo},
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &srcInfo},
                {.dstSet = *cullingHiZDescriptorSets[setIndex], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &srcInfo}
            };
            device.updateDescriptorSets(writes, nullptr);
        }
    }
}

bool CullingRenderer::createCullingPipelines()
{
    try
    {
        cullingPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*cullingDescriptorSetLayout });

        vk::raii::ShaderModule cullingModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_comp.spv"));
        vk::PipelineShaderStageCreateInfo cullingStage{ .stage = vk::ShaderStageFlagBits::eCompute, .module = cullingModule, .pName = "compMain" };
        cullingPipeline = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{ .stage = cullingStage, .layout = cullingPipelineLayout });

        cullingDepthPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*cullingDepthDescriptorSetLayout });

        vk::raii::ShaderModule depthModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_depth.spv"));
        vk::PipelineShaderStageCreateInfo depthStage{ .stage = vk::ShaderStageFlagBits::eVertex, .module = depthModule, .pName = "vertMain" };

        const auto binding = Vertex::getBindingDescription();
        const auto posOnly = Vertex::getPositionOnlyAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo depthVi{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(posOnly.size()),
            .pVertexAttributeDescriptions = posOnly.data() };

        vk::PipelineInputAssemblyStateCreateInfo ia{ .topology = vk::PrimitiveTopology::eTriangleList };
        vk::PipelineViewportStateCreateInfo vp{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rs{
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f };
        vk::PipelineMultisampleStateCreateInfo ms{ .rasterizationSamples = vk::SampleCountFlagBits::e1 };
        vk::PipelineDepthStencilStateCreateInfo ds{ .depthTestEnable = vk::True, .depthWriteEnable = vk::True, .depthCompareOp = vk::CompareOp::eLessOrEqual };
        std::vector dyn = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynInfo{ .dynamicStateCount = static_cast<uint32_t>(dyn.size()), .pDynamicStates = dyn.data() };
        vk::PipelineColorBlendStateCreateInfo cb{ .attachmentCount = 0 };

        vk::PipelineShaderStageCreateInfo depthStages[] = { depthStage };
        const vk::Format depthFormat = findDepthFormat();
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> depthChain = {
            {
                .stageCount = 1,
                .pStages = depthStages,
                .pVertexInputState = &depthVi,
                .pInputAssemblyState = &ia,
                .pViewportState = &vp,
                .pRasterizationState = &rs,
                .pMultisampleState = &ms,
                .pDepthStencilState = &ds,
                .pColorBlendState = &cb,
                .pDynamicState = &dynInfo,
                .layout = cullingDepthPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 0,
                .pColorAttachmentFormats = nullptr,
                .depthAttachmentFormat = depthFormat
            }
        };
        cullingDepthPipeline = vk::raii::Pipeline(device, nullptr, depthChain.get<vk::GraphicsPipelineCreateInfo>());

        cullingDrawPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*cullingDrawDescriptorSetLayout });

        vk::raii::ShaderModule drawModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_draw.spv"));
        vk::PipelineShaderStageCreateInfo drawStages[] = {
            { .stage = vk::ShaderStageFlagBits::eVertex, .module = drawModule, .pName = "vertMain" },
            { .stage = vk::ShaderStageFlagBits::eFragment, .module = drawModule, .pName = "fragMain" }
        };

        const auto pn = Vertex::getPositionNormalAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo drawVi{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(pn.size()),
            .pVertexAttributeDescriptions = pn.data() };

        vk::PipelineColorBlendAttachmentState cba{ .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo drawCb{ .attachmentCount = 1, .pAttachments = &cba };

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> drawChain = {
            {
                .stageCount = 2,
                .pStages = drawStages,
                .pVertexInputState = &drawVi,
                .pInputAssemblyState = &ia,
                .pViewportState = &vp,
                .pRasterizationState = &rs,
                .pMultisampleState = &ms,
                .pDepthStencilState = &ds,
                .pColorBlendState = &drawCb,
                .pDynamicState = &dynInfo,
                .layout = cullingDrawPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };
        cullingDrawPipeline = vk::raii::Pipeline(device, nullptr, drawChain.get<vk::GraphicsPipelineCreateInfo>());

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling pipelines: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingHiZPipeline()
{
    try
    {
        cullingHiZPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*cullingHiZDescriptorSetLayout });

        vk::raii::ShaderModule module = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "culling_hiz_build.spv"));
        vk::PipelineShaderStageCreateInfo stage{ .stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "compMain" };
        cullingHiZPipeline = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{ .stage = stage, .layout = cullingHiZPipelineLayout });

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling Hi-Z pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingCommandPool()
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
        std::cerr << "Failed to create culling command pool: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingCommandBuffers()
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
        std::cerr << "Failed to create culling command buffers: " << e.what() << std::endl;
        return false;
    }
}

bool CullingRenderer::createCullingSyncObjects()
{
    try
    {
        cullingCompleteSemaphores.clear();
        cullingCompleteSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            cullingCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create culling sync objects: " << e.what() << std::endl;
        return false;
    }
}

void CullingRenderer::extractFrustumPlanes(const glm::mat4& m, glm::vec4* out)
{
    out[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]);
    out[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]);
    out[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]);
    out[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]);
    out[4] = glm::vec4(m[0][2], m[1][2], m[2][2], m[3][2]);
    out[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]);

    for (int i = 0; i < 6; ++i)
    {
        const float len = glm::length(glm::vec3(out[i]));
        if (len > 0.0f) out[i] /= len;
    }
}

void CullingRenderer::updateCullingBuffers(uint32_t frameIndex)
{
    SceneUBO sceneUbo{
        .projection = glm::perspective(
            glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f,
            600.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1.0f;
    std::memcpy(cullingGlobalUboResources.BuffersMapped[frameIndex], &sceneUbo, sizeof(sceneUbo));

    static auto lastFrame = std::chrono::high_resolution_clock::now();
    const auto now = std::chrono::high_resolution_clock::now();
    frameMs = std::chrono::duration<float, std::milli>(now - lastFrame).count();
    lastFrame = now;

    if (!sceneInstances.empty())
    {
        std::memcpy(cullingInstanceBufferResources.BuffersMapped[frameIndex], sceneInstances.data(), sizeof(CullingInstanceData) * sceneInstances.size());
    }

    DrawCommand drawCmd{};
    drawCmd.indexCount = static_cast<uint32_t>(cubeMesh.indices.size());
    drawCmd.instanceCount = cullingEnabled ? 0u : totalInstanceCount;
    drawCmd.firstIndex = 0;
    drawCmd.vertexOffset = 0;
    drawCmd.firstInstance = 0;
    std::memcpy(cullingIndirectBufferResources.BuffersMapped[frameIndex], &drawCmd, sizeof(drawCmd));

    CullingParamsUBO params{};
    extractFrustumPlanes(sceneUbo.projection * sceneUbo.view, params.frustumPlanes);
    params.aabbMin = glm::vec4(-0.5f, -0.5f, -0.5f, 0.0f);
    params.aabbMax = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
    params.hiZInfo = glm::vec4(
        static_cast<float>(cullingDepthExtent.width),
        static_cast<float>(cullingDepthExtent.height),
        static_cast<float>(cullingHiZMipCount),
        0.0015f);
    params.totalInstances = totalInstanceCount;
    params.useCulling = cullingEnabled ? 1u : 0u;
    std::memcpy(cullingParamsBufferResources.BuffersMapped[frameIndex], &params, sizeof(params));
}

void CullingRenderer::recordCullingHiZ(vk::raii::CommandBuffer& commandBuffer)
{
    vk::ImageMemoryBarrier2 hiZInitBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = {},
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = cullingHiZLayout,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cullingHiZTexture.textureImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, cullingHiZMipCount, 0, 1 }
    };
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hiZInitBarrier });
    cullingHiZLayout = vk::ImageLayout::eGeneral;

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *cullingHiZPipeline);
    for (uint32_t mip = 0; mip < cullingHiZMipCount; ++mip)
    {
        const uint32_t setIndex = currentFrame * cullingHiZMipCount + mip;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *cullingHiZPipelineLayout, 0, *cullingHiZDescriptorSets[setIndex], nullptr);

        const uint32_t mipWidth = std::max(1u, cullingDepthExtent.width >> mip);
        const uint32_t mipHeight = std::max(1u, cullingDepthExtent.height >> mip);
        commandBuffer.dispatch((mipWidth + 7u) / 8u, (mipHeight + 7u) / 8u, 1);

        vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
        };
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{ .memoryBarrierCount = 1, .pMemoryBarriers = &barrier });
    }

    vk::ImageMemoryBarrier2 hiZReadBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = cullingHiZTexture.textureImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, cullingHiZMipCount, 0, 1 }
    };
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hiZReadBarrier });
    cullingHiZLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

void CullingRenderer::recordCullingCommandBuffer(uint32_t)
{
    auto& commandBuffer = computeCommandBuffers[currentFrame];
    commandBuffer.begin({});

    commandBuffer.resetQueryPool(*cullingTimestampQueryPool, currentFrame * 2u, 2u);
    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *cullingTimestampQueryPool, currentFrame * 2u);

    vk::ImageMemoryBarrier2 toDepthAttach{
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
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDepthAttach });
    cullingDepthLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    const vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    const vk::RenderingAttachmentInfo depthAttachment{
        .imageView = cullingDepthTexture.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearDepth
    };
    const vk::RenderingInfo depthRenderingInfo{
        .renderArea = { .offset = { 0, 0 }, .extent = cullingDepthExtent },
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &depthAttachment
    };

    commandBuffer.beginRendering(depthRenderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(cullingDepthExtent.width), static_cast<float>(cullingDepthExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), cullingDepthExtent));
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *cullingDepthPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *cullingDepthPipelineLayout, 0, *cullingDepthDescriptorSets[currentFrame], nullptr);
    commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
    commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), totalInstanceCount, 0, 0, 0);
    commandBuffer.endRendering();

    vk::ImageMemoryBarrier2 toDepthRead{
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
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDepthRead });
    cullingDepthLayout = vk::ImageLayout::eDepthReadOnlyOptimal;

    recordCullingHiZ(commandBuffer);

    commandBuffer.fillBuffer(cullingVisibleCountBuffer, 0, sizeof(uint32_t), 0u);
    vk::BufferMemoryBarrier countResetBarrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = cullingVisibleCountBuffer,
        .offset = 0,
        .size = sizeof(uint32_t)
    };
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, { countResetBarrier }, {});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *cullingPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *cullingPipelineLayout, 0, *cullingDescriptorSets[currentFrame], nullptr);
    commandBuffer.dispatch((totalInstanceCount + kWorkgroupSize - 1u) / kWorkgroupSize, 1, 1);

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
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, { statsBarrier, countBarrier }, {});

    commandBuffer.copyBuffer(*cullingStatsBufferResources.Buffers[currentFrame], *cullingStatsReadbackBuffer, vk::BufferCopy(0, 0, sizeof(CullingStats)));
    commandBuffer.copyBuffer(cullingVisibleCountBuffer, cullingVisibleReadbackBuffer, vk::BufferCopy(0, 0, sizeof(uint32_t)));

    commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *cullingTimestampQueryPool, currentFrame * 2u + 1u);
    commandBuffer.end();
}

void CullingRenderer::recordCullingDrawCommands(vk::raii::CommandBuffer& commandBuffer)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *cullingDrawPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *cullingDrawPipelineLayout, 0, *cullingDrawDescriptorSets[currentFrame], nullptr);
    commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
    commandBuffer.drawIndexedIndirect(*cullingIndirectBufferResources.Buffers[currentFrame], 0, 1, sizeof(DrawCommand));
}

bool CullingRenderer::initUI()
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

void CullingRenderer::updateCullingUI()
{
    static float uiUpdateAccum = 0.0f;
    static float displayedFrameMs = 0.0f;
    static float displayedFps = 0.0f;

    const float dt = ImGui::GetIO().DeltaTime;
    uiUpdateAccum += dt;

    if (uiUpdateAccum >= 0.3f)
    {
        displayedFrameMs = frameMs;
        displayedFps = (displayedFrameMs > 0.0f) ? (1000.0f / displayedFrameMs) : 0.0f;
        uiUpdateAccum = 0.0f;
    }

    ImGui::Begin("Culling", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("Enable Culling", &cullingEnabled);

    ImGui::Text("Instances: %u", totalInstanceCount);
    ImGui::Text("Visible: %u", visibleCountCpu);
    ImGui::Text("Frame: %.3f ms", displayedFrameMs);
    ImGui::Text("FPS: %.1f", displayedFps);
    ImGui::Text("Culling GPU: %.3f ms", cullingGpuMs);
    ImGui::End();
}

void CullingRenderer::updateUIFrame()
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
    updateCullingUI();
    ImGui::Render();
}

void CullingRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
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
        globalVertexOffset += cmdList->VtxBuffer.Size;
    }
}

void CullingRenderer::recordCommandBuffer(uint32_t imageIndex)
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
        .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordCullingDrawCommands(commandBuffer);
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

void CullingRenderer::updateCullingStats()
{
    uint64_t timestamps[2] = {};
    vk::Device deviceHandle = *device;
    const vk::Result res = deviceHandle.getQueryPoolResults(
        *cullingTimestampQueryPool,
        currentFrame * 2u,
        2u,
        sizeof(timestamps),
        timestamps,
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

    if (res == vk::Result::eSuccess)
    {
        const double timestampPeriod = physicalDevice.getProperties().limits.timestampPeriod;
        cullingGpuMs = static_cast<float>((timestamps[1] - timestamps[0]) * timestampPeriod * 1e-6);
    }

    if (cullingStatsReadbackMapped != nullptr)
    {
        const auto* stats = reinterpret_cast<const CullingStats*>(cullingStatsReadbackMapped);
        visibleCountCpu = stats->visibleCount;
        totalInstanceCount = stats->totalCount;
    }

    if (cullingVisibleCountMapped != nullptr)
    {
        visibleCountCpu = *reinterpret_cast<const uint32_t*>(cullingVisibleCountMapped);
    }
}

void CullingRenderer::render()
{
    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        if (!rebuildSwapchainDependentResources())
        {
            throw std::runtime_error("failed to rebuild culling resources after swapchain recreation");
        }
        return;
    }

    updateCullingBuffers(currentFrame);
    updateUIFrame();

    computeCommandBuffers[currentFrame].reset();
    recordCullingCommandBuffer(imageIndex);

    vk::SubmitInfo cullSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*computeCommandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*cullingCompleteSemaphores[currentFrame]
    };
    computeQueue.submit(cullSubmitInfo, nullptr);

    updateCullingStats();

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();
    recordCommandBuffer(imageIndex);

    vk::Semaphore waitSemaphores[] = { *presentCompleteSemaphores[currentFrame], *cullingCompleteSemaphores[currentFrame] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eVertexInput };
    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 2,
        .pWaitSemaphores = waitSemaphores,
        .pWaitDstStageMask = waitStages,
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
        if (!rebuildSwapchainDependentResources())
        {
            throw std::runtime_error("failed to rebuild culling resources after swapchain recreation");
        }
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void CullingRenderer::cleanup()
{
    shutdownUI();

    if (cullingVisibleCountMapped != nullptr)
    {
        cullingVisibleReadbackMemory.unmapMemory();
        cullingVisibleCountMapped = nullptr;
    }
    if (cullingStatsReadbackMapped != nullptr)
    {
        cullingStatsReadbackMemory.unmapMemory();
        cullingStatsReadbackMapped = nullptr;
    }
}

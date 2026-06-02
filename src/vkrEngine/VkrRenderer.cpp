#include "VkrRenderer.h"

#include <Base/VulkanBase_UI.h>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

void VkrRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool VkrRenderer::initVulkan()
{
    // Sponza is a large atrium (~25m wide, ~15m tall). Start at a distance.
    camera = Camera(glm::vec3(0.0f, 80.0f, 25.0f));
    camera.MovementSpeed = 200.0f;
    return VulkanBase::initVulkan("VulkanRenderer - vkrEngine");
}

bool VkrRenderer::prepareResource()
{
    // Load Sponza scene 
    const std::string sponzaDir = std::string(VK_GLTF_SAMPLES_DIR) + "Models/Sponza/glTF/";
    const std::string sponzaPath = sponzaDir + "Sponza.gltf";

    std::cout << "[vkrEngine] Loading Sponza from: " << sponzaPath << std::endl;

    if (!sponzaModel.loadFromFile(sponzaPath, sponzaDir))
    {
        std::cerr << "[vkrEngine] Failed to load Sponza model!" << std::endl;
        return false;
    }
    scene.addModel(&sponzaModel, glm::mat4(1.0f), "Sponza");

    // Create GPU resources 
    if (!createModelGpuResources(sponzaModel)) return false;
    if (!createDummyWhiteTexture()) return false;
    if (!createDummyNormalTexture()) return false;

    // Create material textures
    for (auto& mat : sponzaModel.materials)
    {
        if (!createMaterialGpuResources(*mat, sponzaDir)) return false;
    }

    // UBOs 
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createUniformBuffers(paramsUboResources, sizeof(ParamsUBO));
    createUniformBuffers(csmUboResources, sizeof(CsmUBO));
    createUniformBuffers(shadowParamsUboResources, sizeof(ShadowParamsUBO));

    if (!createDescriptors()) return false;
    if (!createPipeline()) return false;
    if (!createMaterialDescriptorSets()) return false;
    generateIBLResources();
    if (!createCsmResources()) return false;

    // ================================================================
    // Phase 4: Deferred Rendering + SSAO + SSR
    // ================================================================
    createUniformBuffers(deferredSettingsUboResources, sizeof(DeferredSettingsUBO));

    if (!createGBufferResources()) return false;
    if (!createGBufferDescriptorSetLayout()) return false;
    if (!createGBufferDescriptorPool()) return false;
    createGBufferDescriptorSets();
    if (!createGBufferPipeline()) return false;

    if (!createSSAOResources()) return false;
    generateSsaoKernel();
    createSsaoNoiseTexture();
    if (!createSsaoDescriptorSetLayout()) return false;
    if (!createSsaoDescriptorPool()) return false;
    createSsaoDescriptorSets();
    if (!createSsaoPipeline()) return false;
    if (!createSsaoBlurDescriptorSetLayout()) return false;
    if (!createSsaoBlurDescriptorPool()) return false;
    createSsaoBlurDescriptorSets();
    if (!createSsaoBlurPipeline()) return false;

    if (!createSSRResources()) return false;
    if (!createSsrDescriptorSetLayout()) return false;
    if (!createSsrDescriptorPool()) return false;
    createSsrDescriptorSets();
    if (!createSsrPipeline()) return false;

    if (!createDeferredLightingDescriptorSetLayout()) return false;
    if (!createDeferredLightingDescriptorPool()) return false;
    createDeferredLightingDescriptorSets();
    if (!createDeferredLightingPipeline()) return false;

    // Update scene descriptor sets with IBL + CSM bindings 
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{
            .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo paramsBufferInfo{
            .buffer = *paramsUboResources.Buffers[i], .offset = 0, .range = sizeof(ParamsUBO) };
        vk::DescriptorBufferInfo csmBufferInfo{
            .buffer = *csmUboResources.Buffers[i], .offset = 0, .range = sizeof(CsmUBO) };
        vk::DescriptorBufferInfo shadowParamsBufferInfo{
            .buffer = *shadowParamsUboResources.Buffers[i], .offset = 0, .range = sizeof(ShadowParamsUBO) };

        vk::DescriptorImageInfo irradianceInfo{
            .sampler = irradianceCubemapData.textureSampler,
            .imageView = irradianceCubemapData.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo prefilteredInfo{
            .sampler = prefilteredEnvMapData.textureSampler,
            .imageView = prefilteredEnvMapData.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo brdfInfo{
            .sampler = brdfLutData.textureSampler,
            .imageView = brdfLutData.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo shadowMapInfo{
            .sampler = *csmSamplers[i],
            .imageView = *csmArrayViews[i],
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::array<vk::WriteDescriptorSet, 8> writes = { {
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 0,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,
             .pBufferInfo = &sceneBufferInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 1,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &irradianceInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 2,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &prefilteredInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 3,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &brdfInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 4,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,
             .pBufferInfo = &paramsBufferInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 5,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,
             .pBufferInfo = &csmBufferInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 6,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
             .pImageInfo = &shadowMapInfo },
            {.dstSet = *sceneUboResources.descriptorSets[i], .dstBinding = 7,
             .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,
             .pBufferInfo = &shadowParamsBufferInfo },
        } };
        device.updateDescriptorSets(writes, nullptr);
    }

    // GPU Profiler 
    auto props = physicalDevice.getProperties();
    gpuProfiler.init(device, static_cast<float>(props.limits.timestampPeriod));

    // UI 
    if (!initUI()) return false;

    // Create per-swapchain-image semaphores (fixes Device Lost) 
    uint32_t imageCount = static_cast<uint32_t>(swapChainImages.size());
    for (uint32_t i = 0; i < imageCount; ++i)
    {
        m_imageAcquiredSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        m_renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
    }

    std::cout << "[vkrEngine] Initialization complete." << std::endl;
    std::cout << "[vkrEngine] Scene: " << scene.totalTriangles() << " triangles, "
        << scene.totalDrawCalls() << " draw calls" << std::endl;

    // Texture load diagnostics
    uint32_t texLoaded = 0;
    for (auto& mat : sponzaModel.materials)
    {
        if (mat->gpuBaseColorTexture.imageView != vk::raii::ImageView(nullptr)) ++texLoaded;
    }
    std::cout << "[vkrEngine] Materials with textures: " << texLoaded
        << " / " << sponzaModel.materials.size() << std::endl;

    return true;
}

void VkrRenderer::cleanup()
{
    ssaoSettingsUboResources.descriptorSets.clear();
    ssaoBlurUboResources.descriptorSets.clear();
    ssrParamsUboResources.descriptorSets.clear();
    deferredSettingsUboResources.descriptorSets.clear();

    ssaoDescriptorPool = nullptr;
    ssaoBlurDescriptorPool = nullptr;
    ssrDescriptorPool = nullptr;
    deferredDescriptorPool = nullptr;

    destroyGBufferResources();
    destroySSAOResources();
    destroySSRResources();

    std::cout << "[vkrEngine] Cleanup complete." << std::endl;
}

bool VkrRenderer::createModelGpuResources(VkrModel& model)
{
    if (model.vertices.empty() || model.indices.empty())
    {
        std::cerr << "[vkrEngine] Model has no geometry: " << model.name << std::endl;
        return false;
    }

    // Vertex buffer
    {
        vk::DeviceSize bufferSize = sizeof(VkrVertex) * model.vertices.size();

        vk::raii::Buffer       stagingBuf(nullptr);
        vk::raii::DeviceMemory stagingMem(nullptr);
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuf, stagingMem);

        void* mapped = stagingMem.mapMemory(0, bufferSize);
        memcpy(mapped, model.vertices.data(), static_cast<size_t>(bufferSize));
        stagingMem.unmapMemory();

        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            model.vertexBuffer, model.vertexBufferMemory);

        copyBuffer(stagingBuf, model.vertexBuffer, bufferSize);
    }

    // Index buffer
    {
        vk::DeviceSize bufferSize = sizeof(uint32_t) * model.indices.size();

        vk::raii::Buffer       stagingBuf(nullptr);
        vk::raii::DeviceMemory stagingMem(nullptr);
        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuf, stagingMem);

        void* mapped = stagingMem.mapMemory(0, bufferSize);
        memcpy(mapped, model.indices.data(), static_cast<size_t>(bufferSize));
        stagingMem.unmapMemory();

        createBuffer(bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            model.indexBuffer, model.indexBufferMemory);

        copyBuffer(stagingBuf, model.indexBuffer, bufferSize);
    }

    return true;
}

bool VkrRenderer::createDummyWhiteTexture()
{
    constexpr uint32_t size = 4;
    uint8_t whitePixels[size] = { 255, 255, 255, 255 };

    vk::DeviceSize uploadSize = size;
    vk::raii::Buffer stagingBuf(nullptr);
    vk::raii::DeviceMemory stagingMem(nullptr);
    createBuffer(uploadSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuf, stagingMem);

    void* mapped = stagingMem.mapMemory(0, uploadSize);
    memcpy(mapped, whitePixels, size);
    stagingMem.unmapMemory();

    createImage(1, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, dummyWhiteTexture);

    transitionImageLayout(dummyWhiteTexture.textureImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    copyBufferToImage(stagingBuf, dummyWhiteTexture.textureImage, 1, 1);
    transitionImageLayout(dummyWhiteTexture.textureImage,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    dummyWhiteTexture.textureImageView = createImageView(
        dummyWhiteTexture.textureImage, vk::Format::eR8G8B8A8Unorm,
        vk::ImageAspectFlagBits::eColor, 1);

    dummyWhiteTexture.mipLevels = 1;
    return true;
}

bool VkrRenderer::createDummyNormalTexture()
{
    // Flat normal: (0.5, 0.5, 1.0) in tangent space = no perturbation
    constexpr uint32_t size = 4;
    uint8_t flatNormalPixels[size] = { 128, 128, 255, 255 };

    vk::DeviceSize uploadSize = size;
    vk::raii::Buffer stagingBuf(nullptr);
    vk::raii::DeviceMemory stagingMem(nullptr);
    createBuffer(uploadSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuf, stagingMem);

    void* mapped = stagingMem.mapMemory(0, uploadSize);
    memcpy(mapped, flatNormalPixels, size);
    stagingMem.unmapMemory();

    createImage(1, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, dummyNormalTexture);

    transitionImageLayout(dummyNormalTexture.textureImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    copyBufferToImage(stagingBuf, dummyNormalTexture.textureImage, 1, 1);
    transitionImageLayout(dummyNormalTexture.textureImage,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    dummyNormalTexture.textureImageView = createImageView(
        dummyNormalTexture.textureImage, vk::Format::eR8G8B8A8Unorm,
        vk::ImageAspectFlagBits::eColor, 1);

    dummyNormalTexture.mipLevels = 1;
    return true;
}

bool VkrRenderer::loadMaterialTexture(VkrMaterialTexture& tex, const std::string& fullPath,
    vk::Format format)
{
    if (!fs::exists(fullPath))
    {
        std::cerr << "[vkrEngine] Texture not found: " << fullPath << std::endl;
        return false;
    }

    // Load with stbi using the ABSOLUTE path (not VK_TEXTURE_DIR-prefixed)
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(fullPath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels)
    {
        std::cerr << "[vkrEngine] Failed to decode texture: " << fullPath << std::endl;
        return false;
    }

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth) * texHeight * 4;
    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    // Staging buffer
    vk::raii::Buffer stagingBuf(nullptr);
    vk::raii::DeviceMemory stagingMem(nullptr);
    createBuffer(imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuf, stagingMem);

    void* mapped = stagingMem.mapMemory(0, imageSize);
    memcpy(mapped, pixels, static_cast<size_t>(imageSize));
    stagingMem.unmapMemory();
    stbi_image_free(pixels);

    // Create image
    TextureData tempTex;
    createImage(static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), mipLevels,
        format, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, tempTex);

    // Transfer image to GPU
    transitionImageLayout(tempTex.textureImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(stagingBuf, tempTex.textureImage,
        static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    generateMipmaps(tempTex.textureImage, format,
        texWidth, texHeight, mipLevels);

    // Create image view
    tempTex.textureImageView = createImageView(tempTex.textureImage,
        format, vk::ImageAspectFlagBits::eColor, mipLevels);

    // Move to output
    tex.image = std::move(tempTex.textureImage);
    tex.imageMemory = std::move(tempTex.textureImageMemory);
    tex.imageView = std::move(tempTex.textureImageView);

    std::cout << "[vkrEngine] Loaded texture: " << fs::path(fullPath).filename().string()
        << " (" << texWidth << "x" << texHeight << ")" << std::endl;
    return true;
}

bool VkrRenderer::createMaterialGpuResources(VkrMaterial& mat, const std::string& textureBasePath)
{
    if (mat.gpuResourcesCreated) return true;

    if (!mat.baseColorTexturePath.empty())
    {
        std::string fullPath = textureBasePath + mat.baseColorTexturePath;
        loadMaterialTexture(mat.gpuBaseColorTexture, fullPath,
            vk::Format::eR8G8B8A8Srgb); // sRGB for albedo
    }

    if (!mat.normalTexturePath.empty())
    {
        std::string fullPath = textureBasePath + mat.normalTexturePath;
        loadMaterialTexture(mat.gpuNormalTexture, fullPath,
            vk::Format::eR8G8B8A8Unorm); // linear for normal maps
    }

    if (!mat.metallicRoughnessTexturePath.empty())
    {
        std::string fullPath = textureBasePath + mat.metallicRoughnessTexturePath;
        loadMaterialTexture(mat.gpuMetallicRoughnessTexture, fullPath,
            vk::Format::eR8G8B8A8Unorm); // linear for metallic-roughness
    }

    mat.gpuResourcesCreated = true;
    return true;
}

bool VkrRenderer::createDescriptors()
{
    try
    {
        // Scene descriptor set layout (Set 0) 
        // Binding 0: SceneUBO (uniform buffer)
        // Binding 1: irradiance cubemap (combined image sampler)
        // Binding 2: prefiltered env map (combined image sampler)
        // Binding 3: BRDF LUT (combined image sampler)
        // Binding 4: ParamsUBO (uniform buffer)
        // Binding 5: CsmUBO (uniform buffer)
        // Binding 6: shadow map array (combined image sampler)
        // Binding 7: ShadowParamsUBO (uniform buffer)
        {
            std::vector<vk::DescriptorSetLayoutBinding> bindings = {
                {.binding = 0,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
                {.binding = 1,
                  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 2,
                  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 3,
                  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 4,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 5,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
                {.binding = 6,
                  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 7,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
            };

            sceneDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
                vk::DescriptorSetLayoutCreateInfo{
                    .bindingCount = static_cast<uint32_t>(bindings.size()),
                    .pBindings = bindings.data() });
        }

        // Material descriptor set layout (Set 1) 
        // Binding 0: baseColorTexture (sampled image)
        // Binding 1: normalTexture (sampled image)
        // Binding 2: metallicRoughnessTexture (sampled image)
        // Binding 3: shared sampler
        // Binding 4: MaterialUBO (uniform buffer)
        {
            std::vector<vk::DescriptorSetLayoutBinding> bindings = {
                {.binding = 0,
                  .descriptorType = vk::DescriptorType::eSampledImage,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 1,
                  .descriptorType = vk::DescriptorType::eSampledImage,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 2,
                  .descriptorType = vk::DescriptorType::eSampledImage,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 3,
                  .descriptorType = vk::DescriptorType::eSampler,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 4,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
            };

            materialDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
                vk::DescriptorSetLayoutCreateInfo{
                    .bindingCount = static_cast<uint32_t>(bindings.size()),
                    .pBindings = bindings.data() });
        }

        // Descriptor Pool 
        {
            uint32_t sceneSets = MAX_FRAMES_IN_FLIGHT;
            uint32_t matSets = static_cast<uint32_t>(sponzaModel.materials.size());
            uint32_t maxSets = sceneSets + matSets;

            std::vector<vk::DescriptorPoolSize> poolSizes = {
                { vk::DescriptorType::eUniformBuffer,        sceneSets * 5 + matSets }, // sceneUbo + params + csmUbo + shadowParams + materialUbo
                { vk::DescriptorType::eSampledImage,         matSets * 3 },             // baseColor + normal + mr
                { vk::DescriptorType::eSampler,              matSets },                 // shared sampler
                { vk::DescriptorType::eCombinedImageSampler, sceneSets * 4 },           // IBL cubemaps (3) + shadow map array (1)
            };

            descriptorPool = vk::raii::DescriptorPool(device,
                vk::DescriptorPoolCreateInfo{
                    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                    .maxSets = maxSets + 1,  // FIX: extra set for cubeSetLayout[1] (irradiance pipeline)
                    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                    .pPoolSizes = poolSizes.data() });
        }

        // Allocate scene descriptor sets (one per frame) 
        {
            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *sceneDescriptorSetLayout);

            vk::DescriptorSetAllocateInfo allocInfo{
                .descriptorPool = *descriptorPool,
                .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
                .pSetLayouts = layouts.data()
            };

            sceneUboResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);
        }

        // Create shared sampler for textures 
        {
            vk::SamplerCreateInfo samplerInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eRepeat,
                .addressModeV = vk::SamplerAddressMode::eRepeat,
                .addressModeW = vk::SamplerAddressMode::eRepeat,
                .mipLodBias = 0.0f,
                .anisotropyEnable = vk::True,
                .maxAnisotropy = 4.0f,
                .minLod = 0.0f,
                .maxLod = VK_LOD_CLAMP_NONE,
            };
            materialSampler = vk::raii::Sampler(device, samplerInfo);
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkrEngine] Failed to create descriptors: " << e.what() << std::endl;
        return false;
    }
}

bool VkrRenderer::createMaterialDescriptorSets()
{
    try
    {
        uint32_t matCount = static_cast<uint32_t>(sponzaModel.materials.size());

        // Create per-material UBOs
        materialUboResources.clear();
        materialUboResources.resize(matCount);
        for (uint32_t i = 0; i < matCount; ++i)
        {
            createUniformBuffers(materialUboResources[i], sizeof(MaterialUBO), 1);
        }

        // Allocate one descriptor set per material individually
        materialDescriptorSets.clear();
        materialDescriptorSets.reserve(matCount);

        for (uint32_t i = 0; i < matCount; ++i)
        {
            auto sets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
                .descriptorPool = *descriptorPool,
                .descriptorSetCount = 1,
                .pSetLayouts = &*materialDescriptorSetLayout,
                });
            materialDescriptorSets.push_back(std::move(sets[0]));
        }

        // Write descriptors for each material
        vk::DescriptorImageInfo samplerInfo{
            .sampler = *materialSampler,
        };

        for (uint32_t i = 0; i < matCount; ++i)
        {
            auto& mat = *sponzaModel.materials[i];

            // Base color image
            vk::DescriptorImageInfo baseColorInfo{
                .imageView = (mat.gpuBaseColorTexture.imageView != vk::raii::ImageView(nullptr))
                    ? *mat.gpuBaseColorTexture.imageView
                    : *dummyWhiteTexture.textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            // Normal image
            vk::DescriptorImageInfo normalInfo{
                .imageView = (mat.gpuNormalTexture.imageView != vk::raii::ImageView(nullptr))
                    ? *mat.gpuNormalTexture.imageView
                    : *dummyNormalTexture.textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            // Metallic-roughness image
            vk::DescriptorImageInfo mrInfo{
                .imageView = (mat.gpuMetallicRoughnessTexture.imageView != vk::raii::ImageView(nullptr))
                    ? *mat.gpuMetallicRoughnessTexture.imageView
                    : *dummyWhiteTexture.textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            // Material UBO
            vk::DescriptorBufferInfo matUboInfo{
                .buffer = *materialUboResources[i].Buffers[0],
                .offset = 0,
                .range = sizeof(MaterialUBO),
            };

            std::array<vk::WriteDescriptorSet, 5> writes = { {
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i], .dstBinding = 0,
                    .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &baseColorInfo },
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i], .dstBinding = 1,
                    .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &normalInfo },
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i], .dstBinding = 2,
                    .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &mrInfo },
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i], .dstBinding = 3,
                    .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler,
                    .pImageInfo = &samplerInfo },
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i], .dstBinding = 4,
                    .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &matUboInfo },
            } };

            device.updateDescriptorSets(writes, nullptr);
        }

        // Populate initial material UBOs
        updateMaterialUBOs();

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkrEngine] Failed to create material descriptor sets: " << e.what() << std::endl;
        return false;
    }
}


// Pipeline


bool VkrRenderer::createPipeline()
{
    try
    {
        // Load shaders
        auto vertCode = readFile(std::string(VK_SHADERS_DIR) + "scene_vert.spv");
        auto fragCode = readFile(std::string(VK_SHADERS_DIR) + "scene_frag.spv");

        vk::raii::ShaderModule vertModule = createShaderModule(vertCode);
        vk::raii::ShaderModule fragModule = createShaderModule(fragCode);

        // Shader stages
        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertModule,
                .pName = "vertMain" },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragModule,
                .pName = "fragMain" },
        };

        // Vertex input — pos + normal + texCoord + tangent (stride=48)
        vk::VertexInputBindingDescription bindingDesc{ 0, sizeof(VkrVertex), vk::VertexInputRate::eVertex };
        std::array<vk::VertexInputAttributeDescription, 4> attrDescs = { {
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat,    offsetof(VkrVertex, pos)      },  // pos
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat,    offsetof(VkrVertex, normal)   },  // normal
            vk::VertexInputAttributeDescription{ 2, 0, vk::Format::eR32G32Sfloat,       offsetof(VkrVertex, texCoord) },  // texCoord
            vk::VertexInputAttributeDescription{ 3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VkrVertex, tangent)  },  // tangent
        } };

        vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDesc,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size()),
            .pVertexAttributeDescriptions = attrDescs.data(),
        };

        // Input assembly
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False,
        };

        // Viewport & Scissor
        vk::Viewport viewport{
            .x = 0.0f, .y = 0.0f,
            .width = static_cast<float>(swapChainExtent.width),
            .height = static_cast<float>(swapChainExtent.height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };
        vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = swapChainExtent };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        // Rasterization
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f,
        };

        // Multisampling
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False,
        };

        // Depth/Stencil
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False,
        };

        // Color blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR |
                              vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB |
                              vk::ColorComponentFlagBits::eA,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        // Dynamic state
        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };

        // Push constant: model matrix (mat4)
        vk::PushConstantRange pushRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset = 0,
            .size = sizeof(glm::mat4),
        };

        // Pipeline layout
        std::vector<vk::DescriptorSetLayout> setLayouts = {
            *sceneDescriptorSetLayout,
            *materialDescriptorSetLayout,
        };

        pipelineLayout = vk::raii::PipelineLayout(device,
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
                .pSetLayouts = setLayouts.data(),
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pushRange,
            });

        // Dynamic rendering: no explicit RenderPass — use VkPipelineRenderingCreateInfo
        vk::PipelineRenderingCreateInfo renderingInfo{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = findDepthFormat(),
        };

        // Graphics pipeline (with dynamic rendering)
        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .pNext = &renderingInfo,
            .stageCount = static_cast<uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *pipelineLayout,
            .renderPass = nullptr, // dynamic rendering
            .subpass = 0,
        };

        pipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkrEngine] Failed to create pipeline: " << e.what() << std::endl;
        return false;
    }
}

void VkrRenderer::render()
{
    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Read back PREVIOUS frame's GPU timestamps 
    gpuProfiler.endFrame();

    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
        throw std::runtime_error("failed to wait for fence!");

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX,
        *m_imageAcquiredSemaphores[currentFrame % m_imageAcquiredSemaphores.size()], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    vkrUpdateUIFrame(this);
    updateSceneUBO(currentFrame);
    updateCsmBuffers(currentFrame);
    updateSsaoBuffers(currentFrame);
    updateSsaoBlurBuffers(currentFrame);
    updateSsrBuffers(currentFrame);
    updateDeferredSettingsBuffer(currentFrame);

    recordCommandBuffer(imageIndex);

    // Submit (per-image semaphores) 
    const vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    uint32_t semaIdx = imageIndex % static_cast<uint32_t>(m_renderFinishedSemaphores.size());
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*m_imageAcquiredSemaphores[currentFrame % m_imageAcquiredSemaphores.size()],
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*m_renderFinishedSemaphores[semaIdx],
    };
    graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

    // Present 
    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*m_renderFinishedSemaphores[semaIdx],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex,
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

    if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    auto cpuEnd = std::chrono::high_resolution_clock::now();
    statCpuMs = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();
}

void VkrRenderer::updateSceneUBO(uint32_t frameIndex)
{
    SceneUBO ubo{};
    ubo.projection = glm::perspective(glm::radians(camera.Zoom),
        static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
        CAMERA_NEAR, CAMERA_FAR);
    ubo.projection[1][1] *= -1.0f; // Vulkan inverted Y
    ubo.view = camera.GetViewMatrix();
    ubo.invProjection = glm::inverse(ubo.projection);
    ubo.invView = glm::inverse(ubo.view);
    ubo.camPos = glm::vec4(camera.Position, CAMERA_NEAR);

    // Light direction (shared with CSM; optionally rotating)
    static float lightAngle = 0.0f;
    float deltaTime = platform ? platform->frameTimer : 0.0f;
    if (uiRotateLight) lightAngle += deltaTime * 0.15f;
    // Light from upper-front: balanced angle so horizontal & vertical surfaces both get N·L > 0
    glm::vec3 lightDirVec = glm::normalize(
        glm::vec3(0.5f + std::cos(lightAngle) * 0.3f, -0.55f, 0.3f + std::sin(lightAngle) * 0.3f));
    glm::vec3 lightColVec = scene.dirLight().color * scene.dirLight().intensity;
    ubo.lightDir = glm::vec4(lightDirVec, CAMERA_FAR);
    ubo.lightColor = glm::vec4(lightColVec, 0.0f);

    void* mapped = sceneUboResources.BuffersMapped[frameIndex];
    memcpy(mapped, &ubo, sizeof(SceneUBO));

    // Update Params UBO (exposure, gamma, lighting mode — UI-tweakable)
    ParamsUBO params{};
    params.exposure = uiExposure;
    params.gamma = uiGamma;
    params.lightingMode = static_cast<uint32_t>(uiLightingMode);
    params.enableDirLight = uiEnableDirectionalLight ? 1u : 0u;
    void* paramsMapped = paramsUboResources.BuffersMapped[frameIndex];
    memcpy(paramsMapped, &params, sizeof(ParamsUBO));
}

void VkrRenderer::updateMaterialUBOs()
{
    for (size_t i = 0; i < sponzaModel.materials.size(); ++i)
    {
        auto& mat = *sponzaModel.materials[i];
        MaterialUBO ubo{};
        ubo.baseColorFactor = mat.baseColorFactor;
        ubo.metallicFactor = mat.metallicFactor;
        ubo.roughnessFactor = mat.roughnessFactor;
        ubo.normalScale = 1.0f;
        ubo.occlusionStrength = 1.0f;
        ubo.emissiveFactor = mat.emissiveFactor;
        ubo.alphaCutoff = 0.5f;

        // Flags: bit0=hasBaseColorTex, bit1=hasNormalTex, bit2=hasMetallicRoughnessTex
        ubo.flags = 0;
        if (mat.gpuBaseColorTexture.imageView != vk::raii::ImageView(nullptr))
            ubo.flags |= 0x1u;
        if (mat.gpuNormalTexture.imageView != vk::raii::ImageView(nullptr))
            ubo.flags |= 0x2u;
        if (mat.gpuMetallicRoughnessTexture.imageView != vk::raii::ImageView(nullptr))
            ubo.flags |= 0x4u;

        void* mapped = materialUboResources[i].BuffersMapped[0];
        memcpy(mapped, &ubo, sizeof(MaterialUBO));
    }
}


// IBL Generation (Equirect → Cubemap → Irradiance → Prefiltered → BRDF LUT)


void VkrRenderer::transitionImageLayoutCmd(
    vk::raii::CommandBuffer& commandBuffer,
    vk::Image image,
    vk::ImageAspectFlags aspectMask,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    uint32_t baseMipLevel,
    uint32_t levelCount,
    uint32_t baseArrayLayer,
    uint32_t layerCount,
    vk::PipelineStageFlags srcStage,
    vk::PipelineStageFlags dstStage,
    vk::AccessFlags srcAccessMask,
    vk::AccessFlags dstAccessMask)
{
    vk::ImageMemoryBarrier barrier{
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = { aspectMask, baseMipLevel, levelCount, baseArrayLayer, layerCount },
    };
    commandBuffer.pipelineBarrier(srcStage, dstStage, {}, {}, {}, barrier);
}

void VkrRenderer::generateIBLResources()
{
    std::cout << "[vkrEngine] Generating IBL resources..." << std::endl;

    const vk::Format envFormat = vk::Format::eR16G16B16A16Sfloat;
    const uint32_t envDim = 512u;
    const uint32_t irradianceDim = 64u;
    const uint32_t prefilterDim = 512u;
    const uint32_t prefilterMipLevels = static_cast<uint32_t>(std::floor(std::log2(prefilterDim))) + 1u;
    const uint32_t brdfDim = 512u;

    // Load HDR equirectangular environment map
    LoadHDRTextureFromFile("newport_loft.hdr", hdrEquirectData);
    createTextureSampler(hdrEquirectData.textureSampler);

    // Create cubemap images
    createImage(envDim, envDim, 1, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, envCubemapData);
    createImage(irradianceDim, irradianceDim, 1, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, irradianceCubemapData);
    createImage(prefilterDim, prefilterDim, prefilterMipLevels, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, prefilteredEnvMapData);
    createImage(brdfDim, brdfDim, 1, 1, {}, envFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, brdfLutData);

    envCubemapData.mipLevels = 1;
    irradianceCubemapData.mipLevels = 1;
    prefilteredEnvMapData.mipLevels = prefilterMipLevels;
    brdfLutData.mipLevels = 1;

    // Create image views
    envCubemapData.textureImageView = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
        .image = envCubemapData.textureImage,
        .viewType = vk::ImageViewType::eCube,
        .format = envFormat,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 } });
    irradianceCubemapData.textureImageView = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
        .image = irradianceCubemapData.textureImage,
        .viewType = vk::ImageViewType::eCube,
        .format = envFormat,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 } });
    prefilteredEnvMapData.textureImageView = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
        .image = prefilteredEnvMapData.textureImage,
        .viewType = vk::ImageViewType::eCube,
        .format = envFormat,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, prefilterMipLevels, 0, 6 } });
    brdfLutData.textureImageView = createImageView(brdfLutData.textureImage, envFormat,
        vk::ImageAspectFlagBits::eColor, 1);

    // Create samplers for cubemaps
    vk::SamplerCreateInfo envSamplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(prefilterMipLevels),
    };
    envCubemapData.textureSampler = vk::raii::Sampler(device, envSamplerInfo);
    irradianceCubemapData.textureSampler = vk::raii::Sampler(device, envSamplerInfo);
    prefilteredEnvMapData.textureSampler = vk::raii::Sampler(device, envSamplerInfo);
    brdfLutData.textureSampler = vk::raii::Sampler(device, vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .minLod = 0.0f,
        .maxLod = 0.0f });

    // Generate cube mesh
    Mesh cubeMesh;
    generateCube(cubeMesh);
    createVertexBuffer(cubeMesh);
    createIndexBuffer(cubeMesh);

    // Build IBL pipelines 
    auto cmd = beginSingleTimeCommands();

    // Transition images to attachment-optimal
    transitionImageLayoutCmd(*cmd, envCubemapData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        0, 1, 0, 6,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, vk::AccessFlagBits::eColorAttachmentWrite);
    transitionImageLayoutCmd(*cmd, irradianceCubemapData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        0, 1, 0, 6,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, vk::AccessFlagBits::eColorAttachmentWrite);
    transitionImageLayoutCmd(*cmd, prefilteredEnvMapData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        0, prefilterMipLevels, 0, 6,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, vk::AccessFlagBits::eColorAttachmentWrite);
    transitionImageLayoutCmd(*cmd, brdfLutData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        0, 1, 0, 1,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, vk::AccessFlagBits::eColorAttachmentWrite);

    // Load shaders
    auto filterCubeCode = readFile(std::string(VK_SHADERS_DIR) + "filtercube.spv");
    auto irradianceCode = readFile(std::string(VK_SHADERS_DIR) + "irradiancecube.spv");
    auto prefilterCode = readFile(std::string(VK_SHADERS_DIR) + "prefilterenvmap.spv");
    auto brdfCode = readFile(std::string(VK_SHADERS_DIR) + "genbrdflut.spv");

    vk::raii::ShaderModule filterModule = createShaderModule(filterCubeCode);
    vk::raii::ShaderModule irradianceModule = createShaderModule(irradianceCode);
    vk::raii::ShaderModule prefilterModule = createShaderModule(prefilterCode);
    vk::raii::ShaderModule brdfModule = createShaderModule(brdfCode);

    // Common pipeline states
    auto bindingDesc = Vertex::getBindingDescription();
    std::array<vk::VertexInputAttributeDescription, 1> posOnlyAttr = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)) };
    vk::PipelineVertexInputStateCreateInfo posOnlyVertexInput{
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDesc,
        .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = posOnlyAttr.data() };
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
    vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };
    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::False, .depthWriteEnable = vk::False };
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };
    std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

    // Equirect → Cubemap pipeline 
    vk::DescriptorSetLayoutBinding equirectBinding{
        .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment };
    vk::raii::DescriptorSetLayout equirectSetLayout(device,
        vk::DescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &equirectBinding });

    struct PushMat4 { glm::mat4 mvp; };
    vk::PushConstantRange equirectPush{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushMat4) };
    vk::raii::PipelineLayout equirectPipelineLayout(device, vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1, .pSetLayouts = &*equirectSetLayout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &equirectPush });

    vk::DescriptorPoolSize equirectPoolSize{ .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1 };
    vk::raii::DescriptorPool equirectPool(device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &equirectPoolSize });
    vk::raii::DescriptorSet equirectSet = std::move(
        device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = *equirectPool, .descriptorSetCount = 1,
            .pSetLayouts = &*equirectSetLayout }).front());
    vk::DescriptorImageInfo hdrInfo{
        .sampler = hdrEquirectData.textureSampler, .imageView = hdrEquirectData.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    device.updateDescriptorSets(vk::WriteDescriptorSet{
        .dstSet = *equirectSet, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hdrInfo }, nullptr);

    std::array<vk::PipelineShaderStageCreateInfo, 2> equirectStages = {
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = filterModule, .pName = "vertMain" },
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = filterModule, .pName = "fragMain" } };
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> equirectPipeInfo = {
        {.stageCount = 2, .pStages = equirectStages.data(), .pVertexInputState = &posOnlyVertexInput,
          .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState,
          .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
          .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending,
          .pDynamicState = &dynamicState, .layout = equirectPipelineLayout },
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat } };
    vk::raii::Pipeline equirectPipeline(device, nullptr, equirectPipeInfo.get<vk::GraphicsPipelineCreateInfo>());

    // Irradiance pipeline 
    vk::DescriptorSetLayoutBinding cubeBinding{
        .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment };
    vk::raii::DescriptorSetLayout cubeSetLayout(device,
        vk::DescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &cubeBinding });

    struct PushIrradiance { glm::mat4 mvp; float deltaPhi; float deltaTheta; float pad0; float pad1; };
    vk::PushConstantRange irradPush{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, sizeof(PushIrradiance) };
    vk::raii::PipelineLayout irradPipelineLayout(device, vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1, .pSetLayouts = &*cubeSetLayout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &irradPush });

    struct PushPrefilter { glm::mat4 mvp; float roughness; uint32_t numSamples; float pad0; float pad1; };
    vk::PushConstantRange prefilterPush{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, sizeof(PushPrefilter) };
    vk::raii::PipelineLayout prefilterPipelineLayout(device, vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1, .pSetLayouts = &*cubeSetLayout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &prefilterPush });

    vk::DescriptorPoolSize cubePoolSize{ .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2 };
    vk::raii::DescriptorPool cubePool(device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &cubePoolSize });
    std::array<vk::DescriptorSetLayout, 2> cubeLayouts = { *cubeSetLayout, *cubeSetLayout };
    vk::raii::DescriptorSets cubeSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *cubePool, .descriptorSetCount = 2, .pSetLayouts = cubeLayouts.data() });
    vk::DescriptorImageInfo envCubeInfo{
        .sampler = envCubemapData.textureSampler, .imageView = envCubemapData.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
    device.updateDescriptorSets(vk::WriteDescriptorSet{
        .dstSet = *cubeSets[0], .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &envCubeInfo }, nullptr);
    device.updateDescriptorSets(vk::WriteDescriptorSet{
        .dstSet = *cubeSets[1], .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &envCubeInfo }, nullptr);

    std::array<vk::PipelineShaderStageCreateInfo, 2> irradianceStages = {
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = irradianceModule, .pName = "vertMain" },
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = irradianceModule, .pName = "fragMain" } };
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> irradPipeInfo = {
        {.stageCount = 2, .pStages = irradianceStages.data(), .pVertexInputState = &posOnlyVertexInput,
          .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState,
          .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
          .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending,
          .pDynamicState = &dynamicState, .layout = irradPipelineLayout },
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat } };
    vk::raii::Pipeline irradiancePipeline(device, nullptr, irradPipeInfo.get<vk::GraphicsPipelineCreateInfo>());

    std::array<vk::PipelineShaderStageCreateInfo, 2> prefilterStages = {
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = prefilterModule, .pName = "vertMain" },
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = prefilterModule, .pName = "fragMain" } };
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> prefilterPipeInfo = {
        {.stageCount = 2, .pStages = prefilterStages.data(), .pVertexInputState = &posOnlyVertexInput,
          .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState,
          .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
          .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending,
          .pDynamicState = &dynamicState, .layout = prefilterPipelineLayout },
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat } };
    vk::raii::Pipeline prefilterPipeline(device, nullptr, prefilterPipeInfo.get<vk::GraphicsPipelineCreateInfo>());

    // BRDF LUT pipeline (fullscreen triangle, no vertex input) 
    vk::raii::PipelineLayout brdfPipelineLayout(device, vk::PipelineLayoutCreateInfo{});
    vk::PipelineVertexInputStateCreateInfo emptyVertexInput{};
    std::array<vk::PipelineShaderStageCreateInfo, 2> brdfStages = {
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = brdfModule, .pName = "vertMain" },
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = brdfModule, .pName = "fragMain" } };
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> brdfPipeInfo = {
        {.stageCount = 2, .pStages = brdfStages.data(), .pVertexInputState = &emptyVertexInput,
          .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState,
          .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
          .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending,
          .pDynamicState = &dynamicState, .layout = brdfPipelineLayout },
        {.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat } };
    vk::raii::Pipeline brdfPipeline(device, nullptr, brdfPipeInfo.get<vk::GraphicsPipelineCreateInfo>());

    // Render cubemap faces 
    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    captureProjection[1][1] *= -1;
    std::array<glm::mat4, 6> captureViews = {
        glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    };

    // 1) Equirect → Cubemap
    for (uint32_t face = 0; face < 6; face++)
    {
        vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
            .image = envCubemapData.textureImage, .viewType = vk::ImageViewType::e2D,
            .format = envFormat,
            .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, face, 1 } });
        vk::RenderingAttachmentInfo colorAtt{
            .imageView = *faceView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) };
        vk::RenderingInfo renderingInfo{
            .renderArea = { {0, 0}, {envDim, envDim} }, .layerCount = 1,
            .colorAttachmentCount = 1, .pColorAttachments = &colorAtt };
        cmd->beginRendering(renderingInfo);
        cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(envDim), static_cast<float>(envDim), 0.0f, 1.0f));
        cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { envDim, envDim }));
        cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *equirectPipeline);
        cmd->bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
        cmd->bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexType::eUint16);
        cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *equirectPipelineLayout, 0, *equirectSet, {});
        PushMat4 pc{ .mvp = captureProjection * captureViews[face] };
        cmd->pushConstants<PushMat4>(*equirectPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pc);
        cmd->drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), 1, 0, 0, 0);
        cmd->endRendering();
    }
    transitionImageLayoutCmd(*cmd, envCubemapData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 1, 0, 6,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

    // 2) Irradiance cubemap
    for (uint32_t face = 0; face < 6; face++)
    {
        vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
            .image = irradianceCubemapData.textureImage, .viewType = vk::ImageViewType::e2D,
            .format = envFormat,
            .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, face, 1 } });
        vk::RenderingAttachmentInfo colorAtt{
            .imageView = *faceView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) };
        vk::RenderingInfo renderingInfo{
            .renderArea = { {0, 0}, {irradianceDim, irradianceDim} }, .layerCount = 1,
            .colorAttachmentCount = 1, .pColorAttachments = &colorAtt };
        cmd->beginRendering(renderingInfo);
        cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(irradianceDim), static_cast<float>(irradianceDim), 0.0f, 1.0f));
        cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { irradianceDim, irradianceDim }));
        cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *irradiancePipeline);
        cmd->bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
        cmd->bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexType::eUint16);
        cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *irradPipelineLayout, 0, *cubeSets[0], {});
        PushIrradiance ipc{ .mvp = captureProjection * captureViews[face], .deltaPhi = 0.025f, .deltaTheta = 0.025f };
        cmd->pushConstants<PushIrradiance>(*irradPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, ipc);
        cmd->drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), 1, 0, 0, 0);
        cmd->endRendering();
    }
    transitionImageLayoutCmd(*cmd, irradianceCubemapData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 1, 0, 6,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

    // 3) Prefiltered env map
    for (uint32_t mip = 0; mip < prefilterMipLevels; mip++)
    {
        uint32_t mipWidth = std::max(1u, prefilterDim >> mip);
        uint32_t mipHeight = std::max(1u, prefilterDim >> mip);
        float roughness = static_cast<float>(mip) / static_cast<float>(prefilterMipLevels - 1u);

        for (uint32_t face = 0; face < 6; face++)
        {
            vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
                .image = prefilteredEnvMapData.textureImage, .viewType = vk::ImageViewType::e2D,
                .format = envFormat,
                .subresourceRange = { vk::ImageAspectFlagBits::eColor, mip, 1, face, 1 } });
            vk::RenderingAttachmentInfo colorAtt{
                .imageView = *faceView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) };
            vk::RenderingInfo renderingInfo{
                .renderArea = { {0, 0}, {mipWidth, mipHeight} }, .layerCount = 1,
                .colorAttachmentCount = 1, .pColorAttachments = &colorAtt };
            cmd->beginRendering(renderingInfo);
            cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(mipWidth), static_cast<float>(mipHeight), 0.0f, 1.0f));
            cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { mipWidth, mipHeight }));
            cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *prefilterPipeline);
            cmd->bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
            cmd->bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexType::eUint16);
            cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *prefilterPipelineLayout, 0, *cubeSets[1], {});
            PushPrefilter ppc{ .mvp = captureProjection * captureViews[face], .roughness = roughness, .numSamples = 64u };
            cmd->pushConstants<PushPrefilter>(*prefilterPipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, ppc);
            cmd->drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), 1, 0, 0, 0);
            cmd->endRendering();
        }
    }
    transitionImageLayoutCmd(*cmd, prefilteredEnvMapData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, prefilterMipLevels, 0, 6,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

    // 4) BRDF LUT
    {
        vk::RenderingAttachmentInfo colorAtt{
            .imageView = brdfLutData.textureImageView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) };
        vk::RenderingInfo renderingInfo{
            .renderArea = { {0, 0}, {brdfDim, brdfDim} }, .layerCount = 1,
            .colorAttachmentCount = 1, .pColorAttachments = &colorAtt };
        cmd->beginRendering(renderingInfo);
        cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(brdfDim), static_cast<float>(brdfDim), 0.0f, 1.0f));
        cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { brdfDim, brdfDim }));
        cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *brdfPipeline);
        cmd->draw(3, 1, 0, 0);
        cmd->endRendering();
    }
    transitionImageLayoutCmd(*cmd, brdfLutData.textureImage, vk::ImageAspectFlagBits::eColor,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 1, 0, 1,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

    endSingleTimeCommands(*cmd);

    std::cout << "[vkrEngine] IBL resources generated successfully." << std::endl;
}

// CSM (Cascaded Shadow Maps) — Phase 3
bool VkrRenderer::createCsmResources()
{
    std::cout << "[vkrEngine] Creating CSM resources (double-buffered)..." << std::endl;

    if (!createCsmDescriptorPool()) return false;

    vk::Format depthFormat = findSupportedFormat(
        { vk::Format::eD32Sfloat, vk::Format::eD16Unorm },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
    );

    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToBorder,
        .addressModeV = vk::SamplerAddressMode::eClampToBorder,
        .addressModeW = vk::SamplerAddressMode::eClampToBorder,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::False,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f, .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatOpaqueWhite,
    };

    // Create one shadow map per frame-in-flight (double-buffered)
    csmSamplers.clear();
    csmArrayViews.clear();
    for (uint32_t fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi)
    {
        try
        {
            createImage(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1, CASCADE_COUNT, {},
                depthFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal, csmTextureArrays[fi]);

            csmLayerViewsArray[fi].clear();
            for (uint32_t layer = 0; layer < CASCADE_COUNT; ++layer)
            {
                csmLayerViewsArray[fi].emplace_back(device, vk::ImageViewCreateInfo{
                    .image = *csmTextureArrays[fi].textureImage,
                    .viewType = vk::ImageViewType::e2DArray,
                    .format = depthFormat,
                    .subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, layer, 1 }
                    });
            }

            csmArrayViews.emplace_back(device, vk::ImageViewCreateInfo{
                .image = *csmTextureArrays[fi].textureImage,
                .viewType = vk::ImageViewType::e2DArray,
                .format = depthFormat,
                .subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, CASCADE_COUNT }
                });

            csmSamplers.emplace_back(device, samplerInfo);
        }
        catch (const std::exception& e)
        {
            std::cerr << "[vkrEngine] CSM resource creation failed for frame " << fi << ": "
                << e.what() << std::endl;
            return false;
        }
    }

    createCsmDescriptorSets();
    if (!createCsmDepthPipeline()) return false;

    std::cout << "[vkrEngine] CSM resources created (x" << MAX_FRAMES_IN_FLIGHT << ")." << std::endl;
    return true;
}

bool VkrRenderer::createCsmDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT },
        };

        csmDepthDescriptorPool = vk::raii::DescriptorPool(device,
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
        std::cerr << "[vkrEngine] Failed to create CSM descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void VkrRenderer::createCsmDescriptorSets()
{
    // Create descriptor set layout for CSM depth pass
    vk::DescriptorSetLayoutBinding csmUboBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };
    csmDepthDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &csmUboBinding,
        });

    // Allocate descriptor sets
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *csmDepthDescriptorSetLayout);
    csmDepthDescriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *csmDepthDescriptorPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = layouts.data(),
        });

    // Write descriptors
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo csmBuf{
            .buffer = *csmUboResources.Buffers[i],
            .offset = 0,
            .range = sizeof(CsmUBO),
        };
        vk::WriteDescriptorSet write{
            .dstSet = *csmDepthDescriptorSets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &csmBuf,
        };
        device.updateDescriptorSets(write, nullptr);
    }
}

bool VkrRenderer::createCsmDepthPipeline()
{
    try
    {
        // Pipeline layout: Set 0 (CsmUBO) + push constant (mat4 model + uint cascadeIndex)
        vk::PushConstantRange pushRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset = 0,
            .size = sizeof(glm::mat4) + sizeof(uint32_t),
        };
        csmDepthPipelineLayout = vk::raii::PipelineLayout(device,
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount = 1,
                .pSetLayouts = &*csmDepthDescriptorSetLayout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pushRange,
            });

        // Shader
        auto shaderCode = readFile(std::string(VK_SHADERS_DIR) + "csm_depth.spv");
        vk::raii::ShaderModule shader = createShaderModule(shaderCode);

        vk::PipelineShaderStageCreateInfo vertStage{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *shader, .pName = "vertMain",
        };
        vk::PipelineShaderStageCreateInfo fragStage{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *shader, .pName = "fragMain",
        };
        vk::PipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

        // Vertex input: position only (VkrVertex has pos at offset 0)
        vk::VertexInputBindingDescription bindingDesc{ 0, sizeof(VkrVertex), vk::VertexInputRate::eVertex };
        vk::VertexInputAttributeDescription attrDesc{ 0, 0, vk::Format::eR32G32B32Sfloat, 0 };

        vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDesc,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &attrDesc,
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False,
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1, .scissorCount = 1,
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
            .lineWidth = 1.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLessOrEqual,
        };

        // No color attachments in depth pass
        vk::PipelineColorBlendStateCreateInfo colorBlend{
            .attachmentCount = 0,
            .pAttachments = nullptr,
        };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };

        vk::Format depthFormat = findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD16Unorm },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
        );

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
            },
            {
                .colorAttachmentCount = 0,
                .depthAttachmentFormat = depthFormat,
            }
        };

        csmDepthPipeline = vk::raii::Pipeline(device, nullptr,
            chain.get<vk::GraphicsPipelineCreateInfo>());

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkrEngine] Failed to create CSM depth pipeline: " << e.what() << std::endl;
        return false;
    }
}

void VkrRenderer::calculateCascadeSplits()
{
    const float nearPlane = CAMERA_NEAR;
    const float farPlane = CAMERA_FAR;

    cascadeSplitDepths[0] = nearPlane;
    cascadeSplitDepths[CASCADE_COUNT] = farPlane;

    for (uint32_t i = 1; i < CASCADE_COUNT; ++i)
    {
        float ratio = static_cast<float>(i) / static_cast<float>(CASCADE_COUNT);
        float logSplit = nearPlane * std::pow(farPlane / nearPlane, ratio);
        float linSplit = nearPlane + (farPlane - nearPlane) * ratio;
        cascadeSplitDepths[i] = uiSplitLambda * logSplit + (1.0f - uiSplitLambda) * linSplit;
    }
}

void VkrRenderer::computeCascadeViewProj(uint32_t cascadeIndex, const glm::vec3& lightDir)
{
    // =========================================================================
    // Camera-Independent Shadow Projection (using model's actual AABB)
    //
    // The shadow map covers the entire scene from the light's perspective.
    // Since the directional light direction is fixed and the scene AABB is
    // fixed, shadows remain completely static regardless of camera movement.
    // Cascade selection is handled in the shader via viewDepth.
    // =========================================================================

    // Scene bounding box from the loaded model, with padding
    constexpr float kAabbPad = 20.0f;
    const glm::vec3 sceneMin = sponzaModel.aabbMin - glm::vec3(kAabbPad);
    const glm::vec3 sceneMax = sponzaModel.aabbMax + glm::vec3(kAabbPad);

    // 8 corners of the scene bounding box in world space
    const std::array<glm::vec3, 8> cornersWS = { {
        glm::vec3(sceneMin.x, sceneMin.y, sceneMin.z),
        glm::vec3(sceneMax.x, sceneMin.y, sceneMin.z),
        glm::vec3(sceneMax.x, sceneMax.y, sceneMin.z),
        glm::vec3(sceneMin.x, sceneMax.y, sceneMin.z),
        glm::vec3(sceneMin.x, sceneMin.y, sceneMax.z),
        glm::vec3(sceneMax.x, sceneMin.y, sceneMax.z),
        glm::vec3(sceneMax.x, sceneMax.y, sceneMax.z),
        glm::vec3(sceneMin.x, sceneMax.y, sceneMax.z),
    } };

    // Scene center for light to look at
    const glm::vec3 sceneCenter = (sceneMin + sceneMax) * 0.5f;

    // Light view matrix: position outside the AABB, looking at its center.
    // Use the AABB diagonal length + padding to ensure the light is always
    // outside the model bounds, regardless of model scale.
    const float lightDist = glm::length(sceneMax - sceneMin) * 1.5f;
    const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    const glm::vec3 lightUp = (std::abs(glm::dot(lightDir, worldUp)) > 0.999f)
        ? glm::vec3(1.0f, 0.0f, 0.0f) : worldUp;
    const glm::vec3 lightPos = sceneCenter - lightDir * lightDist;
    const glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, lightUp);

    // Transform AABB corners to light view space
    glm::vec3 minAABB(std::numeric_limits<float>::max());
    glm::vec3 maxAABB(std::numeric_limits<float>::lowest());
    for (const auto& corner : cornersWS)
    {
        const glm::vec4 ls = lightView * glm::vec4(corner, 1.0f);
        minAABB = glm::min(minAABB, glm::vec3(ls));
        maxAABB = glm::max(maxAABB, glm::vec3(ls));
    }

    // Expand Z range for potential occluders outside the AABB
    const float zPadding = 50.0f;
    minAABB.z -= zPadding;                                   // further (more negative)
    maxAABB.z = std::min(maxAABB.z + zPadding, -0.01f);      // closer, clamp to negative

    // ---- Texel Snapping ----
    // Quantize the AABB to the shadow map texel grid. Since the scene AABB
    // is fixed, this mainly prevents shimmering if the light direction changes.
    const float worldUnitsPerTexelX = (maxAABB.x - minAABB.x) / float(SHADOW_MAP_SIZE);
    const float worldUnitsPerTexelY = (maxAABB.y - minAABB.y) / float(SHADOW_MAP_SIZE);
    const auto snap = [](float v, float texelSize) -> float {
        return std::floor(v / texelSize) * texelSize;
        };

    const float snappedMinX = snap(minAABB.x, worldUnitsPerTexelX);
    const float snappedMaxX = snap(maxAABB.x, worldUnitsPerTexelX);
    const float snappedMinY = snap(minAABB.y, worldUnitsPerTexelY);
    const float snappedMaxY = snap(maxAABB.y, worldUnitsPerTexelY);

    // Diagnostic: first cascade of first 3 frames
    if (cascadeIndex == 0)
    {
        static int diagFrame = 0;
        if (diagFrame < 3)
        {
            std::cout << "[CSM diag] frame " << diagFrame
                << " lightDir=(" << lightDir.x << "," << lightDir.y << "," << lightDir.z << ")"
                << std::endl;
            std::cout << "[CSM diag] modelAABB:("
                << sponzaModel.aabbMin.x << "," << sponzaModel.aabbMin.y << "," << sponzaModel.aabbMin.z << ")-("
                << sponzaModel.aabbMax.x << "," << sponzaModel.aabbMax.y << "," << sponzaModel.aabbMax.z << ")"
                << std::endl;
            std::cout << "[CSM diag] lightView AABB:("
                << minAABB.x << "," << minAABB.y << "," << minAABB.z << ")-("
                << maxAABB.x << "," << maxAABB.y << "," << maxAABB.z << ")"
                << " zNear=" << -maxAABB.z << " zFar=" << -minAABB.z
                << " texelSize=(" << worldUnitsPerTexelX << "," << worldUnitsPerTexelY << ")"
                << std::endl;
            ++diagFrame;
        }
    }

    // Build orthographic projection from the snapped AABB.
    // All cascades share the same projection — cascade selection is done
    // in the shader via viewDepth, which only affects which layer to sample.
    glm::mat4 lightProj = glm::ortho(
        snappedMinX, snappedMaxX,
        snappedMinY, snappedMaxY,
        -maxAABB.z, -minAABB.z);
    lightProj[1][1] *= -1;  // Vulkan NDC flip

    cascadeViewProj[cascadeIndex] = lightProj * lightView;
}

void VkrRenderer::updateCsmBuffers(uint32_t frameIndex)
{
    // --- Light direction (read from SceneUBO, already set by updateSceneUBO) ---
    const auto* sceneUbo = static_cast<const SceneUBO*>(
        sceneUboResources.BuffersMapped[frameIndex]);
    const glm::vec3 lightDir = sceneUbo->lightDir;

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
    ShadowParamsUBO sp{};
    sp.shadowFilterMode = uiShadowFilterMode;
    sp.pcfRadiusTexels = uiPcfRadiusTexels;
    sp.pcssLightSizeTexels = uiPcssLightSizeTexels;
    sp.shadowBiasMin = 0.0006f;
    sp.invShadowMapSize = glm::vec2(1.0f / float(SHADOW_MAP_SIZE), 1.0f / float(SHADOW_MAP_SIZE));
    std::memcpy(shadowParamsUboResources.BuffersMapped[frameIndex], &sp, sizeof(sp));
}

void VkrRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    // CSM shadow map synchronization is handled in render() by waiting on
    // both inFlightFences before recording commands.

    auto& cmd = commandBuffers[currentFrame];

    cmd.begin(vk::CommandBufferBeginInfo{});
    gpuProfiler.beginFrame(*cmd);

    // Pass 0: CSM Depth Pass (double-buffered per currentFrame)
    // DEBUG: Set BYPASS_CSM to true to skip shadow map rendering (shadow factor = 1.0).
    // If the crash disappears, the problem is in the shadow path (likely PCSS shader).
    constexpr bool BYPASS_CSM = false; // CSM shadows enabled

    auto& csmTex = csmTextureArrays[currentFrame];
    auto& csmViews = csmLayerViewsArray[currentFrame];
    auto& csmLayout = csmArrayLayouts[currentFrame];

    // Transition CSM array to ShaderReadOnly (skip depth writes when bypassed)
    if (BYPASS_CSM)
    {
        // Clear to 1.0 (far depth = no shadow) so the shader always sees "lit"
        vk::ImageMemoryBarrier2 clearBarrier{
            .srcStageMask = (csmLayout == vk::ImageLayout::eUndefined)
                ? vk::PipelineStageFlagBits2::eTopOfPipe
                : vk::PipelineStageFlagBits2::eFragmentShader,
            .srcAccessMask = (csmLayout == vk::ImageLayout::eUndefined)
                ? vk::AccessFlagBits2::eNone
                : vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = csmLayout,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *csmTex.textureImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
            }
        };
        vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &clearBarrier };
        cmd.pipelineBarrier2(depInfo);

        vk::ClearDepthStencilValue clearVal(1.0f, 0);
        cmd.clearDepthStencilImage(*csmTex.textureImage, vk::ImageLayout::eTransferDstOptimal,
            clearVal, vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
            });

        vk::ImageMemoryBarrier2 readBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *csmTex.textureImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
            }
        };
        vk::DependencyInfo depInfo2{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &readBarrier };
        cmd.pipelineBarrier2(depInfo2);
        csmLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    else
    {
        vk::PipelineStageFlags2 srcStage;
        vk::AccessFlags2        srcAccess;
        if (csmLayout == vk::ImageLayout::eUndefined)
        {
            srcStage = vk::PipelineStageFlagBits2::eTopOfPipe;
            srcAccess = vk::AccessFlagBits2::eNone;
        }
        else
        {
            srcStage = vk::PipelineStageFlagBits2::eFragmentShader;
            srcAccess = vk::AccessFlagBits2::eShaderRead;
        }

        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests
                          | vk::PipelineStageFlagBits2::eLateFragmentTests,
            .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout = csmLayout,
            .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *csmTex.textureImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
            }
        };
        vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
        cmd.pipelineBarrier2(depInfo);

        csmLayout = vk::ImageLayout::eDepthAttachmentOptimal;

        // Render each cascade
        for (uint32_t cascadeIdx = 0; cascadeIdx < CASCADE_COUNT; ++cascadeIdx)
        {
            vk::RenderingAttachmentInfo depthAttachment{
                .imageView = *csmViews[cascadeIdx],
                .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
            };

            vk::RenderingInfo renderInfo{
                .renderArea = {.offset = {0, 0}, .extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}},
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
                *csmDepthPipelineLayout, 0, *csmDepthDescriptorSets[currentFrame], nullptr);

            cmd.bindVertexBuffers(0, *sponzaModel.vertexBuffer, vk::DeviceSize{ 0 });
            cmd.bindIndexBuffer(*sponzaModel.indexBuffer, 0, vk::IndexType::eUint32);

            const glm::mat4 identity(1.0f);
            cmd.pushConstants<glm::mat4>(*csmDepthPipelineLayout,
                vk::ShaderStageFlagBits::eVertex, 0, identity);
            cmd.pushConstants<uint32_t>(*csmDepthPipelineLayout,
                vk::ShaderStageFlagBits::eVertex, sizeof(glm::mat4), cascadeIdx);

            for (const auto& sub : sponzaModel.subMeshes)
                cmd.drawIndexed(sub.indexCount, 1, sub.firstIndex, 0, 0);

            cmd.endRendering();
        }

        // Transition CSM array for shader reading
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
                .image = *csmTex.textureImage,
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0, .levelCount = 1,
                    .baseArrayLayer = 0, .layerCount = CASCADE_COUNT
                }
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
        }
        csmLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    // ================================================================
    // Pass 1: GBuffer Pass — render Sponza to MRT
    // ================================================================
    {
        // Transition GBuffer images
        std::array<std::pair<TextureData&, vk::ImageLayout&>, 3> gbufferColors = { {
            {gbufferAlbedo.texture, gbufferAlbedo.layout},
            {gbufferNormalRoughness.texture, gbufferNormalRoughness.layout},
            {gbufferPbr.texture, gbufferPbr.layout},
        } };

        for (auto& [tex, layout] : gbufferColors)
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = (layout == vk::ImageLayout::eUndefined)
                    ? vk::PipelineStageFlagBits2::eTopOfPipe
                    : vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = (layout == vk::ImageLayout::eUndefined)
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = layout,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *tex.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // Transition GBuffer depth
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = (gbufferDepth.layout == vk::ImageLayout::eUndefined)
                    ? vk::PipelineStageFlagBits2::eTopOfPipe
                    : vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = (gbufferDepth.layout == vk::ImageLayout::eUndefined)
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .oldLayout = gbufferDepth.layout,
                .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *gbufferDepth.texture.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            gbufferDepth.layout = vk::ImageLayout::eDepthAttachmentOptimal;
        }

        // GBuffer MRT rendering
        std::array<vk::RenderingAttachmentInfo, 3> gbufferColorAttachments{ {
            {.imageView = *gbufferAlbedo.texture.textureImageView,
             .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .loadOp = vk::AttachmentLoadOp::eClear,
             .storeOp = vk::AttachmentStoreOp::eStore,
             .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}},
            {.imageView = *gbufferNormalRoughness.texture.textureImageView,
             .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .loadOp = vk::AttachmentLoadOp::eClear,
             .storeOp = vk::AttachmentStoreOp::eStore,
             .clearValue = vk::ClearColorValue{0.5f, 0.5f, 0.5f, 0.04f}},
            {.imageView = *gbufferPbr.texture.textureImageView,
             .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
             .loadOp = vk::AttachmentLoadOp::eClear,
             .storeOp = vk::AttachmentStoreOp::eStore,
             .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}},
        } };

        vk::RenderingAttachmentInfo gbufferDepthAttachment{
            .imageView = *gbufferDepth.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearDepthStencilValue{1.0f, 0},
        };

        vk::RenderingInfo gbufferRenderInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = static_cast<uint32_t>(gbufferColorAttachments.size()),
            .pColorAttachments = gbufferColorAttachments.data(),
            .pDepthAttachment = &gbufferDepthAttachment,
        };

        cmd.beginRendering(gbufferRenderInfo);

        vk::Viewport viewport{
            .x = 0.0f, .y = 0.0f,
            .width = static_cast<float>(swapChainExtent.width),
            .height = static_cast<float>(swapChainExtent.height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };
        cmd.setViewport(0, viewport);
        cmd.setScissor(0, vk::Rect2D{ .offset = {0, 0}, .extent = swapChainExtent });

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *gbufferPipeline);

        gpuProfiler.beginPass(*cmd, "GBuffer");
        // Render all renderables with per-material bindings
        for (const auto& renderable : scene.renderables())
        {
            auto& model = *renderable.model;
            if (model.subMeshes.empty()) continue;

            cmd.bindVertexBuffers(0, *model.vertexBuffer, vk::DeviceSize{ 0 });
            cmd.bindIndexBuffer(*model.indexBuffer, 0, vk::IndexType::eUint32);

            for (const auto& sub : model.subMeshes)
            {
                // Bind Set 0 (scene-level)
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                    *gbufferPipelineLayout, 0,
                    *sceneUboResources.descriptorSets[currentFrame], nullptr);

                // Bind Set 1 (material-level)
                int32_t matIdx = sub.materialIndex;
                if (matIdx >= 0 && matIdx < static_cast<int32_t>(materialDescriptorSets.size()))
                {
                    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                        *gbufferPipelineLayout, 1,
                        *materialDescriptorSets[matIdx], nullptr);
                }

                // Push model matrix
                cmd.pushConstants<glm::mat4>(*gbufferPipelineLayout,
                    vk::ShaderStageFlagBits::eVertex, 0, renderable.transform);

                cmd.drawIndexed(sub.indexCount, 1, sub.firstIndex, 0, 0);
            }
        }
        statDrawCalls = static_cast<uint32_t>(sponzaModel.subMeshes.size());
        statTriangles = sponzaModel.totalTriangles;
        gpuProfiler.endPass(*cmd);

        cmd.endRendering();
    }

    // ================================================================
    // Pass 2: SSAO Pass
    // ================================================================
    {
        // Transition SSAO output to color attachment
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = (ssaoColor.layout == vk::ImageLayout::eUndefined)
                    ? vk::PipelineStageFlagBits2::eTopOfPipe
                    : vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = (ssaoColor.layout == vk::ImageLayout::eUndefined)
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = ssaoColor.layout,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *ssaoColor.texture.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            ssaoColor.layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // Transition GBuffer inputs to shader read
        {
            vk::ImageMemoryBarrier2 barriers[2] = {
                {
                    .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *gbufferNormalRoughness.texture.textureImage,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                },
                {
                    .srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests,
                    .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *gbufferDepth.texture.textureImage,
                    .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
                },
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 2, .pImageMemoryBarriers = barriers };
            cmd.pipelineBarrier2(depInfo);
            gbufferNormalRoughness.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            gbufferDepth.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }

        vk::RenderingAttachmentInfo ssaoColorAttachment{
            .imageView = *ssaoColor.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{1.0f, 1.0f, 1.0f, 1.0f},
        };
        vk::RenderingInfo ssaoRenderInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &ssaoColorAttachment,
        };
        cmd.beginRendering(ssaoRenderInfo);
        cmd.setViewport(0, vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f });
        cmd.setScissor(0, vk::Rect2D{ {0, 0}, swapChainExtent });

        gpuProfiler.beginPass(*cmd, "SSAO");
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *ssaoPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *ssaoPipelineLayout, 0,
            *ssaoSettingsUboResources.descriptorSets[currentFrame], nullptr);
        cmd.draw(3, 1, 0, 0);
        gpuProfiler.endPass(*cmd);

        cmd.endRendering();
    }

    // ================================================================
    // Pass 3: SSAO Blur Pass
    // ================================================================
    {
        // Transition SSAO blur output
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = (ssaoBlurColor.layout == vk::ImageLayout::eUndefined)
                    ? vk::PipelineStageFlagBits2::eTopOfPipe
                    : vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = (ssaoBlurColor.layout == vk::ImageLayout::eUndefined)
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = ssaoBlurColor.layout,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *ssaoBlurColor.texture.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            ssaoBlurColor.layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // Transition SSAO input to shader read
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *ssaoColor.texture.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            ssaoColor.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }

        vk::RenderingAttachmentInfo blurAttachment{
            .imageView = *ssaoBlurColor.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{1.0f, 1.0f, 1.0f, 1.0f},
        };
        vk::RenderingInfo blurRenderInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &blurAttachment,
        };
        cmd.beginRendering(blurRenderInfo);
        cmd.setViewport(0, vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f });
        cmd.setScissor(0, vk::Rect2D{ {0, 0}, swapChainExtent });

        gpuProfiler.beginPass(*cmd, "SSAO Blur");
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *ssaoBlurPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *ssaoBlurPipelineLayout, 0,
            *ssaoBlurUboResources.descriptorSets[currentFrame], nullptr);
        cmd.draw(3, 1, 0, 0);
        gpuProfiler.endPass(*cmd);

        cmd.endRendering();
    }

    // ================================================================
    // Pass 4: SSR Pass
    // ================================================================
    {
        // Transition SSR output
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = (ssrColor.layout == vk::ImageLayout::eUndefined)
                    ? vk::PipelineStageFlagBits2::eTopOfPipe
                    : vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = (ssrColor.layout == vk::ImageLayout::eUndefined)
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = ssrColor.layout,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *ssrColor.texture.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            ssrColor.layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // Transition GBuffer albedo to shader read (for SSR color sampling)
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *gbufferAlbedo.texture.textureImage,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            cmd.pipelineBarrier2(depInfo);
            gbufferAlbedo.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }

        vk::RenderingAttachmentInfo ssrAttachment{
            .imageView = *ssrColor.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        vk::RenderingInfo ssrRenderInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &ssrAttachment,
        };
        cmd.beginRendering(ssrRenderInfo);
        cmd.setViewport(0, vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f });
        cmd.setScissor(0, vk::Rect2D{ {0, 0}, swapChainExtent });

        gpuProfiler.beginPass(*cmd, "SSR");
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *ssrPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *ssrPipelineLayout, 0,
            *ssrParamsUboResources.descriptorSets[currentFrame], nullptr);
        cmd.draw(3, 1, 0, 0);
        gpuProfiler.endPass(*cmd);

        cmd.endRendering();
    }

    // ================================================================
    // Pass 5: Deferred Lighting Pass (full-screen quad → swapchain)
    // ================================================================
    {
        // Transition all GBuffer inputs to shader read
        {
            std::array<vk::ImageMemoryBarrier2, 4> barriers = { {
                {.srcStageMask = (gbufferAlbedo.layout == vk::ImageLayout::eColorAttachmentOptimal)
                     ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
                     : vk::PipelineStageFlagBits2::eFragmentShader,
                 .srcAccessMask = (gbufferAlbedo.layout == vk::ImageLayout::eColorAttachmentOptimal)
                     ? vk::AccessFlagBits2::eColorAttachmentWrite
                     : vk::AccessFlagBits2::eShaderRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = gbufferAlbedo.layout,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .image = *gbufferAlbedo.texture.textureImage,
                 .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
                {.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = gbufferNormalRoughness.layout,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .image = *gbufferNormalRoughness.texture.textureImage,
                 .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
                {.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = gbufferPbr.layout,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .image = *gbufferPbr.texture.textureImage,
                 .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
                {.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = gbufferDepth.layout,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .image = *gbufferDepth.texture.textureImage,
                 .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}},
            } };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 4, .pImageMemoryBarriers = barriers.data() };
            cmd.pipelineBarrier2(depInfo);
            gbufferAlbedo.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            gbufferPbr.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }

        // Transition SSAO blur and SSR to shader read
        {
            vk::ImageMemoryBarrier2 barriers[2] = {
                {
                    .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *ssaoBlurColor.texture.textureImage,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                },
                {
                    .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *ssrColor.texture.textureImage,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                },
            };
            vk::DependencyInfo depInfo{ .imageMemoryBarrierCount = 2, .pImageMemoryBarriers = barriers };
            cmd.pipelineBarrier2(depInfo);
            ssaoBlurColor.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            ssrColor.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }

        // Transition swapchain image to color attachment
        transition_image_layout(cmd,
            swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::AccessFlagBits2::eNone,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

        vk::RenderingAttachmentInfo deferredColorAttachment{
            .imageView = *swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.02f, 0.02f, 0.04f, 1.0f},
        };

        vk::RenderingInfo deferredRenderInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &deferredColorAttachment,
        };
        cmd.beginRendering(deferredRenderInfo);
        cmd.setViewport(0, vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f });
        cmd.setScissor(0, vk::Rect2D{ {0, 0}, swapChainExtent });

        gpuProfiler.beginPass(*cmd, "Deferred Lighting");
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *deferredPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *deferredPipelineLayout, 0,
            *sceneUboResources.descriptorSets[currentFrame], nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *deferredPipelineLayout, 1,
            *deferredSettingsUboResources.descriptorSets[currentFrame], nullptr);
        cmd.draw(3, 1, 0, 0);
        gpuProfiler.endPass(*cmd);

        cmd.endRendering();
    }

    // ================================================================
    // Pass 6: ImGui UI Pass (load onto deferred output)
    // ================================================================
    vk::Viewport viewport{
        .x = 0.0f, .y = 0.0f,
        .width = static_cast<float>(swapChainExtent.width),
        .height = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    {
        vk::RenderingAttachmentInfo uiAttachment{
            .imageView = *swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        vk::RenderingInfo uiRenderingInfo{
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &uiAttachment,
        };
        cmd.beginRendering(uiRenderingInfo);
        cmd.setViewport(0, viewport);
        cmd.setScissor(0, vk::Rect2D{ .offset = {0, 0}, .extent = swapChainExtent });
        vkrRecordUICmdBuffer(this, cmd, currentFrame);
        cmd.endRendering();
    }

    transition_image_layout(cmd,
        swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);

    cmd.end();
}

void VkrRenderer::renderModel(vk::CommandBuffer cmd, const VkrModel& model, const glm::mat4& transform)
{
    if (model.subMeshes.empty()) return;

    cmd.bindVertexBuffers(0, *model.vertexBuffer, vk::DeviceSize{ 0 });
    cmd.bindIndexBuffer(*model.indexBuffer, 0, vk::IndexType::eUint32);

    for (const auto& sub : model.subMeshes)
    {
        // Bind scene descriptor set (set 0)
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *pipelineLayout, 0,
            *sceneUboResources.descriptorSets[currentFrame], nullptr);

        // Bind material descriptor set (set 1)
        int32_t matIdx = sub.materialIndex;
        if (matIdx >= 0 && matIdx < static_cast<int32_t>(materialDescriptorSets.size()))
        {
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                *pipelineLayout, 1,
                *materialDescriptorSets[matIdx], nullptr);
        }
        else
        {
            // Bind first material's set as fallback (should have white texture)
            if (!materialDescriptorSets.empty())
            {
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                    *pipelineLayout, 1,
                    *materialDescriptorSets[0], nullptr);
            }
        }

        // model matrix
        cmd.pushConstants<glm::mat4>(*pipelineLayout,
            vk::ShaderStageFlagBits::eVertex, 0, transform);

        // Draw — indices are already global (baseVertex added during loading)
        cmd.drawIndexed(sub.indexCount, 1, sub.firstIndex, 0, 0);

        ++statDrawCalls;
        statTriangles += sub.indexCount / 3;
    }
}

// ====================================================================
// Phase 4: GBuffer Methods
// ====================================================================

bool VkrRenderer::createGBufferResources()
{
    try
    {
        gbufferAlbedo.format = vk::Format::eR16G16B16A16Sfloat;
        gbufferNormalRoughness.format = vk::Format::eR16G16B16A16Sfloat;
        gbufferPbr.format = vk::Format::eR16G16B16A16Sfloat;
        gbufferDepth.format = vk::Format::eD32Sfloat;

        auto createAttachment = [this](GBufferAttachment& att, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect) {
            createImage(swapChainExtent.width, swapChainExtent.height, 1, att.format, vk::ImageTiling::eOptimal,
                usage, vk::MemoryPropertyFlagBits::eDeviceLocal, att.texture);
            att.texture.textureImageView = createImageView(att.texture.textureImage, att.format, aspect, 1);
            att.layout = vk::ImageLayout::eUndefined;
            };

        createAttachment(gbufferAlbedo, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        createAttachment(gbufferNormalRoughness, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        createAttachment(gbufferPbr, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor);
        createAttachment(gbufferDepth, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eDepth);

        if (gbufferSampler == vk::raii::Sampler(nullptr))
        {
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
                .maxLod = 0.0f,
                .borderColor = vk::BorderColor::eFloatOpaqueWhite,
                .unnormalizedCoordinates = vk::False
            };
            gbufferSampler = vk::raii::Sampler(device, samplerInfo);
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkrEngine] Failed to create GBuffer: " << e.what() << std::endl;
        return false;
    }
}

void VkrRenderer::destroyGBufferResources()
{
    gbufferAlbedo.texture.textureImageView = nullptr;
    gbufferAlbedo.texture.textureImage = nullptr;
    gbufferAlbedo.texture.textureImageMemory = nullptr;
    gbufferNormalRoughness.texture.textureImageView = nullptr;
    gbufferNormalRoughness.texture.textureImage = nullptr;
    gbufferNormalRoughness.texture.textureImageMemory = nullptr;
    gbufferPbr.texture.textureImageView = nullptr;
    gbufferPbr.texture.textureImage = nullptr;
    gbufferPbr.texture.textureImageMemory = nullptr;
    gbufferDepth.texture.textureImageView = nullptr;
    gbufferDepth.texture.textureImage = nullptr;
    gbufferDepth.texture.textureImageMemory = nullptr;
    gbufferAlbedo.layout = vk::ImageLayout::eUndefined;
    gbufferNormalRoughness.layout = vk::ImageLayout::eUndefined;
    gbufferPbr.layout = vk::ImageLayout::eUndefined;
    gbufferDepth.layout = vk::ImageLayout::eUndefined;
}

bool VkrRenderer::createGBufferDescriptorSetLayout()
{
    // GBuffer pass reuses the existing scene + material descriptor set layouts.
    // No separate layout needed.
    return true;
}

bool VkrRenderer::createGBufferDescriptorPool()
{
    // GBuffer pass reuses the main descriptorPool for scene + material sets.
    // No separate pool needed.
    return true;
}

void VkrRenderer::createGBufferDescriptorSets()
{
    // GBuffer reuses scene descriptor sets from sceneUboResources
    // and material descriptor sets from materialDescriptorSets.
}

bool VkrRenderer::createGBufferPipeline()
{
    try
    {
        auto vertCode = readFile(std::string(VK_SHADERS_DIR) + "deferred_gbuffer.spv");
        auto fragCode = vertCode; // Same module, different entry points

        vk::raii::ShaderModule shaderModule = createShaderModule(vertCode);

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            {.stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule, .pName = "vertMain"},
            {.stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"},
        };

        vk::VertexInputBindingDescription bindingDesc{ 0, sizeof(VkrVertex), vk::VertexInputRate::eVertex };
        std::array<vk::VertexInputAttributeDescription, 4> attrDescs = { {
            {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VkrVertex, pos)},
            {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VkrVertex, normal)},
            {2, 0, vk::Format::eR32G32Sfloat, offsetof(VkrVertex, texCoord)},
            {3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VkrVertex, tangent)},
        } };

        vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDesc,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size()), .pVertexAttributeDescriptions = attrDescs.data(),
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f,
        };
        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True, .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess, .depthBoundsTestEnable = vk::False, .stencilTestEnable = vk::False,
        };

        std::array<vk::PipelineColorBlendAttachmentState, 3> blendAttachments{ {
            {.blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
            {.blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
            {.blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
        } };
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False, .attachmentCount = static_cast<uint32_t>(blendAttachments.size()), .pAttachments = blendAttachments.data(),
        };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        // Push constant: model matrix
        vk::PushConstantRange pushRange{ .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = sizeof(glm::mat4) };

        std::array<vk::DescriptorSetLayout, 2> setLayouts = { *sceneDescriptorSetLayout, *materialDescriptorSetLayout };
        gbufferPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 2, .pSetLayouts = setLayouts.data(),
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange,
            });

        std::array<vk::Format, 3> colorFormats = { gbufferAlbedo.format, gbufferNormalRoughness.format, gbufferPbr.format };

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = { {
            .stageCount = 2, .pStages = shaderStages.data(), .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly, .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState, .layout = gbufferPipelineLayout, .renderPass = nullptr,
        }, {
            .colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()), .pColorAttachmentFormats = colorFormats.data(),
            .depthAttachmentFormat = gbufferDepth.format,
        } };

        gbufferPipeline = vk::raii::Pipeline(device, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e) { std::cerr << "Failed to create GBuffer pipeline: " << e.what() << std::endl; return false; }
}

// ====================================================================
// Phase 4: SSAO Methods
// ====================================================================

bool VkrRenderer::createSSAOResources()
{
    try
    {
        ssaoColor.format = vk::Format::eR8Unorm;
        ssaoBlurColor.format = vk::Format::eR8Unorm;

        auto createAtt = [this](GBufferAttachment& att) {
            createImage(swapChainExtent.width, swapChainExtent.height, 1, att.format, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal, att.texture);
            att.texture.textureImageView = createImageView(att.texture.textureImage, att.format, vk::ImageAspectFlagBits::eColor, 1);
            att.layout = vk::ImageLayout::eUndefined;
            };

        createAtt(ssaoColor);
        createAtt(ssaoBlurColor);

        createUniformBuffers(ssaoSettingsUboResources, sizeof(SSAOSettingsUBO));
        createUniformBuffers(ssaoBlurUboResources, sizeof(BlurUBO));

        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO resources error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::destroySSAOResources()
{
    ssaoColor.texture.textureImageView = nullptr;
    ssaoColor.texture.textureImage = nullptr;
    ssaoColor.texture.textureImageMemory = nullptr;
    ssaoBlurColor.texture.textureImageView = nullptr;
    ssaoBlurColor.texture.textureImage = nullptr;
    ssaoBlurColor.texture.textureImageMemory = nullptr;
    ssaoColor.layout = vk::ImageLayout::eUndefined;
    ssaoBlurColor.layout = vk::ImageLayout::eUndefined;
}

void VkrRenderer::generateSsaoKernel()
{
    ssaoKernel.resize(SSAO_KERNEL_SIZE);
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator(42); // Fixed seed for reproducibility

    for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i)
    {
        glm::vec3 sample(randomFloats(generator) * 2.0f - 1.0f,
            randomFloats(generator) * 2.0f - 1.0f,
            randomFloats(generator));
        sample = glm::normalize(sample);
        float scale = static_cast<float>(i) / static_cast<float>(SSAO_KERNEL_SIZE);
        scale = 0.1f + 0.9f * scale * scale;
        sample *= scale;
        ssaoKernel[i] = glm::vec4(sample, 0.0f);
    }
}

void VkrRenderer::createSsaoNoiseTexture()
{
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator(123);

    std::vector<glm::vec4> noiseData(SSAO_NOISE_SIZE * SSAO_NOISE_SIZE);
    for (auto& n : noiseData)
    {
        n = glm::vec4(randomFloats(generator) * 2.0f - 1.0f,
            randomFloats(generator) * 2.0f - 1.0f,
            0.0f, 0.0f);
        n = glm::vec4(glm::normalize(glm::vec3(n)), 0.0f);
    }

    vk::DeviceSize bufferSize = sizeof(glm::vec4) * noiseData.size();
    vk::raii::Buffer stagingBuf(nullptr);
    vk::raii::DeviceMemory stagingMem(nullptr);
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuf, stagingMem);
    void* mapped = stagingMem.mapMemory(0, bufferSize);
    memcpy(mapped, noiseData.data(), static_cast<size_t>(bufferSize));
    stagingMem.unmapMemory();

    ssaoNoiseTexture.mipLevels = 1;
    createImage(SSAO_NOISE_SIZE, SSAO_NOISE_SIZE, 1, vk::Format::eR32G32B32A32Sfloat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, ssaoNoiseTexture);
    transitionImageLayout(ssaoNoiseTexture.textureImage, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, 1);
    copyBufferToImage(stagingBuf, ssaoNoiseTexture.textureImage, SSAO_NOISE_SIZE, SSAO_NOISE_SIZE);
    transitionImageLayout(ssaoNoiseTexture.textureImage, vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal, 1);
    ssaoNoiseTexture.textureImageView = createImageView(ssaoNoiseTexture.textureImage,
        vk::Format::eR32G32B32A32Sfloat, vk::ImageAspectFlagBits::eColor, 1);

    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eNearest, .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eRepeat, .addressModeV = vk::SamplerAddressMode::eRepeat, .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f, .anisotropyEnable = vk::False, .maxAnisotropy = 1.0f,
        .compareEnable = vk::False, .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f, .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatOpaqueWhite, .unnormalizedCoordinates = vk::False,
    };
    ssaoNoiseSampler = vk::raii::Sampler(device, samplerInfo);
}

bool VkrRenderer::createSsaoDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
        };
        ssaoDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO DSL error: " << e.what() << std::endl; return false; }
}

bool VkrRenderer::createSsaoDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u},
        };
        ssaoDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO pool error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::createSsaoDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *ssaoDescriptorSetLayout);
    ssaoSettingsUboResources.descriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *ssaoDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo settingsInfo{ .buffer = *ssaoSettingsUboResources.Buffers[i], .offset = 0, .range = sizeof(SSAOSettingsUBO) };
        vk::DescriptorImageInfo normalInfo{ .sampler = gbufferSampler, .imageView = gbufferNormalRoughness.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo depthInfo{ .sampler = gbufferSampler, .imageView = gbufferDepth.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo noiseInfo{ .sampler = ssaoNoiseSampler, .imageView = ssaoNoiseTexture.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::array<vk::WriteDescriptorSet, 4> writes = { {
            {.dstSet = *ssaoSettingsUboResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &settingsInfo},
            {.dstSet = *ssaoSettingsUboResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo},
            {.dstSet = *ssaoSettingsUboResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
            {.dstSet = *ssaoSettingsUboResources.descriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &noiseInfo},
        } };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool VkrRenderer::createSsaoPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "ssao.spv"));
        vk::PipelineShaderStageCreateInfo stages[] = {
            {.stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule, .pName = "vertMain"},
            {.stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"},
        };

        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };
        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{ .depthTestEnable = vk::False, .depthWriteEnable = vk::False };
        vk::PipelineColorBlendAttachmentState blend{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &blend };

        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        ssaoPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*ssaoDescriptorSetLayout, .pushConstantRangeCount = 0 });

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = { {
            .stageCount = 2, .pStages = stages, .pVertexInputState = &vertexInput, .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState, .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending, .pDynamicState = &dynamicState,
            .layout = ssaoPipelineLayout, .renderPass = nullptr,
        }, {
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &ssaoColor.format, .depthAttachmentFormat = vk::Format::eUndefined,
        } };
        ssaoPipeline = vk::raii::Pipeline(device, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO pipeline error: " << e.what() << std::endl; return false; }
}

bool VkrRenderer::createSsaoBlurDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
        };
        ssaoBlurDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO Blur DSL error: " << e.what() << std::endl; return false; }
}

bool VkrRenderer::createSsaoBlurDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT},
        };
        ssaoBlurDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO Blur pool error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::createSsaoBlurDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *ssaoBlurDescriptorSetLayout);
    ssaoBlurUboResources.descriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *ssaoBlurDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo blurInfo{ .buffer = *ssaoBlurUboResources.Buffers[i], .offset = 0, .range = sizeof(BlurUBO) };
        vk::DescriptorImageInfo ssaoInfo{ .sampler = gbufferSampler, .imageView = ssaoColor.texture.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        std::array<vk::WriteDescriptorSet, 2> writes = { {
            {.dstSet = *ssaoBlurUboResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &blurInfo},
            {.dstSet = *ssaoBlurUboResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &ssaoInfo},
        } };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool VkrRenderer::createSsaoBlurPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "ssao_blur.spv"));
        vk::PipelineShaderStageCreateInfo stages[] = {
            {.stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule, .pName = "vertMain"},
            {.stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"},
        };
        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };
        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{ .depthTestEnable = vk::False, .depthWriteEnable = vk::False };
        vk::PipelineColorBlendAttachmentState blend{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &blend };
        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        ssaoBlurPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*ssaoBlurDescriptorSetLayout, .pushConstantRangeCount = 0 });

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = { {
            .stageCount = 2, .pStages = stages, .pVertexInputState = &vertexInput, .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState, .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending, .pDynamicState = &dynamicState,
            .layout = ssaoBlurPipelineLayout, .renderPass = nullptr,
        }, {
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &ssaoBlurColor.format, .depthAttachmentFormat = vk::Format::eUndefined,
        } };
        ssaoBlurPipeline = vk::raii::Pipeline(device, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSAO Blur pipeline error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::updateSsaoBuffers(uint32_t frameIndex)
{
    SSAOSettingsUBO ssao{};
    glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom),
        static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
        CAMERA_NEAR, CAMERA_FAR);
    proj[1][1] *= -1.0f;  // Vulkan Y flip
    ssao.projection = proj;
    ssao.invProjection = glm::inverse(proj);
    ssao.invView = glm::inverse(camera.GetViewMatrix());
    ssao.params0 = glm::vec4(uiSsaoRadius, uiSsaoBias, uiSsaoIntensity, uiSsaoEnabled ? 1.0f : 0.0f);
    ssao.flags = glm::ivec4(uiSsaoDebugView, 0, 0, 0);
    for (uint32_t i = 0; i < SSAO_KERNEL_SIZE && i < ssaoKernel.size(); ++i)
        ssao.kernel[i] = ssaoKernel[i];
    memcpy(ssaoSettingsUboResources.BuffersMapped[frameIndex], &ssao, sizeof(SSAOSettingsUBO));
}

void VkrRenderer::updateSsaoBlurBuffers(uint32_t frameIndex)
{
    BlurUBO blur{};
    blur.texelSize = glm::vec2(1.0f / swapChainExtent.width, 1.0f / swapChainExtent.height);
    std::memcpy(ssaoBlurUboResources.BuffersMapped[frameIndex], &blur, sizeof(BlurUBO));
}

// ====================================================================
// Phase 4: SSR Methods
// ====================================================================

bool VkrRenderer::createSSRResources()
{
    try
    {
        ssrColor.format = vk::Format::eR16G16B16A16Sfloat;
        createImage(swapChainExtent.width, swapChainExtent.height, 1, ssrColor.format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, ssrColor.texture);
        ssrColor.texture.textureImageView = createImageView(ssrColor.texture.textureImage, ssrColor.format, vk::ImageAspectFlagBits::eColor, 1);
        ssrColor.layout = vk::ImageLayout::eUndefined;

        createUniformBuffers(ssrParamsUboResources, sizeof(SSRParams));
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSR resources error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::destroySSRResources()
{
    ssrColor.texture.textureImageView = nullptr;
    ssrColor.texture.textureImage = nullptr;
    ssrColor.texture.textureImageMemory = nullptr;
    ssrColor.layout = vk::ImageLayout::eUndefined;
}

bool VkrRenderer::createSsrDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 4, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
        };
        ssrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSR DSL error: " << e.what() << std::endl; return false; }
}

bool VkrRenderer::createSsrDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u},
        };
        ssrDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSR pool error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::createSsrDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *ssrDescriptorSetLayout);
    ssrParamsUboResources.descriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *ssrDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneUboInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo paramsInfo{ .buffer = *ssrParamsUboResources.Buffers[i], .offset = 0, .range = sizeof(SSRParams) };
        vk::DescriptorImageInfo depthInfo{ .sampler = gbufferSampler, .imageView = gbufferDepth.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo colorInfo{ .sampler = gbufferSampler, .imageView = gbufferAlbedo.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo normalInfo{ .sampler = gbufferSampler, .imageView = gbufferNormalRoughness.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::array<vk::WriteDescriptorSet, 5> writes = { {
            {.dstSet = *ssrParamsUboResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneUboInfo},
            {.dstSet = *ssrParamsUboResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
            {.dstSet = *ssrParamsUboResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &colorInfo},
            {.dstSet = *ssrParamsUboResources.descriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo},
            {.dstSet = *ssrParamsUboResources.descriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo},
        } };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool VkrRenderer::createSsrPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "ssr.spv"));
        vk::PipelineShaderStageCreateInfo stages[] = {
            {.stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule, .pName = "vertMain"},
            {.stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"},
        };
        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };
        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{ .depthTestEnable = vk::False, .depthWriteEnable = vk::False };
        vk::PipelineColorBlendAttachmentState blend{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &blend };
        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        ssrPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*ssrDescriptorSetLayout, .pushConstantRangeCount = 0 });

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = { {
            .stageCount = 2, .pStages = stages, .pVertexInputState = &vertexInput, .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState, .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending, .pDynamicState = &dynamicState,
            .layout = ssrPipelineLayout, .renderPass = nullptr,
        }, {
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &ssrColor.format, .depthAttachmentFormat = vk::Format::eUndefined,
        } };
        ssrPipeline = vk::raii::Pipeline(device, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e) { std::cerr << "SSR pipeline error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::updateSsrBuffers(uint32_t frameIndex)
{
    SSRParams params{};
    params.maxRayDistance = uiSsrMaxRayDistance;
    params.thickness = uiSsrThickness;
    params.stride = uiSsrStride;
    params.intensity = uiSsrEnabled ? uiSsrIntensity : 0.0f;
    params.invResolution = glm::vec2(1.0f / swapChainExtent.width, 1.0f / swapChainExtent.height);
    params.debugMode = 0;
    params.maxSteps = uiSsrMaxSteps;
    std::memcpy(ssrParamsUboResources.BuffersMapped[frameIndex], &params, sizeof(SSRParams));
}

// ====================================================================
// Phase 4: Deferred Lighting Methods
// ====================================================================

bool VkrRenderer::createDeferredLightingDescriptorSetLayout()
{
    // Set 1: GBuffer + SSAO + SSR + DeferredSettings (Set 0 = scene-level, reused)
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
        };
        deferredDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "Deferred lighting DSL error: " << e.what() << std::endl; return false; }
}

bool VkrRenderer::createDeferredLightingDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 6u},
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT},
        };
        deferredDescriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data() });
        return true;
    }
    catch (const std::exception& e) { std::cerr << "Deferred lighting pool error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::createDeferredLightingDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *deferredDescriptorSetLayout);
    deferredSettingsUboResources.descriptorSets = vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = *deferredDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data() });

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorImageInfo albedoInfo{ .sampler = gbufferSampler, .imageView = gbufferAlbedo.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo normalInfo{ .sampler = gbufferSampler, .imageView = gbufferNormalRoughness.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo pbrInfo{ .sampler = gbufferSampler, .imageView = gbufferPbr.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo depthInfo{ .sampler = gbufferSampler, .imageView = gbufferDepth.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo ssaoInfo{ .sampler = gbufferSampler, .imageView = ssaoBlurColor.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo ssrInfo{ .sampler = gbufferSampler, .imageView = ssrColor.texture.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorBufferInfo settingsInfo{ .buffer = *deferredSettingsUboResources.Buffers[i], .offset = 0, .range = sizeof(DeferredSettingsUBO) };

        std::array<vk::WriteDescriptorSet, 7> writes = { {
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &albedoInfo},
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo},
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &pbrInfo},
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &ssaoInfo},
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &ssrInfo},
            {.dstSet = *deferredSettingsUboResources.descriptorSets[i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &settingsInfo},
        } };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool VkrRenderer::createDeferredLightingPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "deferred_lighting.spv"));
        vk::PipelineShaderStageCreateInfo stages[] = {
            {.stage = vk::ShaderStageFlagBits::eVertex, .module = *shaderModule, .pName = "vertMain"},
            {.stage = vk::ShaderStageFlagBits::eFragment, .module = *shaderModule, .pName = "fragMain"},
        };
        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };
        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{ .depthTestEnable = vk::False, .depthWriteEnable = vk::False };
        vk::PipelineColorBlendAttachmentState blend{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .attachmentCount = 1, .pAttachments = &blend };
        std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        // Pipeline uses 2 sets: Set 0 (scene-level) + Set 1 (GBuffer/SSAO/SSR)
        // Set 2 (DeferredSettingsUBO) is bound separately
        std::array<vk::DescriptorSetLayout, 2> setLayouts = { *sceneDescriptorSetLayout, *deferredDescriptorSetLayout };
        deferredPipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 2, .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 });

        vk::Format swapchainFormat = swapChainImageFormat;
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = { {
            .stageCount = 2, .pStages = stages, .pVertexInputState = &vertexInput, .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState, .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending, .pDynamicState = &dynamicState,
            .layout = deferredPipelineLayout, .renderPass = nullptr,
        }, {
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchainFormat, .depthAttachmentFormat = vk::Format::eUndefined,
        } };
        deferredPipeline = vk::raii::Pipeline(device, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e) { std::cerr << "Deferred lighting pipeline error: " << e.what() << std::endl; return false; }
}

void VkrRenderer::updateDeferredSettingsBuffer(uint32_t frameIndex)
{
    DeferredSettingsUBO settings{};
    settings.ssaoEnabled = uiSsaoEnabled ? 1.0f : 0.0f;
    settings.ssrEnabled = uiSsrEnabled ? 1.0f : 0.0f;
    settings.debugView = static_cast<float>(uiDeferredDebugView);
    std::memcpy(deferredSettingsUboResources.BuffersMapped[frameIndex], &settings, sizeof(DeferredSettingsUBO));
}

bool VkrRenderer::recreateDeferredSizedResources()
{
    // 1. Clear descriptor set handles (they don't auto-free; pools own the lifecycle)
    ssaoSettingsUboResources.descriptorSets.clear();
    ssaoBlurUboResources.descriptorSets.clear();
    ssrParamsUboResources.descriptorSets.clear();
    deferredSettingsUboResources.descriptorSets.clear();

    // 2. Destroy pools — vkDestroyDescriptorPool frees all sets allocated from them
    ssaoDescriptorPool = nullptr;
    ssaoBlurDescriptorPool = nullptr;
    ssrDescriptorPool = nullptr;
    deferredDescriptorPool = nullptr;

    // 3. Destroy old textures
    destroyGBufferResources();
    destroySSAOResources();
    destroySSRResources();

    // 4. Recreate textures at new swapchain size
    if (!createGBufferResources()) return false;
    if (!createSSAOResources()) return false;
    if (!createSSRResources()) return false;

    // 5. Recreate pools and allocate new descriptor sets
    if (!createSsaoDescriptorPool()) return false;
    createSsaoDescriptorSets();
    if (!createSsaoBlurDescriptorPool()) return false;
    createSsaoBlurDescriptorSets();
    if (!createSsrDescriptorPool()) return false;
    createSsrDescriptorSets();
    if (!createDeferredLightingDescriptorPool()) return false;
    createDeferredLightingDescriptorSets();

    return true;
}

void VkrRenderer::recreateSwapChain()
{
    VulkanBase::recreateSwapChain();
    recreateDeferredSizedResources();
}

bool VkrRenderer::initUI()
{
    return initVulkanUI();
}

void VkrRenderer::updateUIPanel()
{
    if (!showStatsWindow) return;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("vkrEngine — PBR+IBL+CSM", &showStatsWindow,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing))
    {
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Separator();

        // Scene Stats
        ImGui::Text("Scene: %s", sponzaModel.name.c_str());
        ImGui::Text("  Triangles: %u", statTriangles);
        ImGui::Text("  Draw Calls: %u", statDrawCalls);
        ImGui::Text("  Sub-meshes: %zu", sponzaModel.subMeshes.size());
        ImGui::Text("  Materials: %zu", sponzaModel.materials.size());
        ImGui::Separator();

        ImGui::Text("CPU Frame: %.2f ms", statCpuMs);
        ImGui::Text("GPU Total:  %.2f ms", gpuProfiler.totalGpuMs());
        ImGui::Separator();

        // PBR / IBL Controls
        if (ImGui::CollapsingHeader("PBR / IBL Settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const char* lightingModes[] = { "Phase 1: Simple", "Phase 2: PBR+IBL" };
            ImGui::Combo("Lighting Mode", &uiLightingMode, lightingModes, 2);
            ImGui::Separator();

            ImGui::SliderFloat("Exposure", &uiExposure, 0.1f, 15.0f, "%.2f");
            ImGui::SliderFloat("Gamma", &uiGamma, 0.5f, 4.0f, "%.2f");
            ImGui::Checkbox("Directional Light", &uiEnableDirectionalLight);
        }

        // CSM Controls
        if (ImGui::CollapsingHeader("CSM Shadows", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const char* filterModes[] = { "Hard", "PCF", "PCSS" };
            ImGui::Combo("Filter Mode", &uiShadowFilterMode, filterModes, 3);
            ImGui::Separator();

            ImGui::SliderFloat("Split Lambda", &uiSplitLambda, 0.0f, 1.0f,
                "%.2f (0=linear, 1=log)");
            ImGui::Checkbox("Rotate Light", &uiRotateLight);

            if (uiShadowFilterMode == 1) // PCF
            {
                ImGui::SliderFloat("PCF Radius (texels)", &uiPcfRadiusTexels, 1.0f, 5.0f, "%.0f");
            }
            else if (uiShadowFilterMode == 2) // PCSS
            {
                ImGui::SliderFloat("Light Size (texels)", &uiPcssLightSizeTexels, 5.0f, 50.0f, "%.0f");
            }

            ImGui::Separator();
            ImGui::Text("Split Depths:");
            for (uint32_t i = 1; i <= CASCADE_COUNT; ++i)
            {
                ImGui::Text("  Cascade %u: %.2f", i, cascadeSplitDepths[i]);
            }
        }
        ImGui::Separator();

        // Phase 4: SSAO Controls
        if (ImGui::CollapsingHeader("SSAO", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("SSAO Enabled", &uiSsaoEnabled);
            ImGui::SliderFloat("SSAO Radius", &uiSsaoRadius, 0.1f, 3.0f, "%.2f");
            ImGui::SliderFloat("SSAO Bias", &uiSsaoBias, 0.001f, 0.2f, "%.3f");
            ImGui::SliderFloat("SSAO Intensity", &uiSsaoIntensity, 0.5f, 5.0f, "%.1f");
            ImGui::Separator();
            const char* ssaoDebugModes[] = { "Final AO", "Linear Depth", "Normal", "Noise", "Hit Count", "Range Check", "ViewPos Z" };
            ImGui::Combo("SSAO Debug", &uiSsaoDebugView, ssaoDebugModes, 7);
        }

        // Phase 4: SSR Controls
        if (ImGui::CollapsingHeader("SSR"))
        {
            ImGui::Checkbox("SSR Enabled", &uiSsrEnabled);
            ImGui::SliderInt("SSR Max Steps", &uiSsrMaxSteps, 16, 200);
            ImGui::SliderFloat("SSR Max Distance", &uiSsrMaxRayDistance, 1.0f, 50.0f, "%.1f");
            ImGui::SliderFloat("SSR Thickness", &uiSsrThickness, 0.01f, 1.0f, "%.3f");
            ImGui::SliderFloat("SSR Stride", &uiSsrStride, 0.01f, 0.5f, "%.3f");
            ImGui::SliderFloat("SSR Intensity", &uiSsrIntensity, 0.0f, 3.0f, "%.2f");
        }

        // Phase 4: Debug Views
        if (ImGui::CollapsingHeader("Debug Views"))
        {
            const char* debugViews[] = { "Final", "Albedo", "Normal", "PBR", "Depth" };
            ImGui::Combo("View", &uiDeferredDebugView, debugViews, 5);
        }
        ImGui::Separator();

        ImGui::Text("Camera: (%.1f, %.1f, %.1f)",
            camera.Position.x, camera.Position.y, camera.Position.z);

        // GPU Pass breakdown 
        if (ImGui::CollapsingHeader("GPU Pass Timings"))
        {
            const auto& timings = gpuProfiler.getTimings();
            for (const auto& t : timings)
            {
                ImGui::Text("  %s: %.3f ms", t.name.c_str(), t.durationMs);
            }
        }
    }
    ImGui::End();
}

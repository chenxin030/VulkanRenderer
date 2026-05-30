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
#include <stdexcept>

namespace fs = std::filesystem;

// ============================================================================
// Lifecycle
// ============================================================================

void VkrRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool VkrRenderer::initVulkan()
{
    // Sponza is a large atrium (~25m wide, ~15m tall). Start at a distance.
    camera = Camera(glm::vec3(0.0f, 8.0f, 25.0f));
    camera.MovementSpeed = 80.0f;
    return VulkanBase::initVulkan("VulkanRenderer - vkrEngine");
}

bool VkrRenderer::prepareResource()
{
    // ---- Load Sponza scene ----
    const std::string sponzaDir = std::string(VK_GLTF_SAMPLES_DIR) + "Models/Sponza/glTF/";
    const std::string sponzaPath = sponzaDir + "Sponza.gltf";

    std::cout << "[vkrEngine] Loading Sponza from: " << sponzaPath << std::endl;

    if (!sponzaModel.loadFromFile(sponzaPath, sponzaDir))
    {
        std::cerr << "[vkrEngine] Failed to load Sponza model!" << std::endl;
        return false;
    }

    // Add to scene
    scene.addModel(&sponzaModel, glm::mat4(1.0f), "Sponza");

    // ---- Create GPU resources ----
    if (!createModelGpuResources(sponzaModel)) return false;
    if (!createDummyWhiteTexture()) return false;

    // Create material textures
    for (auto& mat : sponzaModel.materials)
    {
        if (!createMaterialGpuResources(*mat, sponzaDir)) return false;
    }

    // ---- Descriptors ----
    if (!createDescriptors()) return false;

    // ---- Pipeline ----
    if (!createPipeline()) return false;

    // ---- Material descriptor sets ----
    if (!createMaterialDescriptorSets()) return false;

    // ---- UBO ----
    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));

    // ---- GPU Profiler ----
    auto props = physicalDevice.getProperties();
    gpuProfiler.init(device, static_cast<float>(props.limits.timestampPeriod));

    // ---- UI ----
    if (!initUI()) return false;

    // ---- Create per-swapchain-image semaphores (fixes Device Lost) ----
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
    // VulkanBase handles swapchain, command buffers, sync, depth, UI cleanup
    // vk::raii handles the rest via destructors
    std::cout << "[vkrEngine] Cleanup complete." << std::endl;
}

// ============================================================================
// GPU Resource Creation
// ============================================================================

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

bool VkrRenderer::loadMaterialTexture(VkrMaterialTexture& tex, const std::string& fullPath)
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
        vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, tempTex);

    // Transfer image to GPU
    transitionImageLayout(tempTex.textureImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(stagingBuf, tempTex.textureImage,
        static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    generateMipmaps(tempTex.textureImage, vk::Format::eR8G8B8A8Srgb,
        texWidth, texHeight, mipLevels);

    // Create image view
    tempTex.textureImageView = createImageView(tempTex.textureImage,
        vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);

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

    // Create sampler (shared across materials, but we create it elsewhere)
    // For now, load textures

    bool hasAnyTexture = false;

    if (!mat.baseColorTexturePath.empty())
    {
        std::string fullPath = textureBasePath + mat.baseColorTexturePath;
        if (loadMaterialTexture(mat.gpuBaseColorTexture, fullPath))
            hasAnyTexture = true;
    }

    if (!mat.normalTexturePath.empty())
    {
        std::string fullPath = textureBasePath + mat.normalTexturePath;
        loadMaterialTexture(mat.gpuNormalTexture, fullPath);
    }

    if (!mat.metallicRoughnessTexturePath.empty())
    {
        std::string fullPath = textureBasePath + mat.metallicRoughnessTexturePath;
        loadMaterialTexture(mat.gpuMetallicRoughnessTexture, fullPath);
    }

    mat.gpuResourcesCreated = true;
    return true;
}

// ============================================================================
// Descriptors
// ============================================================================

bool VkrRenderer::createDescriptors()
{
    try
    {
        // ---- Scene descriptor set layout (Set 0) ----
        {
            std::vector<vk::DescriptorSetLayoutBinding> bindings = {
                {.binding = 0,
                  .descriptorType = vk::DescriptorType::eUniformBuffer,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            };

            sceneDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
                vk::DescriptorSetLayoutCreateInfo{
                    .bindingCount = static_cast<uint32_t>(bindings.size()),
                    .pBindings = bindings.data() });
        }

        // ---- Material descriptor set layout (Set 1) — separate image + sampler ----
        {
            std::vector<vk::DescriptorSetLayoutBinding> bindings = {
                {.binding = 0,
                  .descriptorType = vk::DescriptorType::eSampledImage,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
                {.binding = 1,
                  .descriptorType = vk::DescriptorType::eSampler,
                  .descriptorCount = 1,
                  .stageFlags = vk::ShaderStageFlagBits::eFragment },
            };

            materialDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
                vk::DescriptorSetLayoutCreateInfo{
                    .bindingCount = static_cast<uint32_t>(bindings.size()),
                    .pBindings = bindings.data() });
        }

        // ---- Descriptor Pool ----
        {
            // We need: MAX_FRAMES_IN_FLIGHT scene sets + N material sets
            uint32_t maxSets = MAX_FRAMES_IN_FLIGHT + static_cast<uint32_t>(sponzaModel.materials.size()) + 1;

            std::vector<vk::DescriptorPoolSize> poolSizes = {
                { vk::DescriptorType::eUniformBuffer, maxSets },
                { vk::DescriptorType::eSampledImage, maxSets },
                { vk::DescriptorType::eSampler, maxSets },
            };

            descriptorPool = vk::raii::DescriptorPool(device,
                vk::DescriptorPoolCreateInfo{
                    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                    .maxSets = maxSets,
                    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                    .pPoolSizes = poolSizes.data() });
        }

        // ---- Allocate scene descriptor sets (one per frame) ----
        {
            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *sceneDescriptorSetLayout);

            // We need to allocate into sceneUboResources.descriptorSets
            // But sceneUboResources is a MeshBuffer managed by createUniformBuffers
            // Let's allocate manually
            vk::DescriptorSetAllocateInfo allocInfo{
                .descriptorPool = *descriptorPool,
                .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
                .pSetLayouts = layouts.data()
            };

            sceneUboResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);
        }

        // ---- Create shared sampler for materials ----
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
        for (uint32_t i = 0; i < matCount; ++i)
        {
            auto& mat = *sponzaModel.materials[i];

            vk::DescriptorImageInfo imageInfo{
                .imageView = (mat.gpuBaseColorTexture.imageView != vk::raii::ImageView(nullptr))
                    ? *mat.gpuBaseColorTexture.imageView
                    : *dummyWhiteTexture.textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            vk::DescriptorImageInfo samplerInfo{
                .sampler = *materialSampler,
            };

            std::array<vk::WriteDescriptorSet, 2> writes = { {
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &imageInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *materialDescriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eSampler,
                    .pImageInfo = &samplerInfo,
                },
            } };

            device.updateDescriptorSets(writes, nullptr);
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[vkrEngine] Failed to create material descriptor sets: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Pipeline
// ============================================================================

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

        // Vertex input (only pos/normal/texcoord — tangent not consumed in Phase 1)
        vk::VertexInputBindingDescription bindingDesc{ 0, 48, vk::VertexInputRate::eVertex };
        std::array<vk::VertexInputAttributeDescription, 3> attrDescs = { {
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, 0  },  // pos
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat, 12 },  // normal
            vk::VertexInputAttributeDescription{ 2, 0, vk::Format::eR32G32Sfloat,    24 },  // texCoord
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

// ============================================================================
// Rendering
// ============================================================================

void VkrRenderer::render()
{
    auto cpuStart = std::chrono::high_resolution_clock::now();

    // ---- Read back PREVIOUS frame's GPU timestamps ----
    gpuProfiler.endFrame();

    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to wait for fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX,
        *m_imageAcquiredSemaphores[currentFrame % m_imageAcquiredSemaphores.size()], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    // ---- Update UI ----
    vkrUpdateUIFrame(this);

    // ---- Update scene UBO ----
    updateSceneUBO(currentFrame);

    // ---- Record commands (rendering + ImGui) ----
    recordCommandBuffer(imageIndex);

    // ---- Submit (per-image semaphores) ----
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

    // ---- Present ----
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
        0.1f, 2000.0f);
    ubo.projection[1][1] *= -1.0f; // Vulkan inverted Y
    ubo.view = camera.GetViewMatrix();
    ubo.camPos = camera.Position;

    void* mapped = sceneUboResources.BuffersMapped[frameIndex];
    memcpy(mapped, &ubo, sizeof(SceneUBO));

    // Update descriptor set
    vk::DescriptorBufferInfo bufferInfo{
        .buffer = *sceneUboResources.Buffers[frameIndex],
        .offset = 0,
        .range = sizeof(SceneUBO),
    };

    vk::WriteDescriptorSet write{
        .dstSet = *sceneUboResources.descriptorSets[frameIndex],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &bufferInfo,
    };

    device.updateDescriptorSets(write, nullptr);

    // Also update UBO binding in the current material descriptor sets if needed
    // (Material sets only have textures, scene UBO is in set 0)
}

void VkrRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& cmd = commandBuffers[currentFrame];

    // ---- Begin command buffer ----
    cmd.begin(vk::CommandBufferBeginInfo{});

    // ---- GPU Profiler: begin frame ----
    gpuProfiler.beginFrame(*cmd);

    // ---- Transition swapchain image ----
    transition_image_layout(cmd,
        swapChainImages[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eNone,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);

    // ---- Transition depth ----
    transition_image_layout(cmd,
        depthData.textureImage,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal,
        vk::AccessFlagBits2::eNone,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests,
        vk::ImageAspectFlagBits::eDepth);

    // ---- Begin dynamic rendering ----
    vk::RenderingAttachmentInfo colorAttachment{
        .imageView = *swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue{ 0.02f, 0.02f, 0.04f, 1.0f },
    };

    vk::RenderingAttachmentInfo depthAttachment{
        .imageView = *depthData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = vk::ClearDepthStencilValue{ 1.0f, 0 },
    };

    vk::RenderingInfo renderingInfo{
        .renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment,
    };

    cmd.beginRendering(renderingInfo);

    // ---- Set dynamic state ----
    vk::Viewport viewport{
        .x = 0.0f, .y = 0.0f,
        .width = static_cast<float>(swapChainExtent.width),
        .height = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, vk::Rect2D{ .offset = { 0, 0 }, .extent = swapChainExtent });

    // ---- Bind pipeline ----
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    // ---- Render scene (with GPU profiling) ----
    statDrawCalls = 0;
    statTriangles = 0;

    gpuProfiler.beginPass(*cmd, "MainRender");
    for (const auto& renderable : scene.renderables())
    {
        renderModel(*cmd, *renderable.model, renderable.transform);
    }
    gpuProfiler.endPass(*cmd);

    // ---- End rendering ----
    cmd.endRendering();

    // ---- ImGui rendering (loads onto the rendered frame) ----
    vk::RenderingAttachmentInfo uiAttachment{
        .imageView = *swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    vk::RenderingInfo uiRenderingInfo{
        .renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachment,
    };
    cmd.beginRendering(uiRenderingInfo);
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, vk::Rect2D{ .offset = { 0, 0 }, .extent = swapChainExtent });
    vkrRecordUICmdBuffer(this, cmd, currentFrame);
    cmd.endRendering();

    // ---- Transition swapchain image for present ----
    transition_image_layout(cmd,
        swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);

    // ---- End command buffer ----
    cmd.end();
}

void VkrRenderer::renderModel(vk::CommandBuffer cmd, const VkrModel& model, const glm::mat4& transform)
{
    if (model.subMeshes.empty()) return;

    // Bind vertex & index buffers
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

        // Push constants: model matrix
        cmd.pushConstants<glm::mat4>(*pipelineLayout,
            vk::ShaderStageFlagBits::eVertex, 0, transform);

        // Draw — indices are already global (baseVertex added during loading)
        cmd.drawIndexed(sub.indexCount, 1, sub.firstIndex, 0, 0);

        ++statDrawCalls;
        statTriangles += sub.indexCount / 3;
    }
}

void VkrRenderer::recreateSwapChain()
{
    VulkanBase::recreateSwapChain();
    // Pipeline uses dynamic viewport/scissor, so no need to recreate it
}

// ============================================================================
// UI
// ============================================================================

bool VkrRenderer::initUI()
{
    return initVulkanUI();
}

void VkrRenderer::updateUIPanel()
{
    if (!showStatsWindow) return;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 240), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("vkrEngine Stats", &showStatsWindow,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing))
    {
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Separator();

        ImGui::Text("Scene: %s", sponzaModel.name.c_str());
        ImGui::Text("  Triangles: %u", statTriangles);
        ImGui::Text("  Draw Calls: %u", statDrawCalls);
        ImGui::Text("  Sub-meshes: %zu", sponzaModel.subMeshes.size());
        ImGui::Text("  Materials: %zu", sponzaModel.materials.size());
        ImGui::Separator();

        ImGui::Text("CPU Frame: %.2f ms", statCpuMs);
        ImGui::Text("GPU Total:  %.2f ms", gpuProfiler.totalGpuMs());
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

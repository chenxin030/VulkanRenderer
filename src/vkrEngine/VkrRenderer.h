#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>

#include "GpuProfiler.h"
#include "Scene.h"
#include "VkrModel.h"

struct VkrRenderer final : VulkanBase
{
public:
    // ---- Lifecycle ----
    void initialize(Platform* _platform);
    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

    // ---- Override for ImGui ----
    void updateUIPanel() override;
    void recreateSwapChain() override;

    // ---- Scene access ----
    VkrScene scene;
    VkrModel  sponzaModel;

private:
    // ---- Constants ----
    static constexpr uint32_t MAX_MATERIALS = 128;
    static constexpr uint32_t CASCADE_COUNT = 4u;
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048u;
    static constexpr float    CAMERA_NEAR = 0.1f;
    static constexpr float    CAMERA_FAR = 2000.0f;

    // ---- Scene UBO (Set 0, Binding 0) ----
    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 camPos;
        float     pad0;
        glm::vec3 lightDir;     // directional light direction (normalized)
        float     pad1;
        glm::vec3 lightColor;   // directional light color
        float     pad2;
    };

    // ---- Post-processing Params UBO (Set 0, Binding 4) ----
    struct ParamsUBO
    {
        float     exposure = 4.5f;
        float     gamma = 2.2f;
        uint32_t  lightingMode = 1;    // 0=Phase1 Simple, 1=PBR+IBL
        uint32_t  enableDirLight = 1;    // 0=off, 1=on
    };

    // ---- CSM UBO (Set 0, Binding 5) ----
    struct CsmUBO
    {
        glm::mat4 cascadeViewProj[CASCADE_COUNT];
        glm::vec4 cascadeSplitDepths;   // x=split1, y=split2, z=split3, w=far
    };

    // ---- Per-material UBO (Set 1, Binding 4) ----
    struct MaterialUBO
    {
        glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
        float     metallicFactor = 1.0f;
        float     roughnessFactor = 1.0f;
        float     normalScale = 1.0f;
        float     occlusionStrength = 1.0f;
        glm::vec3 emissiveFactor{ 0.0f, 0.0f, 0.0f };
        float     alphaCutoff = 0.5f;
        uint32_t  flags = 0; // bit0: hasBaseColorTex, bit1: hasNormalTex, bit2: hasMetallicRoughnessTex
        uint32_t  pad0 = 0;
        uint32_t  pad1 = 0;
    };

    // ---- GPU Resources (ordered for correct RAII destruction) ----

    // Descriptor pool — FIRST: destroyed last, after all descriptor sets
    vk::raii::DescriptorPool descriptorPool = nullptr;

    // Descriptor set layouts
    vk::raii::DescriptorSetLayout sceneDescriptorSetLayout = nullptr; // Set 0: scene-level
    vk::raii::DescriptorSetLayout materialDescriptorSetLayout = nullptr; // Set 1: per-material

    // Descriptor sets
    MeshBuffer sceneUboResources;                    // Set 0 descriptors + UBO (per frame)
    MeshBuffer paramsUboResources;                   // Params UBO (per frame)
    MeshBuffer csmUboResources;                      // CSM UBO (per frame)
    MeshBuffer shadowParamsUboResources;              // Shadow filter params (per frame)
    std::vector<vk::raii::DescriptorSet> materialDescriptorSets; // Set 1 (per material)
    std::vector<MeshBuffer> materialUboResources;    // Material UBO (per material)

    // IBL resources
    TextureData hdrEquirectData;
    TextureData envCubemapData;
    TextureData irradianceCubemapData;
    TextureData prefilteredEnvMapData;
    TextureData brdfLutData;

    // CSM resources — double-buffered (2 copies) to avoid
    // shadow map read/write races between overlapping frames
    TextureData csmTextureArrays[2];
    std::vector<vk::raii::Sampler>   csmSamplers;
    std::vector<vk::raii::ImageView> csmArrayViews;
    std::vector<vk::raii::ImageView> csmLayerViewsArray[2];
    vk::ImageLayout csmArrayLayouts[2] = {
        vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined };

    // CSM depth pipeline (separate from main pipeline)
    vk::raii::DescriptorSetLayout csmDepthDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool      csmDepthDescriptorPool = nullptr;
    vk::raii::DescriptorSets      csmDepthDescriptorSets = nullptr;
    vk::raii::PipelineLayout      csmDepthPipelineLayout = nullptr;
    vk::raii::Pipeline            csmDepthPipeline = nullptr;

    // IBL generation helpers
    void generateIBLResources();
    void transitionImageLayoutCmd(
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
        vk::AccessFlags dstAccessMask);

    // Pipeline
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline       pipeline = nullptr;

    // Per-swapchain-image semaphores (fixes Device Lost from semaphore reuse)
    std::vector<vk::raii::Semaphore> m_imageAcquiredSemaphores;
    std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;

    // Other GPU resources
    VkrGpuProfiler gpuProfiler;
    vk::raii::Sampler materialSampler = nullptr;
    TextureData dummyWhiteTexture;
    TextureData dummyNormalTexture;   // default (0.5, 0.5, 1.0) = flat normal

    // ---- Stats (updated each frame) ----
    uint32_t statDrawCalls = 0;
    uint32_t statTriangles = 0;
    float    statCpuMs = 0.0f;
    bool     showStatsWindow = true;
    bool     showProfilerWindow = true;

    // ---- UI-tweakable params ----
    float    uiExposure = 4.5f;
    float    uiGamma = 2.2f;
    int      uiLightingMode = 1; // 0=Phase1 Simple, 1=PBR+IBL
    bool     uiEnableDirectionalLight = true;

    // ---- CSM tweakable params ----
    float    uiSplitLambda = 0.45f;  // 0=linear, 1=log; 0.45 balances near/far for Sponza
    int      uiShadowFilterMode = 2;  // 0=Hard, 1=PCF, 2=PCSS
    float    uiPcfRadiusTexels = 2.0f;
    float    uiPcssLightSizeTexels = 25.0f;
    bool     uiVisualizeCascades = false;
    bool     uiRotateLight = false;

    // ---- Cascade Data (CPU-side) ----
    float     cascadeSplitDepths[CASCADE_COUNT + 1] = {}; // [0]=near, [1]=split1, ..., [4]=far
    glm::mat4 cascadeViewProj[CASCADE_COUNT] = {};

    // ---- Internal Methods ----

    /** Create GPU buffers for a model (vertex + index buffer). */
    bool createModelGpuResources(VkrModel& model);

    /** Create GPU resources (image, image view, sampler) for a material's textures. */
    bool createMaterialGpuResources(VkrMaterial& mat, const std::string& textureBasePath);

    /** Create descriptor set layouts and pool. */
    bool createDescriptors();

    /** Create per-material descriptor sets and UBOs. */
    bool createMaterialDescriptorSets();

    /** Create the graphics pipeline. */
    bool createPipeline();

    /** Update scene UBO and params for the current frame. */
    void updateSceneUBO(uint32_t frameIndex);

    /** Update per-material UBOs. */
    void updateMaterialUBOs();

    /** Main command buffer recording. */
    void recordCommandBuffer(uint32_t imageIndex);

    /** Render a single model (all sub-meshes). */
    void renderModel(vk::CommandBuffer cmd, const VkrModel& model, const glm::mat4& transform);

    /** Initialize ImGui. */
    bool initUI();

    /** Load a texture from file into the TextureCache-like system. */
    bool loadMaterialTexture(VkrMaterialTexture& tex, const std::string& fullPath,
        vk::Format format = vk::Format::eR8G8B8A8Srgb);

    /** Create a default 1x1 white texture for fallback. */
    bool createDummyWhiteTexture();

    /** Create a default 1x1 normal texture for fallback ((0.5, 0.5, 1.0) = flat). */
    bool createDummyNormalTexture();

    // ---- CSM Methods ----
    bool createCsmResources();
    void createCsmDescriptorSets();
    bool createCsmDescriptorPool();
    bool createCsmDepthPipeline();
    void updateCsmBuffers(uint32_t frameIndex);
    void calculateCascadeSplits();
    void computeCascadeViewProj(uint32_t cascadeIndex, const glm::vec3& lightDir);
};

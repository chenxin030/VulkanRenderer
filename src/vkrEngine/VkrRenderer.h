#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>

#include "GpuProfiler.h"
#include "Scene.h"
#include "VkrModel.h"

struct VkrRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);
    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

    void updateUIPanel() override;
    void recreateSwapChain() override;

    VkrScene scene;
    VkrModel  sponzaModel;

private:
    static constexpr uint32_t MAX_MATERIALS = 128;
    static constexpr uint32_t CASCADE_COUNT = 4u;
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048u;
    static constexpr uint32_t SSAO_KERNEL_SIZE = 16u;
    static constexpr uint32_t SSAO_NOISE_SIZE = 4u;
    static constexpr float    CAMERA_NEAR = 0.1f;
    static constexpr float    CAMERA_FAR = 2000.0f;

    // Scene UBO (Set 0, Binding 0) — expanded for deferred rendering
    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 invProjection;
        glm::mat4 invView;
        glm::vec4 camPos;           // xyz=camera position, w=near
        glm::vec4 lightDir;         // xyz=directional light direction (normalized), w=far
        glm::vec4 lightColor;       // xyz=directional light color, w=unused
    };

    // Shadow filter parameters (Set 0, Binding 7)
    struct ShadowParamsUBO
    {
        int32_t  shadowFilterMode = 2;    // 0=Hard, 1=PCF, 2=PCSS
        float    pcfRadiusTexels = 2.0f;
        float    pcssLightSizeTexels = 25.0f;
        float    shadowBiasMin = 0.0006f;
        glm::vec2 invShadowMapSize{ 1.0f / 2048.0f, 1.0f / 2048.0f };
        glm::vec2 padding0{ 0.0f, 0.0f };
    };

    // Post-processing Params UBO (Set 0, Binding 4)
    struct ParamsUBO
    {
        float     exposure = 4.5f;
        float     gamma = 2.2f;
        uint32_t  lightingMode = 1;    // 0=Phase1 Simple, 1=PBR+IBL
        uint32_t  enableDirLight = 1;    // 0=off, 1=on
    };

    // CSM UBO (Set 0, Binding 5)
    struct CsmUBO
    {
        glm::mat4 cascadeViewProj[CASCADE_COUNT];
        glm::vec4 cascadeSplitDepths;   // x=split1, y=split2, z=split3, w=far
    };

    //Per-material UBO (Set 1, Binding 4)
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

    // GBuffer attachment helper
    struct GBufferAttachment
    {
        TextureData texture;
        vk::Format format = vk::Format::eUndefined;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };

    // ===================================================================
    // Phase 4: Deferred Settings UBO (Set 2, Binding 0)
    // ===================================================================
    struct DeferredSettingsUBO
    {
        float     ssaoEnabled = 1.0f;
        float     ssrEnabled = 1.0f;
        float     debugView = 0.0f;  // 0=final, 1=albedo, 2=normal, 3=pbr, 4=depth
        float     pad0 = 0.0f;
    };

    // ===================================================================
    // Phase 4: SSAO Settings UBO (Set 0 for SSAO pass)
    // ===================================================================
    struct SSAOSettingsUBO
    {
        glm::mat4 invProjection;
        glm::mat4 invView;
        glm::mat4 projection;    // for view→NDC forward projection of samples
        glm::vec4 params0;       // x: radius, y: bias, z: intensity, w: enabled
        glm::ivec4 flags;         // x: debugView
        int32_t   pad1 = 0;
        int32_t   pad2 = 0;
        int32_t   pad3 = 0;
        glm::vec4 kernel[SSAO_KERNEL_SIZE];
    };

    // ===================================================================
    // Phase 4: SSAO Blur UBO
    // ===================================================================
    struct BlurUBO
    {
        glm::vec2 texelSize;
        glm::vec2 pad0;
    };

    // ===================================================================
    // Phase 4: SSR Params UBO
    // ===================================================================
    struct SSRParams
    {
        float     maxRayDistance = 16.0f;
        float     thickness = 0.12f;
        float     stride = 0.05f;
        float     intensity = 1.0f;
        glm::vec2 invResolution{ 1.0f / 1920.0f, 1.0f / 1080.0f };
        int32_t   debugMode = 0;  // 0=final, 1=hit/miss, 2=hitSteps, 3=depth, 4=normal
        int32_t   maxSteps = 85;
        float     padding0 = 0.0f;
    };

    // ===================================================================
    // Forward pipeline resources (Set 0: scene-level descriptors)
    // ===================================================================
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSetLayout sceneDescriptorSetLayout = nullptr; // Set 0: scene-level
    vk::raii::DescriptorSetLayout materialDescriptorSetLayout = nullptr; // Set 1: per-material
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

    vk::raii::DescriptorSetLayout csmDepthDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool      csmDepthDescriptorPool = nullptr;
    vk::raii::DescriptorSets      csmDepthDescriptorSets = nullptr;
    vk::raii::PipelineLayout      csmDepthPipelineLayout = nullptr;
    vk::raii::Pipeline            csmDepthPipeline = nullptr;

    // ===================================================================
    // Phase 4: GBuffer resources
    // ===================================================================
    GBufferAttachment gbufferAlbedo;
    GBufferAttachment gbufferNormalRoughness;
    GBufferAttachment gbufferPbr;
    GBufferAttachment gbufferDepth;

    vk::raii::Sampler                 gbufferSampler = nullptr;
    vk::raii::DescriptorSetLayout     gbufferDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          gbufferDescriptorPool = nullptr;
    vk::raii::PipelineLayout          gbufferPipelineLayout = nullptr;
    vk::raii::Pipeline                gbufferPipeline = nullptr;

    // ===================================================================
    // Phase 4: SSAO resources
    // ===================================================================
    GBufferAttachment                 ssaoColor;
    GBufferAttachment                 ssaoBlurColor;

    MeshBuffer                        ssaoSettingsUboResources;
    MeshBuffer                        ssaoBlurUboResources;

    vk::raii::DescriptorSetLayout     ssaoDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          ssaoDescriptorPool = nullptr;
    vk::raii::PipelineLayout          ssaoPipelineLayout = nullptr;
    vk::raii::Pipeline                ssaoPipeline = nullptr;

    vk::raii::DescriptorSetLayout     ssaoBlurDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          ssaoBlurDescriptorPool = nullptr;
    vk::raii::PipelineLayout          ssaoBlurPipelineLayout = nullptr;
    vk::raii::Pipeline                ssaoBlurPipeline = nullptr;

    TextureData                       ssaoNoiseTexture;
    vk::raii::Sampler                 ssaoNoiseSampler = nullptr;
    std::vector<glm::vec4>            ssaoKernel;

    // ===================================================================
    // Phase 4: SSR resources
    // ===================================================================
    GBufferAttachment                 ssrColor;
    MeshBuffer                        ssrParamsUboResources;

    vk::raii::DescriptorSetLayout     ssrDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          ssrDescriptorPool = nullptr;
    vk::raii::PipelineLayout          ssrPipelineLayout = nullptr;
    vk::raii::Pipeline                ssrPipeline = nullptr;

    // ===================================================================
    // Phase 4: Deferred Lighting resources
    // ===================================================================
    vk::raii::DescriptorSetLayout     deferredDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          deferredDescriptorPool = nullptr;
    vk::raii::PipelineLayout          deferredPipelineLayout = nullptr;
    vk::raii::Pipeline                deferredPipeline = nullptr;
    MeshBuffer                        deferredSettingsUboResources;

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

    // Stats (updated each frame)
    uint32_t statDrawCalls = 0;
    uint32_t statTriangles = 0;
    float    statCpuMs = 0.0f;
    bool     showStatsWindow = true;
    bool     showProfilerWindow = true;

    // UI-tweakable params
    float    uiExposure = 4.5f;
    float    uiGamma = 2.2f;
    int      uiLightingMode = 1; // 0=Phase1 Simple, 1=PBR+IBL
    bool     uiEnableDirectionalLight = true;

    // CSM tweakable params
    float    uiSplitLambda = 0.45f;
    int      uiShadowFilterMode = 2;  // 0=Hard, 1=PCF, 2=PCSS
    float    uiPcfRadiusTexels = 2.0f;
    float    uiPcssLightSizeTexels = 25.0f;
    bool     uiRotateLight = false;

    // Phase 4: SSAO tweakable params
    bool     uiSsaoEnabled = true;
    float    uiSsaoRadius = 0.5f;
    float    uiSsaoBias = 0.025f;
    float    uiSsaoIntensity = 1.5f;
    int      uiSsaoDebugView = 0;  // 0=final, 1=depth, 2=normal, 3=noise, 4=hits, 5=range, 6=viewZ

    // Phase 4: SSR tweakable params
    bool     uiSsrEnabled = true;
    int      uiSsrMaxSteps = 85;
    float    uiSsrMaxRayDistance = 16.0f;
    float    uiSsrThickness = 0.12f;
    float    uiSsrStride = 0.05f;
    float    uiSsrIntensity = 1.0f;

    // Phase 4: Debug view
    int      uiDeferredDebugView = 0; // 0=final, 1=albedo, 2=normal, 3=pbr, 4=depth

    // Cascade Data (CPU-side) 
    float     cascadeSplitDepths[CASCADE_COUNT + 1] = {}; // [0]=near, [1]=split1, ..., [4]=far
    glm::mat4 cascadeViewProj[CASCADE_COUNT] = {};

    // Create GPU buffers for a model (vertex + index buffer).
    bool createModelGpuResources(VkrModel& model);

    // Create GPU resources (image, image view, sampler) for a material's textures
    bool createMaterialGpuResources(VkrMaterial& mat, const std::string& textureBasePath);

    bool createDescriptors();
    bool createMaterialDescriptorSets();
    bool createPipeline();
    void updateSceneUBO(uint32_t frameIndex);
    void updateMaterialUBOs();
    void recordCommandBuffer(uint32_t imageIndex);

    void renderModel(vk::CommandBuffer cmd, const VkrModel& model, const glm::mat4& transform);

    bool initUI();

    bool loadMaterialTexture(VkrMaterialTexture& tex, const std::string& fullPath,
        vk::Format format = vk::Format::eR8G8B8A8Srgb);

    // Create a default 1x1 white texture for fallback
    bool createDummyWhiteTexture();

    // Create a default 1x1 normal texture for fallback ((0.5, 0.5, 1.0) = flat)
    bool createDummyNormalTexture();

    // CSM Methods
    bool createCsmResources();
    void createCsmDescriptorSets();
    bool createCsmDescriptorPool();
    bool createCsmDepthPipeline();
    void updateCsmBuffers(uint32_t frameIndex);
    void calculateCascadeSplits();
    void computeCascadeViewProj(uint32_t cascadeIndex, const glm::vec3& lightDir);

    // Phase 4: GBuffer Methods
    bool createGBufferResources();
    void destroyGBufferResources();
    bool createGBufferDescriptorSetLayout();
    bool createGBufferDescriptorPool();
    void createGBufferDescriptorSets();
    bool createGBufferPipeline();

    // Phase 4: SSAO Methods
    bool createSSAOResources();
    void destroySSAOResources();
    void generateSsaoKernel();
    void createSsaoNoiseTexture();
    bool createSsaoDescriptorSetLayout();
    bool createSsaoDescriptorPool();
    void createSsaoDescriptorSets();
    bool createSsaoPipeline();
    bool createSsaoBlurDescriptorSetLayout();
    bool createSsaoBlurDescriptorPool();
    void createSsaoBlurDescriptorSets();
    bool createSsaoBlurPipeline();
    void updateSsaoBuffers(uint32_t frameIndex);
    void updateSsaoBlurBuffers(uint32_t frameIndex);

    // Phase 4: SSR Methods
    bool createSSRResources();
    void destroySSRResources();
    bool createSsrDescriptorSetLayout();
    bool createSsrDescriptorPool();
    void createSsrDescriptorSets();
    bool createSsrPipeline();
    void updateSsrBuffers(uint32_t frameIndex);

    // Phase 4: Deferred Lighting Methods
    bool createDeferredLightingDescriptorSetLayout();
    bool createDeferredLightingDescriptorPool();
    void createDeferredLightingDescriptorSets();
    bool createDeferredLightingPipeline();
    void updateDeferredSettingsBuffer(uint32_t frameIndex);

    // Phase 4: Recreate sized resources on swapchain resize
    bool recreateDeferredSizedResources();
};

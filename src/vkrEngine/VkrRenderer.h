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

    // Phase 4: Deferred Settings UBO (Set 2, Binding 0)
    struct DeferredSettingsUBO
    {
        float     ssrEnabled = 1.0f;
        float     pad0 = 0.0f;
        float     pad1 = 0.0f;
        float     pad2 = 0.0f;
    };

    // Phase 5: GPU Occlusion Culling — Per-submesh instance data
    struct CullingInstanceUBO
    {
        glm::vec4 aabbMin;     // xyz=min, w=unused
        glm::vec4 aabbMax;     // xyz=max, w=unused
        glm::ivec4 drawInfo;   // x=indexCount, y=firstIndex, z=materialIndex, w=unused
    };

    struct CullingParamsUBO
    {
        glm::vec4 frustumPlanes[6];
        glm::vec4 hiZInfo;     // x=width, y=height, z=mipCount, w=unused
        uint32_t  totalInstances;
        uint32_t  enabled;
        float     pad0;
        float     pad1;
    };

    // Phase 5: Clustered Shading
    static constexpr uint32_t CLUSTER_X = 16;
    static constexpr uint32_t CLUSTER_Y = 9;
    static constexpr uint32_t CLUSTER_Z = 24;
    static constexpr uint32_t MAX_CLUSTER_LIGHTS = 2048;
    static constexpr uint32_t MAX_LIGHTS_PER_CLUSTER = 64;

    struct GpuPointLight
    {
        glm::vec4 position;   // xyz=position, w=unused
        glm::vec4 color;      // rgb=color, w=intensity
    };

    struct ClusterParamsUBO
    {
        uint32_t  clusterX;
        uint32_t  clusterY;
        uint32_t  clusterZ;
        uint32_t  totalClusters;
        glm::vec3 tileSize;
        float     pad0;
        glm::vec3 cameraPos;
        float     nearZ;
        float     farZ;
        float     zMin;
        float     zMax;
        float     clusteredEnabled;
        float     visualizeClusters;
        float     pad2;
        float     pad3;
    };

    struct LightGridCell
    {
        uint32_t offset;
        uint32_t count;
    };

    // Phase 4: SSR Params UBO
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

    
    // Forward pipeline resources (Set 0: scene-level descriptors)
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

    
    // Phase 4: GBuffer resources
    GBufferAttachment gbufferAlbedo;
    GBufferAttachment gbufferNormalRoughness;
    GBufferAttachment gbufferPbr;
    GBufferAttachment gbufferDepth;

    vk::raii::Sampler                 gbufferSampler = nullptr;
    vk::raii::DescriptorSetLayout     gbufferDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          gbufferDescriptorPool = nullptr;
    vk::raii::PipelineLayout          gbufferPipelineLayout = nullptr;
    vk::raii::Pipeline                gbufferPipeline = nullptr;

    
    // Phase 4: SSR resources
    GBufferAttachment                 ssrColor;
    MeshBuffer                        ssrParamsUboResources;

    vk::raii::DescriptorSetLayout     ssrDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          ssrDescriptorPool = nullptr;
    vk::raii::PipelineLayout          ssrPipelineLayout = nullptr;
    vk::raii::Pipeline                ssrPipeline = nullptr;

    
    // Phase 4: Deferred Lighting resources
    vk::raii::DescriptorSetLayout     deferredDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          deferredDescriptorPool = nullptr;
    vk::raii::PipelineLayout          deferredPipelineLayout = nullptr;
    vk::raii::Pipeline                deferredPipeline = nullptr;
    MeshBuffer                        deferredSettingsUboResources;

    
    // Phase 5: GPU Occlusion Culling (Hi-Z) resources
    TextureData                       hizTexture;
    std::vector<vk::raii::ImageView>  hizMipViews;
    uint32_t                          hizMipCount = 1;
    vk::ImageLayout                   hizLayout = vk::ImageLayout::eUndefined;

    vk::raii::Sampler                 hizSampler = nullptr;

    vk::raii::DescriptorSetLayout     hizBuildDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          hizBuildDescriptorPool = nullptr;
    vk::raii::DescriptorSets          hizBuildDescriptorSets = nullptr;
    vk::raii::PipelineLayout          hizBuildPipelineLayout = nullptr;
    vk::raii::Pipeline                hizBuildPipeline = nullptr;

    // Culling storage buffers (per-frame)
    MeshBuffer                        cullingInstanceUboResources;    // per-submesh AABB data
    MeshBuffer                        cullingVisibleBufferResources;  // visibility results
    MeshBuffer                        cullingParamsUboResources;      // culling params UBO
    uint32_t                          cullingTotalInstances = 0;

    vk::raii::Buffer                  cullingVisibleReadback = nullptr;
    vk::raii::DeviceMemory            cullingVisibleReadbackMemory = nullptr;
    void* cullingVisibleReadbackMapped = nullptr;

    vk::raii::DescriptorSetLayout     cullingCompDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          cullingCompDescriptorPool = nullptr;
    vk::raii::DescriptorSets          cullingCompDescriptorSets = nullptr;
    vk::raii::PipelineLayout          cullingCompPipelineLayout = nullptr;
    vk::raii::Pipeline                cullingCompPipeline = nullptr;

    vk::raii::CommandPool             cullingCommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> cullingCommandBuffers;
    std::vector<vk::raii::Fence>      cullingFences;

    // CPU-side visibility cache (updated from readback, used next frame)
    std::vector<uint32_t>             submeshVisibility;
    uint32_t                          statCulledSubmeshes = 0;

    
    // Phase 5: Clustered Shading resources
    std::vector<GpuPointLight>        clusterSceneLights;

    MeshBuffer                        clusterLightBufferResources;     // light SSBO (per frame)
    MeshBuffer                        clusterParamsUboResources2;      // cluster params UBO (per frame)

    vk::raii::Buffer                  clusterLightGridBuffer = nullptr;
    vk::raii::DeviceMemory            clusterLightGridMemory = nullptr;
    void* clusterLightGridMapped = nullptr;

    vk::raii::Buffer                  clusterLightIndexBuffer = nullptr;
    vk::raii::DeviceMemory            clusterLightIndexMemory = nullptr;
    void* clusterLightIndexMapped = nullptr;

    vk::raii::Buffer                  clusterLightGridReadback = nullptr;
    vk::raii::DeviceMemory            clusterLightGridReadbackMemory = nullptr;
    void* clusterLightGridReadbackMapped = nullptr;

    vk::raii::DescriptorSetLayout     clusterComputeDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          clusterComputeDescriptorPool = nullptr;
    vk::raii::DescriptorSets          clusterComputeDescriptorSets = nullptr;
    vk::raii::PipelineLayout          clusterComputePipelineLayout = nullptr;
    vk::raii::Pipeline                clusterComputePipeline = nullptr;

    vk::raii::CommandPool             clusterCommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> clusterCommandBuffers;
    std::vector<vk::raii::Fence>      clusterFences;

    // Additional deferred descriptor set for cluster light resources (Set 2)
    vk::raii::DescriptorSetLayout     deferredClusterDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool          deferredClusterDescriptorPool = nullptr;
    vk::raii::DescriptorSets          deferredClusterDescriptorSets = nullptr;

    uint32_t                          clusterAvgLightsPerCluster = 0;

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

    // Phase 4: SSR tweakable params
    bool     uiSsrEnabled = true;
    int      uiSsrMaxSteps = 85;
    float    uiSsrMaxRayDistance = 16.0f;
    float    uiSsrThickness = 0.12f;
    float    uiSsrStride = 0.05f;
    float    uiSsrIntensity = 1.0f;

    // Phase 5: GPU culling tweakable params
    bool     uiCullingEnabled = true;

    // Phase 5: Clustered shading tweakable params
    bool     uiClusteredShadingEnabled = true;
    bool     uiVisualizeClusters = false;

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

    // Phase 5: GPU Occlusion Culling Methods
    bool createHiZResources();
    bool createHiZBuildDescriptorSetLayout();
    bool createHiZBuildDescriptorPool();
    void createHiZBuildDescriptorSets();
    bool createHiZBuildPipeline();
    bool createCullingComputeDescriptorSetLayout();
    bool createCullingComputeDescriptorPool();
    void createCullingComputeDescriptorSets();
    bool createCullingComputePipeline();
    bool createCullingBuffers();
    bool createCullingCommandPool();
    bool createCullingCommandBuffers();
    bool createCullingSyncObjects();
    void buildCullingInstanceData();
    void updateCullingBuffers(uint32_t frameIndex);
    void recordHiZBuildCommand(vk::CommandBuffer cmd);
    void recordOcclusionCullCommand(uint32_t frameIndex);
    void readbackCullingResults(uint32_t frameIndex);
    void extractFrustumPlanes(const glm::mat4& vp, glm::vec4* planesOut);

    void generateClusterSceneLights();
    bool createClusterBuffers();
    bool createClusterComputeDescriptorSetLayout();
    bool createClusterComputeDescriptorPool();
    void createClusterComputeDescriptorSets();
    bool createClusterComputePipeline();
    bool createDeferredClusterDescriptorSetLayout();
    bool createDeferredClusterDescriptorPool();
    void createDeferredClusterDescriptorSets();
    bool createClusterCommandPool();
    bool createClusterCommandBuffers();
    bool createClusterSyncObjects();
    void updateClusterBuffers(uint32_t frameIndex);
    void recordClusterComputeCommand(uint32_t frameIndex);
    void readbackClusterStats(uint32_t frameIndex);
    uint32_t getTotalClusters() const { return CLUSTER_X * CLUSTER_Y * CLUSTER_Z; }
    uint32_t getLightIndexBufferSize() const { return getTotalClusters() * MAX_LIGHTS_PER_CLUSTER; }
};

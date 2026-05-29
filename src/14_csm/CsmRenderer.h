#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>

struct CsmRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);
    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    void recreateSwapChain() override;

    // ---- Constants ----
    static constexpr uint32_t CASCADE_COUNT = 4u;
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048u;

    // ---- Scene Geometry ----
    Mesh cubeMesh;
    Mesh planeMesh;
    uint32_t instanceCount = 0;

    // ---- UBO / SSBO ----
    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 camPos;
        float pad0;
    };
    MeshBuffer sceneUboResources;

    // CSM UBO: 4 cascade VP matrices + 4 split depths
    struct CsmUBO
    {
        glm::mat4 cascadeViewProj[CASCADE_COUNT];
        glm::vec4 cascadeSplitDepths;   // [0]=split1, [1]=split2, [2]=split3, [3]=far
    };
    MeshBuffer csmUboResources;

    struct InstanceData
    {
        glm::mat4 model;
        glm::vec4 color;
    };
    MeshBuffer instanceBufferResources;

    MeshBuffer shadowParamsUboResources;

    // ---- Cascade Data (CPU-side) ----
    float cascadeSplitDepths[CASCADE_COUNT + 1] = {}; // [0]=near, [1]=split1, ..., [4]=far
    glm::mat4 cascadeViewProj[CASCADE_COUNT] = {};

    // ---- Shadow Map: Texture Array ----
    TextureData csmTextureArray;
    vk::raii::Sampler csmSampler = nullptr;
    // Per-layer image views for depth-only rendering into each cascade
    std::vector<vk::raii::ImageView> csmLayerViews;
    // Full-array image view for shader sampling
    vk::raii::ImageView csmArrayView = nullptr;
    vk::ImageLayout csmArrayLayout = vk::ImageLayout::eUndefined;

    // ---- Pipeline ----
    vk::raii::DescriptorSetLayout csmDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool      csmDescriptorPool = nullptr;
    vk::raii::DescriptorSets      csmDescriptorSets = nullptr;

    vk::raii::PipelineLayout csmDepthPipelineLayout = nullptr;
    vk::raii::PipelineLayout csmLitPipelineLayout = nullptr;
    vk::raii::Pipeline       csmDepthPipeline = nullptr;
    vk::raii::Pipeline       csmLitPipeline = nullptr;

    // ---- UI State ----
    float splitLambda = 0.75f;
    bool visualizeCascades = false;
    float lightAngle = 0.0f;
    float dirLightIntensity = 0.5f;
    int shadowFilterMode = 2;    // 0: hard, 1: pcf, 2: pcss
    float pcfRadiusTexels = 2.0f;
    float pcssLightSizeTexels = 25.0f;

    // ---- Methods ----
    void createCsmBuffers();
    bool createCsmDescriptorSetLayout();
    bool createCsmDescriptorPool();
    void createCsmDescriptorSets();
    bool createCsmMapResources();
    bool createCsmPipelines();

    void calculateCascadeSplits();
    void computeCascadeViewProj(uint32_t cascadeIndex, const glm::vec3& lightDir);
    void updateCsmBuffers(uint32_t frameIndex);

    bool initUI();
    void updateUIPanel() override;

    void recordCommandBuffer(uint32_t imageIndex);
};

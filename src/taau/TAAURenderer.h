#pragma once

#include <Base/VulkanBase.h>

struct TAAURenderer final : VulkanBase
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

    struct TAAUParamsUBO
    {
        float blendFactor;
        float reactiveClamp;
        float antiFlicker;
        float velocityScale;
        float historyClampGamma;
        float historyRejectThreshold;
        float pad0;
        float pad1;
    };

    vk::raii::DescriptorSetLayout shadowDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool shadowDescriptorPool = nullptr;
    vk::raii::PipelineLayout shadowPipelineLayout = nullptr;
    vk::raii::Pipeline shadowDepthPipeline = nullptr;
    vk::raii::Pipeline shadowLitPipeline = nullptr;

    Mesh cubeMesh;
    uint32_t instanceCount = 0;

    MeshBuffer shadowInstanceBufferResources;
    MeshBuffer sceneUboResources;
    MeshBuffer shadowUboResources;
    MeshBuffer shadowParamsUboResources;

    TextureData shadowMapData;
    vk::Extent2D shadowMapExtent{ 2048u, 2048u };
    vk::ImageLayout shadowMapLayout = vk::ImageLayout::eUndefined;

    int shadowFilterMode = 2;
    float pcfRadiusTexels = 2.0f;
    float pcssLightSizeTexels = 25.0f;

    float dirLightIntensity = 0.5f;
    float pointLightIntensity = 3.5f;
    float areaLightIntensity = 2.5f;

    vk::raii::DescriptorSetLayout taauDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool taauDescriptorPool = nullptr;
    vk::raii::PipelineLayout taauPipelineLayout = nullptr;
    vk::raii::Pipeline taauPipeline = nullptr;
    vk::raii::DescriptorSets taauDescriptorSets = nullptr;

    MeshBuffer taauParamsUboResources;
    TextureData taauInputColorData;
    TextureData taauVelocityData;
    TextureData taauDepthData;
    TextureData taauHistoryColorData[2];

    vk::ImageLayout taauInputLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout taauVelocityLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout taauDepthLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout taauHistoryLayouts[2] = { vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined };

    vk::raii::Sampler taauColorSampler = nullptr;
    vk::raii::Sampler taauDepthSampler = nullptr;

    uint32_t taauHistoryReadIndex = 0;
    bool taauHistoryValid = false;
    bool taauEnabled = true;
    glm::mat4 taauPrevViewProj = glm::mat4(1.0f);
    glm::vec2 taauJitterCurrent = glm::vec2(0.0f);
    glm::vec2 taauJitterPrev = glm::vec2(0.0f);
    uint64_t taauFrameCounter = 0;
    float taauRenderScale = 0.85f;

    struct TAAUParams
    {
        float blendFactor = 0.90f;
        float reactiveClamp = 0.55f;
        float antiFlicker = 0.88f;
        float velocityScale = 1.00f;
        float historyClampGamma = 1.15f;
        float historyRejectThreshold = 0.22f;
    };

    float taauSceneTime = 0.0f;
    bool taauFreezeHistory = false;
    TAAUParams taauParams{};

    bool createShadowDescriptorSetLayout();
    bool createShadowDescriptorPool();
    void createShadowDescriptorSets();
    void createShadowBuffers();
    void updateShadowBuffers(uint32_t currentImage);
    bool createShadowMapResources();
    bool createShadowPipelines();

    bool createTAAUResources();
    bool createTAAUDescriptorSetLayout();
    bool createTAAUDescriptorPool();
    void createTAAUDescriptorSets();
    void updateTAAUDescriptorSet(uint32_t frameIndex, uint32_t historyReadIndex);
    bool createTAAUPipeline();
    void updateTAAUBuffers(uint32_t currentImage);
    void recordTAAU(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void updateTAAUHistory(const glm::mat4& currentViewProj);

    bool initUI();
    void updateUIFrame();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);
    void updateTAAUUI();

    void recordCommandBuffer(uint32_t imageIndex);

    static float halton(uint32_t index, uint32_t base);
};

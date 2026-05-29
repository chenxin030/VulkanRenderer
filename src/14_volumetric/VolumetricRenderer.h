#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>

#include <vector>
#include <array>

struct VolumetricRenderer final : VulkanBase
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

    // Scene geometry
    Mesh cubeMesh;
    Mesh planeMesh;
    uint32_t instanceCount = 0;

    // Scene buffers
    MeshBuffer sceneUboResources;
    MeshBuffer instanceBufferResources;

    // Volumetric rendering resources
    MeshBuffer volumetricParamsUboResources;
    TextureData volumetricColorData;
    TextureData volumetricDepthData;
    TextureData volumetricHistoryData[2];
    TextureData volumetricVelocityData;
    vk::raii::Sampler defaultSampler = nullptr;

    vk::ImageLayout volumetricColorLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout volumetricDepthLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout volumetricHistoryLayouts[2] = { vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined };
    vk::ImageLayout volumetricVelocityLayout = vk::ImageLayout::eUndefined;

    uint32_t volumetricHistoryReadIndex = 0;
    bool volumetricHistoryValid = false;
    glm::mat4 volumetricPrevViewProj = glm::mat4(1.0f);
    glm::vec2 volumetricJitterCurrent = glm::vec2(0.0f);
    glm::vec2 volumetricJitterPrev = glm::vec2(0.0f);
    uint64_t volumetricFrameCounter = 0;
    float volumetricRenderScale = 0.5f;
    float lastVolumetricRenderScale = 0.5f;

    struct VolumetricSettings
    {
        float density = 0.05f;
        float scattering = 0.8f;
        float absorption = 0.3f;
        float anisotropicG = 0.3f;
        float stepSize = 0.5f;
        float maxDistance = 50.0f;
        float temporalFactor = 0.9f;
        float shadowStrength = 0.5f;
        float intensity = 0.5f;
    };

    bool volumetricEnabled = true;
    float volumetricSceneTime = 0.0f;
    bool volumetricFreezeHistory = false;
    VolumetricSettings volumetricParams{};

    // Clustered volumetric resources
    bool clusteredVolumetricEnabled = false;
    MeshBuffer clusterParamsResources;
    TextureData clusterScatteringData;
    vk::ImageLayout clusterScatteringLayout = vk::ImageLayout::eUndefined;

    // Scene pipeline
    vk::raii::DescriptorSetLayout sceneDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool sceneDescriptorPool = nullptr;
    vk::raii::PipelineLayout scenePipelineLayout = nullptr;
    vk::raii::Pipeline scenePipeline = nullptr;
    vk::raii::DescriptorSets sceneDescriptorSets = nullptr;

    // Volumetric pipeline
    vk::raii::DescriptorSetLayout volumetricDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool volumetricDescriptorPool = nullptr;
    vk::raii::PipelineLayout volumetricPipelineLayout = nullptr;
    vk::raii::Pipeline volumetricPipeline = nullptr;
    vk::raii::DescriptorSets volumetricDescriptorSets = nullptr;

    // Scene methods
    void createSceneBuffers();
    bool createSceneDescriptorSetLayout();
    bool createSceneDescriptorPool();
    void createSceneDescriptorSets();
    bool createScenePipeline();
    void updateSceneBuffers(uint32_t currentImage);

    // Volumetric methods
    bool createVolumetricResources();
    void recreateVolumetricSizedResources();
    bool createVolumetricDescriptorSetLayout();
    bool createVolumetricDescriptorPool();
    void createVolumetricDescriptorSets();
    void updateVolumetricDescriptorSet(uint32_t frameIndex, uint32_t historyReadIndex);
    bool createVolumetricPipeline();
    void updateVolumetricBuffers(uint32_t currentImage);
    void recordVolumetric(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void updateVolumetricHistory(const glm::mat4& currentViewProj);

    // Clustered volumetric methods
    bool createClusteredResources();
    bool createClusteredDescriptorSetLayout();
    bool createClusteredDescriptorPool();
    void createClusteredDescriptorSets();
    bool createClusteredPipeline();

    // UI
    bool initUI();
    void updateUIPanel() override;

    // Command buffer
    void recordCommandBuffer(uint32_t imageIndex);

    // Utils
    static float halton(uint32_t index, uint32_t base);
    static glm::vec2 hammersley(uint32_t i, uint32_t n);
};

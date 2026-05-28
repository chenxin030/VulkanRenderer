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

    // Volumetric rendering resources (Phase 1+)
    MeshBuffer volumetricParamsUboResources;
    TextureData volumetricColorData;      // 半分辨率体积光颜色
    TextureData volumetricDepthData;       // 用于重建位置
    TextureData volumetricHistoryData[2];  // Temporal 历史缓冲
    TextureData volumetricVelocityData;     // 速度缓冲

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
    float volumetricRenderScale = 0.5f;  // 半分辨率

    struct VolumetricParams
    {
        float density = 0.1f;           // 参与介质密度
        float scattering = 1.0f;         // 散射系数
        float absorption = 0.5f;        // 吸收系数
        float anisotropicG = 0.3f;      // Henyey-Greenstein 各向异性参数 [-1, 1]
        float stepSize = 0.5f;          // Ray Marching 步长
        float maxDistance = 50.0f;       // 最大采样距离
        float temporalFactor = 0.9f;    // Temporal 混合因子
        float shadowStrength = 0.5f;     // 阴影强度
        float intensity = 1.0f;         // 光源强度
    };

    bool volumetricEnabled = true;
    float volumetricSceneTime = 0.0f;
    bool volumetricFreezeHistory = false;
    VolumetricParams volumetricParams{};

    // Clustered volumetric resources (Phase 3)
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
    bool createVolumetricDescriptorSetLayout();
    bool createVolumetricDescriptorPool();
    void createVolumetricDescriptorSets();
    void updateVolumetricDescriptorSet(uint32_t frameIndex, uint32_t historyReadIndex);
    bool createVolumetricPipeline();
    void updateVolumetricBuffers(uint32_t currentImage);
    void recordVolumetric(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void updateVolumetricHistory(const glm::mat4& currentViewProj);
    void recreateVolumetricResources();

    // Clustered volumetric methods (Phase 3)
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

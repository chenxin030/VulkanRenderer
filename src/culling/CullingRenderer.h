#pragma once

#include <Base/VulkanBase.h>

struct CullingRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 camPos;
    };

    struct CullingInstanceData
    {
        glm::mat4 model;
        glm::vec4 color;
    };

    struct DrawCommand
    {
        uint32_t indexCount;
        uint32_t instanceCount;
        uint32_t firstIndex;
        int32_t vertexOffset;
        uint32_t firstInstance;
    };

    struct CullingStats
    {
        uint32_t totalCount;
        uint32_t visibleCount;
        float gpuMs;
        float frameMs;
    };

    struct CullingParamsUBO
    {
        glm::vec4 frustumPlanes[6];
        glm::vec4 aabbMin;
        glm::vec4 aabbMax;
        glm::vec4 hiZInfo; // x width, y height, z mipCount, w depthBias
        uint32_t totalInstances;
        uint32_t useCulling;
        uint32_t pad2;
        uint32_t pad3;
    };

    Mesh cubeMesh;

    static constexpr uint32_t kWorkgroupSize = 64u;
    uint32_t totalInstanceCount = 0;
    uint32_t visibleCountCpu = 0;
    bool cullingEnabled = true;
    float cullingGpuMs = 0.0f;
    float frameMs = 0.0f;

    MeshBuffer cullingGlobalUboResources;
    MeshBuffer cullingInstanceBufferResources;
    MeshBuffer cullingIndirectBufferResources;
    MeshBuffer cullingVisibleBufferResources;
    MeshBuffer cullingStatsBufferResources;
    MeshBuffer cullingParamsBufferResources;

    vk::raii::Buffer cullingVisibleCountBuffer = nullptr;
    vk::raii::DeviceMemory cullingVisibleCountMemory = nullptr;
    vk::raii::Buffer cullingVisibleReadbackBuffer = nullptr;
    vk::raii::DeviceMemory cullingVisibleReadbackMemory = nullptr;
    void* cullingVisibleCountMapped = nullptr;

    vk::raii::Buffer cullingStatsReadbackBuffer = nullptr;
    vk::raii::DeviceMemory cullingStatsReadbackMemory = nullptr;
    void* cullingStatsReadbackMapped = nullptr;

    vk::raii::DescriptorSetLayout cullingDepthDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool cullingDepthDescriptorPool = nullptr;
    vk::raii::DescriptorSets cullingDepthDescriptorSets = nullptr;
    vk::raii::PipelineLayout cullingDepthPipelineLayout = nullptr;
    vk::raii::Pipeline cullingDepthPipeline = nullptr;

    vk::raii::DescriptorSetLayout cullingDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool cullingDescriptorPool = nullptr;
    vk::raii::DescriptorSets cullingDescriptorSets = nullptr;
    vk::raii::PipelineLayout cullingPipelineLayout = nullptr;
    vk::raii::Pipeline cullingPipeline = nullptr;

    vk::raii::DescriptorSetLayout cullingDrawDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool cullingDrawDescriptorPool = nullptr;
    vk::raii::DescriptorSets cullingDrawDescriptorSets = nullptr;
    vk::raii::PipelineLayout cullingDrawPipelineLayout = nullptr;
    vk::raii::Pipeline cullingDrawPipeline = nullptr;

    vk::raii::DescriptorSetLayout cullingHiZDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool cullingHiZDescriptorPool = nullptr;
    vk::raii::DescriptorSets cullingHiZDescriptorSets = nullptr;
    vk::raii::PipelineLayout cullingHiZPipelineLayout = nullptr;
    vk::raii::Pipeline cullingHiZPipeline = nullptr;

    TextureData cullingDepthTexture;
    TextureData cullingHiZTexture;
    std::vector<vk::raii::ImageView> cullingHiZMipViews;
    vk::Extent2D cullingDepthExtent{ 0u, 0u };
    vk::ImageLayout cullingDepthLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout cullingHiZLayout = vk::ImageLayout::eUndefined;
    uint32_t cullingHiZMipCount = 1;

    vk::raii::CommandPool computeCommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;
    std::vector<vk::raii::Semaphore> cullingCompleteSemaphores;

    vk::raii::QueryPool cullingTimestampQueryPool = nullptr;

    std::vector<CullingInstanceData> sceneInstances;

    bool createCullingBuffers();
    bool createCullingDescriptorSetLayouts();
    bool createCullingDescriptorPools();
    void createCullingDescriptorSets();
    bool createCullingPipelines();
    bool createCullingDepthResources();
    bool createCullingHiZResources();
    bool createCullingHiZDescriptorSetLayout();
    bool createCullingHiZDescriptorPool();
    void createCullingHiZDescriptorSets();
    void updateCullingHiZDescriptorSets();
    bool createCullingHiZPipeline();

    bool createCullingCommandPool();
    bool createCullingCommandBuffers();
    bool createCullingSyncObjects();

    void buildStressScene();
    bool rebuildSwapchainDependentResources();
    void updateCullingBuffers(uint32_t frameIndex);
    void recordCullingHiZ(vk::raii::CommandBuffer& commandBuffer);
    void recordCullingCommandBuffer(uint32_t imageIndex);
    void recordCullingDrawCommands(vk::raii::CommandBuffer& commandBuffer);
    void recordCommandBuffer(uint32_t imageIndex);
    void updateCullingStats();

    bool initUI();
    void updateUIFrame();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);
    void updateCullingUI();

    static void extractFrustumPlanes(const glm::mat4& matrix, glm::vec4* planesOut);
};

#pragma once

#include <Base/VulkanBase.h>
#include <glm/glm.hpp>
#include <vector>

struct ClusteredRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

    bool createSyncObjects() override;
    bool createCommandBuffers() override;

    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 camPos;
        float nearZ;
    };

    struct ClusterParamsUBO
    {
        uint32_t clusterX;
        uint32_t clusterY;
        uint32_t clusterZ;
        uint32_t totalClusters;
        glm::vec3 tileSize;
        float pad0;
        glm::vec3 cameraPos;
        float farZ;
        float zCount;
        float zMin;
        float zMax;
        float pad1;
        float pad2;
    };

    struct PointLight
    {
        glm::vec4 position; // xyz: position, w: unused
        glm::vec4 color;   // rgb: color, w: intensity
    };

    static constexpr uint32_t MAX_LIGHTS = 2048;
    static constexpr uint32_t DEFAULT_CLUSTER_X = 16;
    static constexpr uint32_t DEFAULT_CLUSTER_Y = 9;
    static constexpr uint32_t DEFAULT_CLUSTER_Z = 24;
    static constexpr uint32_t MAX_LIGHTS_PER_CLUSTER = 64;
    static constexpr float CLUSTER_Z_NEAR = 0.1f;
    static constexpr float CLUSTER_Z_FAR = 100.0f;

    Mesh sphereMesh;
    Mesh cubeMesh;
    Mesh planeMesh;

    uint32_t lightCount = 1024;
    std::vector<PointLight> sceneLights;

    uint32_t clusterX = DEFAULT_CLUSTER_X;
    uint32_t clusterY = DEFAULT_CLUSTER_Y;
    uint32_t clusterZ = DEFAULT_CLUSTER_Z;
    bool clusteredShadingEnabled = true;
    bool uiEnabled = true;

    float frameMs = 0.0f;
    float fps = 0.0f;
    uint32_t avgLightsPerCluster = 0;

    vk::raii::DescriptorSetLayout clusteredDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool clusteredDescriptorPool = nullptr;
    vk::raii::PipelineLayout clusteredPipelineLayout = nullptr;
    vk::raii::Pipeline clusteredPipeline = nullptr;
    vk::raii::DescriptorSets clusteredDescriptorSets = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool computeDescriptorPool = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;
    vk::raii::DescriptorSets computeDescriptorSets = nullptr;

    MeshBuffer sceneUboResources;
    MeshBuffer lightBufferResources;
    MeshBuffer clusterParamsResources;

    vk::raii::Buffer clusterAABBsBuffer = nullptr;
    vk::raii::DeviceMemory clusterAABBsMemory = nullptr;
    void* clusterAABBsMapped = nullptr;

    vk::raii::Buffer lightIndexBuffer = nullptr;
    vk::raii::DeviceMemory lightIndexMemory = nullptr;
    void* lightIndexMapped = nullptr;

    vk::raii::Buffer lightGridBuffer = nullptr;
    vk::raii::DeviceMemory lightGridMemory = nullptr;
    void* lightGridMapped = nullptr;

    vk::raii::Buffer lightGridReadbackBuffer = nullptr;
    vk::raii::DeviceMemory lightGridReadbackMemory = nullptr;
    void* lightGridReadbackMapped = nullptr;

    vk::raii::Fence computeFence = nullptr;

    vk::raii::CommandPool computeCommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;
    std::vector<vk::raii::Semaphore> computeCompleteSemaphores;

    vk::raii::DescriptorSetLayout uiDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool uiDescriptorPool = nullptr;
    vk::raii::PipelineLayout uiPipelineLayout = nullptr;
    vk::raii::Pipeline uiPipeline = nullptr;
    vk::raii::DescriptorSets uiDescriptorSets = nullptr;
    TextureData uiFontTexture;

    struct UiFrameBuffers
    {
        vk::raii::Buffer vertexBuffer = nullptr;
        vk::raii::DeviceMemory vertexBufferMemory = nullptr;
        void* vertexMapped = nullptr;
        size_t vertexSize = 0;

        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        void* indexMapped = nullptr;
        size_t indexSize = 0;
    };
    std::vector<UiFrameBuffers> uiFrameBuffers;

    struct GroundUBO
    {
        glm::mat4 model;
        glm::vec4 color;
    };
    MeshBuffer groundUboResources;

    bool createClusteredDescriptorSetLayout();
    bool createClusteredDescriptorPool();
    void createClusteredDescriptorSets();
    bool createClusteredPipeline();

    bool createComputeDescriptorSetLayout();
    bool createComputeDescriptorPool();
    void createComputeDescriptorSets();
    bool createComputePipeline();

    bool createClusterBuffers();
    bool createClusterCommandPool();
    bool createClusterCommandBuffers();
    bool createComputeSyncObjects();

    void generateSceneLights();
    void updateClusterBuffers(uint32_t frameIndex);
    void updateLightBuffer(uint32_t frameIndex);
    void updateClusterStats();

    void recordComputeCommandBuffer();
    void recordCommandBuffer(uint32_t imageIndex);

    bool initUI();
    void shutdownUI();
    void updateUIFrame();
    void updateClusteredUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);

    uint32_t getTotalClusters() const { return clusterX * clusterY * clusterZ; }
    uint32_t getLightIndexBufferSize() const { return getTotalClusters() * MAX_LIGHTS_PER_CLUSTER; }
};

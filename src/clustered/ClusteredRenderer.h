#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>
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

    float frameMs = 0.0f;
    float fps = 0.0f;
    uint32_t avgLightsPerCluster = 0;
    uint32_t allocatedTotalClusters = 0;

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

    std::vector<vk::raii::Fence> computeFences;

    vk::raii::CommandPool computeCommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;
    std::vector<vk::raii::Semaphore> computeCompleteSemaphores;

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

    void recordComputeCommandBuffer(uint32_t frameIndex);
    void recordCommandBuffer(uint32_t frameIndex);

    bool initUI();
    void updateUIPanel() override;

    uint32_t getTotalClusters() const { return clusterX * clusterY * clusterZ; }
    uint32_t getLightIndexBufferSize() const { return getTotalClusters() * MAX_LIGHTS_PER_CLUSTER; }

private:
    void recreateSwapChain() override;
    uint32_t swapChainImageCount = 0;
};

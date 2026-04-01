#pragma once

#include <Base/VulkanBase.h>
#include <Base/ResourceManager.h>

#include <vector>
#include <random>

struct ParticleRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

    struct Particle
    {
        glm::vec3 position;
        float lifetime;
        glm::vec3 velocity;
        float maxLifetime;
        glm::vec4 color;
        float size;
        glm::vec3 padding;
    };

    struct ParticleParamsUBO
    {
        glm::vec3 emitterPosition;
        float emissionRate;
        glm::vec3 gravity;
        float particleLifetime;
        float particleSize;
        float turbulenceStrength;
        float spread;
        float velocity;
    };

    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 camPos;
        float pad0;
    };

    static constexpr uint32_t MAX_PARTICLES = 100000u;
    static constexpr uint32_t COMPUTE_WORKGROUP_SIZE = 64u;

    float emissionRate = 5000.0f;
    float particleLifetime = 3.0f;
    float particleSize = 0.15f;
    float gravity = 9.8f;
    float turbulenceStrength = 0.5f;
    float spread = 1.5f;
    float velocity = 5.0f;

    bool uiEnabled = true;

    MeshBuffer particleBufferResources;
    MeshBuffer particleParamsBufferResources;
    MeshBuffer sceneUboResources;

    vk::raii::DescriptorSetLayout particleDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool particleDescriptorPool = nullptr;
    vk::raii::DescriptorSets particleDescriptorSets = nullptr;
    vk::raii::PipelineLayout particlePipelineLayout = nullptr;
    vk::raii::Pipeline particlePipeline = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool computeDescriptorPool = nullptr;
    vk::raii::DescriptorSets computeDescriptorSets = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;

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
        size_t vertexBufferSize = 0;
        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        void* indexMapped = nullptr;
        size_t indexSize = 0;
        size_t indexBufferSize = 0;
    };
    std::vector<UiFrameBuffers> uiFrameBuffers;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist01{ 0.0f, 1.0f };
    std::uniform_real_distribution<float> dist11{ -1.0f, 1.0f };

    bool createParticleBuffers();
    bool createParticleDescriptorSetLayouts();
    bool createParticleDescriptorPools();
    void createParticleDescriptorSets();
    bool createParticlePipeline();
    bool createComputePipeline();
    bool createComputeCommandPool();
    bool createComputeCommandBuffers();
    bool createComputeSyncObjects();
    bool initUI();
    void shutdownUI();
    void updateUIFrame();
    void updateParticleUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);
    void updateParticleBuffers(uint32_t frameIndex);
    void recordComputeCommandBuffer(uint32_t frameIndex);
    void recordCommandBuffer(uint32_t imageIndex);
    void generateInitialParticles();
};

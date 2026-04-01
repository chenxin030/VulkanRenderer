#pragma once

#include <Base/VulkanBase.h>

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <cstdint>

struct SkinningRenderer final : VulkanBase
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

    static constexpr uint32_t MAX_JOINTS = 64;
    static constexpr uint32_t BONE_COUNT = 16;
    static constexpr float ANIM_TIME = 8.0f;

    Mesh characterMesh;
    Mesh planeMesh;

    MeshBuffer sceneUboResources;
    MeshBuffer jointBufferResources;

    std::vector<glm::mat4> currentJointMatrices;
    std::vector<std::array<int32_t, MAX_JOINTS>> jointParents;

    bool animateEnabled = true;
    bool uiEnabled = true;
    float frameMs = 0.0f;
    float fps = 0.0f;

    vk::raii::DescriptorSetLayout skinnedDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool skinnedDescriptorPool = nullptr;
    vk::raii::PipelineLayout skinnedPipelineLayout = nullptr;
    vk::raii::Pipeline skinnedPipeline = nullptr;
    vk::raii::DescriptorSets skinnedDescriptorSets = nullptr;

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

    bool createSkinnedMesh();
    void generatePlaneMesh();
    void generateSkinnedCharacter();

    bool createSkinnedDescriptorSetLayout();
    bool createSkinnedDescriptorPool();
    void createSkinnedDescriptorSets();
    bool createSkinnedPipeline();

    void generateJointHierarchy();
    void updateJointMatrices(float time);
    void updateSceneUBO(uint32_t frameIndex);
    void updateBuffers(uint32_t frameIndex);

    void recordCommandBuffer(uint32_t imageIndex);

    bool initUI();
    void shutdownUI();
    void updateUIFrame();
    void updateSkinningUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);
};

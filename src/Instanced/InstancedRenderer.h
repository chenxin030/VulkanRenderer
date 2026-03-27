#pragma once

#include <VulkanBase.h>
#include <glm/glm.hpp>
#include <array>

// Independent "triangle" example (no RENDERING_LEVEL).
struct InstancedRenderer final : VulkanBase
{
    void initialize(Platform* _platform, ResourceManager* _resourceManager, Scene* _scene);

    bool initVulkan();
    bool prepareResource();
    void processInput(float) {}
    void render();
    void cleanup();

private:
    struct PushConstants
    {
        glm::mat4 viewProj{ 1.0f };
    };

    Mesh mesh;
    TextureData texture;

    vk::raii::DescriptorSetLayout instancedDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool      instancedDescriptorPool = nullptr;
    vk::raii::PipelineLayout      instancedPipelineLayout = nullptr;
    vk::raii::Pipeline            instancedPipeline = nullptr;
    uint32_t instanceCount = 3;
    std::array<float, 3> instanceAnglesRad{ 0.0f, 0.0f, 0.0f };

    MeshBuffer instancedBufferResources;
    MeshBuffer globalUboResources;

    bool createInstancedDescriptorSetLayout();
    bool createInstancedDescriptorPool();
    void createInstancedDescriptorSets();
    bool createInstancedPipeline();
    void createInstanceBuffer();
    void updateInstancedBuffers(uint32_t frameIndex);

    void recordCommandBuffer(uint32_t imageIndex);
};


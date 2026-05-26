#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>
#include <glm/glm.hpp>
#include <array>
#include <Mesh.h>

struct InstancedRenderer final : VulkanBase
{
    void initialize(Platform* _platform);

    bool initVulkan();
    bool initUI();
    bool prepareResource();
    void render();
    void cleanup();

private:
    void updateUIPanel() override;

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


#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>
#include <glm/glm.hpp>
#include <vector>

struct PBRRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool initUI();
    bool prepareResource();
    void render();
    void waitIdle() {
        device.waitIdle();
        shutdownVulkanUI();
    }

private:
    void updateUIPanel() override;

private:
    // PBR pipeline resources
    vk::raii::DescriptorSetLayout pbrDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout      pbrPipelineLayout = nullptr;
    vk::raii::Pipeline            pbrPipeline = nullptr;
    vk::raii::DescriptorPool      pbrDescriptorPool = nullptr;

    Mesh sphereMesh;

    MeshBuffer pbrInstanceBufferResources;
    MeshBuffer sceneUboResources;
    MeshBuffer lightUboResources;

    uint32_t instanceCount = 49;

    bool createPBRDescriptorSetLayout();
    bool createPBRDescriptorPool();
    void createPBRDescriptorSets();
    bool createPBRPipeline();
    void createPBRBuffers();
    void updatePBRInstanceBuffers(uint32_t frameIndex);

    void recordCommandBuffer(uint32_t imageIndex);
};


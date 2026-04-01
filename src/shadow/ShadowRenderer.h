#pragma once

#include <Base/VulkanBase.h>

struct ShadowRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    vk::raii::DescriptorSetLayout shadowDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool      shadowDescriptorPool = nullptr;
    vk::raii::PipelineLayout      shadowPipelineLayout = nullptr;
    vk::raii::Pipeline            shadowDepthPipeline = nullptr;
    vk::raii::Pipeline            shadowLitPipeline = nullptr;

    Mesh cubeMesh;
    uint32_t instanceCount = 0;

    MeshBuffer shadowInstanceBufferResources;
    MeshBuffer sceneUboResources;
    MeshBuffer shadowUboResources;
    MeshBuffer shadowParamsUboResources;

    TextureData shadowMapData;
    vk::Extent2D shadowMapExtent{ 2048u, 2048u };
    vk::ImageLayout shadowMapLayout = vk::ImageLayout::eUndefined;

    int shadowFilterMode = 2;      // 0: hard, 1: pcf, 2: pcss
    float pcfRadiusTexels = 2.0f;
    float pcssLightSizeTexels = 25.0f;

    float dirLightIntensity = 0.5f;
    float pointLightIntensity = 3.5f;
    float areaLightIntensity = 2.5f;

    bool createShadowDescriptorSetLayout();
    bool createShadowDescriptorPool();
    void createShadowDescriptorSets();

    void createShadowBuffers();
    void updateShadowBuffers(uint32_t frameIndex);

    bool createShadowMapResources();
    bool createShadowPipelines();

    bool initUI();
    void updateUIFrame();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);
    void updateShadowUI();

    void recordCommandBuffer(uint32_t imageIndex);
};


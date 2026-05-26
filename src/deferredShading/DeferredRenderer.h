#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>

struct DeferredRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    struct GBufferAttachment
    {
        TextureData texture;
        vk::Format format = vk::Format::eUndefined;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };

    GBufferAttachment gbufferAlbedo;
    GBufferAttachment gbufferNormal;
    GBufferAttachment gbufferMaterial;
    GBufferAttachment gbufferDepth;

    vk::raii::DescriptorSetLayout gbufferDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool gbufferDescriptorPool = nullptr;
    vk::raii::PipelineLayout gbufferPipelineLayout = nullptr;
    vk::raii::Pipeline gbufferPipeline = nullptr;

    vk::raii::DescriptorSetLayout deferredDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool deferredDescriptorPool = nullptr;
    vk::raii::PipelineLayout deferredPipelineLayout = nullptr;
    vk::raii::Pipeline deferredPipeline = nullptr;

    Mesh sphereMesh;

    MeshBuffer gbufferInstanceBufferResources;
    MeshBuffer sceneUboResources;
    MeshBuffer lightUboResources;
    MeshBuffer deferredSettingsUboResources;

    uint32_t instanceCount = 49;

    vk::raii::Sampler gbufferSampler = nullptr;

    float ambientStrength = 0.03f;
    float exposure = 1.0f;
    float gamma = 2.2f;
    float lightIntensityScale = 1.0f;
    bool animateLights = true;
    int debugView = 0;

    bool createGBufferDescriptorSetLayout();
    bool createDeferredDescriptorSetLayout();

    bool createGBufferDescriptorPool();
    bool createDeferredDescriptorPool();

    void createGBufferDescriptorSets();
    void createDeferredDescriptorSets();

    bool createGBufferResources();
    void destroyGBufferResources();

    bool createGBufferPipeline();
    bool createDeferredPipeline();

    void createDeferredBuffers();
    void updateDeferredBuffers(uint32_t frameIndex);

    bool recreateDeferredSizedResources();

    bool initUI();
    void updateUIPanel() override;

    void recordCommandBuffer(uint32_t imageIndex);
};

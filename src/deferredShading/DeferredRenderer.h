#pragma once

#include <Base/VulkanBase.h>

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

    GBufferAttachment gbufferAlbedo;
    GBufferAttachment gbufferNormal;
    GBufferAttachment gbufferMaterial;
    GBufferAttachment gbufferDepth;

    vk::raii::Sampler gbufferSampler = nullptr;

    // PBR/lighting controls
    float ambientStrength = 0.03f;
    float exposure = 1.0f;
    float gamma = 2.2f;
    float lightIntensityScale = 1.0f;
    bool animateLights = true;
    int debugView = 0; // 0: lit, 1: albedo, 2: normal, 3: material, 4: depth

    bool uiEnabled = true;
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
    void shutdownUI();
    void updateUIFrame();
    void updateDeferredUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);

    void recordCommandBuffer(uint32_t imageIndex);
};

#pragma once

#include <Base/VulkanBase.h>

struct GIRenderer final : VulkanBase
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
        vk::raii::DescriptorSets descriptorSet = nullptr;
    };

    vk::raii::DescriptorSetLayout gbufferDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool gbufferDescriptorPool = nullptr;
    vk::raii::PipelineLayout gbufferPipelineLayout = nullptr;
    vk::raii::Pipeline gbufferPipeline = nullptr;

    vk::raii::DescriptorSetLayout ssaoDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool ssaoDescriptorPool = nullptr;
    vk::raii::PipelineLayout ssaoPipelineLayout = nullptr;
    vk::raii::Pipeline ssaoPipeline = nullptr;

    vk::raii::DescriptorSetLayout ssaoBlurDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool ssaoBlurDescriptorPool = nullptr;
    vk::raii::PipelineLayout ssaoBlurPipelineLayout = nullptr;
    vk::raii::Pipeline ssaoBlurPipeline = nullptr;

    vk::raii::DescriptorSetLayout lightingDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool lightingDescriptorPool = nullptr;
    vk::raii::PipelineLayout lightingPipelineLayout = nullptr;
    vk::raii::Pipeline lightingPipeline = nullptr;

    Mesh sphereMesh;

    MeshBuffer gbufferInstanceBufferResources;
    MeshBuffer sceneUboResources;
    MeshBuffer lightUboResources;
    MeshBuffer ssaoSettingsUboResources;
    MeshBuffer ssaoBlurUboResources;

    uint32_t instanceCount = 49;

    GBufferAttachment gbufferAlbedo;
    GBufferAttachment gbufferNormal;
    GBufferAttachment gbufferMaterial;
    GBufferAttachment gbufferDepth;

    GBufferAttachment ssaoColor;
    GBufferAttachment ssaoBlurColor;

    vk::raii::Sampler gbufferSampler = nullptr;
    vk::raii::Sampler noiseSampler = nullptr;
    TextureData noiseTexture;

    static constexpr int SSAO_SAMPLE_COUNT = 16;
    std::vector<glm::vec4> ssaoKernel;

    bool ssaoEnabled = true;
    float ssaoRadius = 0.5f;
    float ssaoBias = 0.025f;
    float ssaoIntensity = 1.5f;

    float ambientStrength = 0.03f;
    float exposure = 1.0f;
    float gamma = 2.2f;
    float lightIntensityScale = 1.0f;
    bool animateLights = true;
    int debugView = 0;

    bool createGBufferDescriptorSetLayout();
    bool createSsaoDescriptorSetLayout();
    bool createSsaoBlurDescriptorSetLayout();
    bool createLightingDescriptorSetLayout();

    bool createGBufferDescriptorPool();
    bool createSsaoDescriptorPool();
    bool createSsaoBlurDescriptorPool();
    bool createLightingDescriptorPool();

    void createGBufferDescriptorSets();
    void createSsaoDescriptorSets();
    void createSsaoBlurDescriptorSets();
    void createLightingDescriptorSets();

    bool createGBufferResources();
    bool createSsaoResources();
    void destroyGBufferResources();
    void destroySsaoResources();

    bool createGBufferPipeline();
    bool createSsaoPipeline();
    bool createSsaoBlurPipeline();
    bool createLightingPipeline();

    void createBuffers();
    void updateBuffers(uint32_t frameIndex);
    void generateSsaoKernel();
    void createNoiseTexture();

    bool recreateSizedResources();

    bool initUI();
    void updateUIFrame();
    void updateGIVUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);

    void recordCommandBuffer(uint32_t imageIndex);
};

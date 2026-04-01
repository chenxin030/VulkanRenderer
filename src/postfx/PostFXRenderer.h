#pragma once

#include <Base/VulkanBase.h>

struct PostFXRenderer final : VulkanBase
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

    struct GBuffer
    {
        GBufferAttachment albedo;
        GBufferAttachment normal;
        GBufferAttachment material;
        GBufferAttachment depth;
    } gbuffer;

    vk::raii::Sampler gbufferSampler = nullptr;

    // Post-processing framebuffers (ping-pong)
    struct PostBuffer
    {
        TextureData color;  // RGBA16 float HDR scene
        TextureData bloom; // Quarter-res bright-pass bloom
        vk::ImageLayout colorLayout = vk::ImageLayout::eUndefined;
        vk::ImageLayout bloomLayout = vk::ImageLayout::eUndefined;
    };
    std::vector<PostBuffer> postBuffers; // ping-pong: MAX_FRAMES_IN_FLIGHT

    // Intermediate bloom blur textures (ping-pong)
    struct BloomBlurBuffer
    {
        TextureData horizontal; // Quarter-res, blur H
        TextureData vertical;   // Quarter-res, blur V
        vk::ImageLayout hLayout = vk::ImageLayout::eUndefined;
        vk::ImageLayout vLayout = vk::ImageLayout::eUndefined;
    };
    std::vector<BloomBlurBuffer> blurBuffers;

    vk::raii::Sampler postSampler = nullptr; // linear clamp

    vk::raii::DescriptorSetLayout gbufferDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool gbufferDescriptorPool = nullptr;
    vk::raii::PipelineLayout gbufferPipelineLayout = nullptr;
    vk::raii::Pipeline gbufferPipeline = nullptr;

    bool createGBufferDescriptorSetLayout();
    bool createGBufferDescriptorPool();
    void createGBufferDescriptorSets();
    bool createGBufferResources();
    void destroyGBufferResources();
    bool createGBufferPipeline();
    bool recreateGBufferSizedResources();

    MeshBuffer sceneUboResources;
    MeshBuffer lightUboResources;
    MeshBuffer deferredSettingsUboResources;
    MeshBuffer postFxSettingsUboResources;
    MeshBuffer instanceBufferResources;

    uint32_t instanceCount = 49;

    // Scene geometry
    Mesh sphereMesh;

    vk::raii::DescriptorSetLayout deferredDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool deferredDescriptorPool = nullptr;
    vk::raii::PipelineLayout deferredPipelineLayout = nullptr;
    vk::raii::Pipeline deferredPipeline = nullptr;

    bool createDeferredDescriptorSetLayout();
    bool createDeferredDescriptorPool();
    void createDeferredDescriptorSets();
    bool createDeferredPipeline();
    void createDeferredBuffers();
    void updateDeferredBuffers(uint32_t imageIndex);

    // Post-processing
    void updatePostSettingsBuffers(uint32_t imageIndex);

    // Bloom passes
    vk::raii::DescriptorSetLayout bloomExtractDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool bloomExtractDescriptorPool = nullptr;
    vk::raii::PipelineLayout bloomExtractPipelineLayout = nullptr;
    vk::raii::Pipeline bloomExtractPipeline = nullptr;

    vk::raii::DescriptorSetLayout blurDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool blurDescriptorPool = nullptr;
    vk::raii::PipelineLayout blurHPipelineLayout = nullptr;
    vk::raii::PipelineLayout blurVPipelineLayout = nullptr;
    vk::raii::Pipeline blurHPipeline = nullptr;
    vk::raii::Pipeline blurVPipeline = nullptr;
    vk::PushConstantRange blurPushConstRange{ .stageFlags = vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = 20u };

    vk::raii::DescriptorSetLayout bloomCompositeDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool bloomCompositeDescriptorPool = nullptr;
    vk::raii::PipelineLayout bloomCompositePipelineLayout = nullptr;
    vk::raii::Pipeline bloomCompositePipeline = nullptr;

    // Per-pass descriptor sets (each pass owns its own sets)
    vk::raii::DescriptorSets bloomExtractDescriptorSets = nullptr; // bloom extract: scene texture + settings
    vk::raii::DescriptorSets blurDescriptorSets = nullptr;        // V-blur: reads blurH
    vk::raii::DescriptorSets blurHDescriptorSets = nullptr;     // H-blur: reads bloom
    vk::raii::DescriptorSets bloomCompositeDescriptorSets = nullptr; // composite: scene + bloom + settings

    bool createPostDescriptorSetLayout();
    bool createPostDescriptorPool();
    void createPostDescriptorSets();
    bool createPostBuffers();
    void destroyPostBuffers();
    bool recreatePostSizedResources();
    void destroyPostBuffersPerFrame();

    void updatePostDescriptorSets();
    bool createBloomExtractPipeline();
    bool createBlurPipelines();
    bool createBloomCompositePipeline();

    bool initUI();
    void updateUIFrame(uint32_t imageIndex);
    void updatePostFXUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

    void recordGBufferPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void recordDeferredPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void recordBloomExtractPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void recordBlurPasses(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void recordBloomCompositePass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void recordUIPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
    void recordCommandBuffer(uint32_t imageIndex);

    // Settings (controlled by UI)
    float exposure = 1.0f;
    float gamma = 2.2f;
    float ambientStrength = 0.03f;
    float lightIntensityScale = 1.0f;
    bool animateLights = true;
    int debugView = 0;

    // Post-processing
    bool bloomEnabled = true;
    float bloomThreshold = 0.8f;
    float bloomIntensity = 1.5f;
    float bloomRadius = 4.0f;

    int toneMappingMode = 1; // 0: Reinhard, 1: ACES Filmic
    bool chromaticAberrationEnabled = false;
    float chromaticAberrationStrength = 0.003f;
    bool vignetteEnabled = true;
    float vignetteIntensity = 0.4f;
};

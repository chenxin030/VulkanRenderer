#pragma once

#include <Base/VulkanBase.h>

struct SSRRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    struct SSRSceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 invProjection;
        glm::vec4 cameraPosNear;
        glm::vec4 cameraFarPadding;
    };

    struct SSRParams
    {
        float maxRayDistance;
        float thickness;
        float stride;
        float intensity;

        glm::vec2 invResolution;
        int debugMode;
        int maxSteps;
        float padding0;
    };

    vk::raii::DescriptorSetLayout shadowDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool shadowDescriptorPool = nullptr;
    vk::raii::PipelineLayout shadowPipelineLayout = nullptr;
    vk::raii::Pipeline shadowDepthPipeline = nullptr;
    vk::raii::Pipeline shadowLitPipeline = nullptr;

    Mesh cubeMesh;
    uint32_t instanceCount = 0;

    MeshBuffer shadowInstanceBufferResources;
    MeshBuffer sceneUboResources;
    MeshBuffer shadowUboResources;
    MeshBuffer shadowParamsUboResources;

    TextureData shadowMapData;
    vk::Extent2D shadowMapExtent{ 2048u, 2048u };
    vk::ImageLayout shadowMapLayout = vk::ImageLayout::eUndefined;

    int shadowFilterMode = 2;
    float pcfRadiusTexels = 2.0f;
    float pcssLightSizeTexels = 25.0f;

    float dirLightIntensity = 0.5f;
    float pointLightIntensity = 3.5f;
    float areaLightIntensity = 2.5f;

    vk::raii::DescriptorSetLayout ssrDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool ssrDescriptorPool = nullptr;
    vk::raii::PipelineLayout ssrPipelineLayout = nullptr;
    vk::raii::Pipeline ssrPipeline = nullptr;
    vk::raii::DescriptorSets ssrDescriptorSets = nullptr;

    MeshBuffer ssrSceneUboResources;
    MeshBuffer ssrParamsUboResources;
    TextureData ssrColorData;
    TextureData ssrNormalData;
    TextureData ssrDepthData;
    vk::ImageLayout ssrColorLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout ssrNormalLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout ssrDepthLayout = vk::ImageLayout::eUndefined;
    vk::raii::Sampler ssrColorSampler = nullptr;
    vk::raii::Sampler ssrDepthSampler = nullptr;

    int ssrMaxSteps = 85;
    float ssrMaxRayDistance = 16.0f;
    float ssrThickness = 0.12f;
    float ssrStride = 0.05f;
    float ssrIntensity = 1.0f;
    int ssrDebugMode = 0;
    bool ssrEnabled = true;

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

    bool createShadowDescriptorSetLayout();
    bool createShadowDescriptorPool();
    void createShadowDescriptorSets();
    void createShadowBuffers();
    void updateShadowBuffers(uint32_t currentImage);
    bool createShadowMapResources();
    bool createShadowPipelines();

    bool createSSRResources();
    bool createSSRDescriptorSetLayout();
    bool createSSRDescriptorPool();
    void createSSRDescriptorSets();
    bool createSSRPipeline();
    void updateSSRBuffers(uint32_t currentImage);
    void recordSSR(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

    bool initUI();
    void shutdownUI();
    void updateUIFrame();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);

    void updateSSRUI();
    void recordCommandBuffer(uint32_t imageIndex);
};

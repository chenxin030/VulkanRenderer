#pragma once

#include <Base/VulkanBase.h>

struct AtmosphericRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    struct PushConstants
    {
        glm::mat4 invView;
        glm::mat4 invProjection;
        glm::vec4 cameraPos;
        glm::vec2 viewportSize;
        glm::vec2 padding;
        float sunElevation;   // radians from horizon
        float sunAzimuth;    // radians
        float timeOfDay;      // 0-24 hours
        float turbidity;      // 1-10, atmospheric haze
        float mieCoefficient;
        float rayleighCoefficient;
    };

    vk::raii::DescriptorSetLayout skyDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool skyDescriptorPool = nullptr;
    vk::raii::PipelineLayout skyPipelineLayout = nullptr;
    vk::raii::Pipeline skyPipeline = nullptr;
    vk::raii::DescriptorSets skyDescriptorSets = nullptr;

    bool initUI();
    void updateUIFrame();
    void updateAtmosphericUI();
    void recordUI(vk::raii::CommandBuffer& commandBuffer);

    // Scene settings
    float sunElevation = 0.3f;    // radians above horizon
    float sunAzimuth = 0.5f;      // radians
    float timeOfDay = 12.0f;      // noon
    float turbidity = 2.0f;       // clear sky
    bool animateSun = false;

    bool createSkyDescriptorSetLayout();
    bool createSkyDescriptorPool();
    void createSkyDescriptorSets();
    bool createSkyPipeline();

    void recordCommandBuffer(uint32_t imageIndex);
};

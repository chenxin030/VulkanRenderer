#pragma once

#include <Core/VulkanBase.h>

// Independent "triangle" example (no RENDERING_LEVEL).
struct TriangleRenderer final : VulkanBase
{
    void initialize(Platform* _platform, ResourceManager* _resourceManager, Scene* _scene);

    bool initVulkan();
    void prepareResource();
    void processInput(float) {}
    void render();
    void cleanup();

private:
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    bool createTrianglePipeline();
    void recordCommandBuffer(uint32_t imageIndex);
};


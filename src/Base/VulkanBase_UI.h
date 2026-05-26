#pragma once

#include <imgui.h>
#include <vector>
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

struct UiFrameBuffers
{
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    void* vertexMapped = nullptr;
    size_t vertexSize = 0;
    size_t vertexBufferSize = 0;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    void* indexMapped = nullptr;
    size_t indexSize = 0;
    size_t indexBufferSize = 0;
};

struct VulkanBase;

struct UiPushConsts
{
    glm::vec2 scale;
    glm::vec2 translate;
};

bool initVulkanUI(VulkanBase* self);
void vkrShutdownUI(VulkanBase* self);
void vkrUpdateUIFrame(VulkanBase* self);
void vkrRecordUICmdBuffer(VulkanBase* self, vk::raii::CommandBuffer& cmdBuffer);
void vkrRecordUICmdBuffer(VulkanBase* self, vk::raii::CommandBuffer& cmdBuffer, uint32_t frameIndex);
void vkrReallocateUIMemory(VulkanBase* self, UiFrameBuffers& fb, size_t vtxBytes, size_t idxBytes);
void vkrUpdateUIPanel(VulkanBase* self);

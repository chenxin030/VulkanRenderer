#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>

// Forward declaration — complete type provided by VulkanBase.h in .cpp files
struct VulkanBase;

// ====================================================================
// VkrRenderDevice — Narrow seam for pass states.
// Replaces VkrRenderer* back-channel (m_orch) with a focused interface.
// Pass states access only what they need: Device, PhysicalDevice,
// VulkanBase helpers, and maxFramesInFlight.
// ====================================================================
struct VkrRenderDevice
{
    vk::raii::Device*        device = nullptr;
    vk::raii::PhysicalDevice* physicalDevice = nullptr;
    VulkanBase*              vkBase = nullptr;
    uint32_t                 maxFramesInFlight = 2;
};

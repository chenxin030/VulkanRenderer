#include "VulkanBase.h"

#include <fstream>
#include <set>

uint32_t VulkanBase::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

std::vector<char> VulkanBase::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

vk::raii::ShaderModule VulkanBase::createShaderModule(const std::vector<char>& code)
{
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };
    return vk::raii::ShaderModule(device, createInfo);
}

QueueFamilyIndices VulkanBase::findQueueFamilies(const vk::raii::PhysicalDevice& device)
{
    QueueFamilyIndices indices;
    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); i++)
    {
        const auto& qf = queueFamilies[i];
        if ((qf.queueFlags & vk::QueueFlagBits::eGraphics) && !indices.graphicsFamily.has_value())
        {
            indices.graphicsFamily = i;
        }
        if ((qf.queueFlags & vk::QueueFlagBits::eCompute) && !indices.computeFamily.has_value())
        {
            indices.computeFamily = i;
        }
        if (!indices.presentFamily.has_value() && device.getSurfaceSupportKHR(i, *surface))
        {
            indices.presentFamily = i;
        }
        if ((qf.queueFlags & vk::QueueFlagBits::eTransfer) && !(qf.queueFlags & vk::QueueFlagBits::eGraphics))
        {
            if (!indices.transferFamily.has_value())
            {
                indices.transferFamily = i;
            }
        }
        if (indices.isComplete() && indices.transferFamily.has_value())
        {
            break;
        }
    }

    if (!indices.transferFamily.has_value() && indices.graphicsFamily.has_value())
    {
        indices.transferFamily = indices.graphicsFamily;
        std::cout << "no dedicated transfer queue, reuse graphics queue for transfer\n";
    }

    return indices;
}

SwapChainSupportDetails VulkanBase::querySwapChainSupport(const vk::raii::PhysicalDevice& device)
{
    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(*surface);
    details.formats = device.getSurfaceFormatsKHR(*surface);
    details.presentModes = device.getSurfacePresentModesKHR(*surface);
    return details;
}

uint32_t VulkanBase::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

bool VulkanBase::checkDeviceExtensionSupport(vk::raii::PhysicalDevice& device)
{
    auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensionsSet(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end());
    for (const auto& extension : availableDeviceExtensions)
    {
        requiredExtensionsSet.erase(extension.extensionName);
    }
    return requiredExtensionsSet.empty();
}

vk::SurfaceFormatKHR VulkanBase::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

vk::PresentModeKHR VulkanBase::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox)
        {
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D VulkanBase::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    int width = 0, height = 0;
    platform->GetWindowSize(&width, &height);

    vk::Extent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };

    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
}


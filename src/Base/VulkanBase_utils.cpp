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


vk::Format VulkanBase::findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
    try {
        for (vk::Format format : candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("Failed to find supported format");
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to find supported format: " << e.what() << std::endl;
        throw;
    }
}

vk::Format VulkanBase::findDepthFormat() {
    try {
        vk::Format depthFormat = findSupportedFormat(
            { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);
        std::cout << "Found depth format: " << static_cast<int>(depthFormat) << std::endl;
        return depthFormat;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to find supported depth format, falling back to D32_SFLOAT: " << e.what() << std::endl;
        // Fallback to D32_SFLOAT which is widely supported
        return vk::Format::eD32Sfloat;
    }
}

bool VulkanBase::hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void VulkanBase::generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
    // Check if image format supports linear blit-ing
    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier = { .srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eTransferRead, .oldLayout = vk::ImageLayout::eTransferDstOptimal, .newLayout = vk::ImageLayout::eTransferSrcOptimal, .srcQueueFamilyIndex = vk::QueueFamilyIgnored, .dstQueueFamilyIndex = vk::QueueFamilyIgnored, .image = image };
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

        vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
        offsets[0] = vk::Offset3D(0, 0, 0);
        offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
        dstOffsets[0] = vk::Offset3D(0, 0, 0);
        dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1);
        vk::ImageBlit blit = { .srcSubresource = {}, .srcOffsets = offsets, .dstSubresource = {}, .dstOffsets = dstOffsets };
        blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
        blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);

        commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, { blit }, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

    endSingleTimeCommands(*commandBuffer);
}
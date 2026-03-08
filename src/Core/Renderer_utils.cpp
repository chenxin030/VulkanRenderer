#include <Renderer.h>
#include <set>
#include <fstream>

uint32_t Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
    try {
        // Get memory properties
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        // Find suitable memory type
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to find memory type: " << e.what() << std::endl;
        throw;
    }
}

std::vector<char> Renderer::readFile(const std::string& filename) {
    try {
        // Open file at end to get size
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Get file size
        size_t fileSize = file.tellg();
        std::vector<char> buffer(fileSize);

        // Go back to beginning of file and read data
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        // Close file
        file.close();

        return buffer;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to read file: " << e.what() << std::endl;
        throw;
    }
}
vk::raii::ShaderModule Renderer::createShaderModule(const std::vector<char>& code) {
    try {
        // Create shader module
        vk::ShaderModuleCreateInfo createInfo{
          .codeSize = code.size(),
          .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };

        return vk::raii::ShaderModule(device, createInfo);
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create shader module: " << e.what() << std::endl;
        throw;
    }
}

QueueFamilyIndices Renderer::findQueueFamilies(const vk::raii::PhysicalDevice& device) {
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    // Find queue families that support graphics, compute, present, and (optionally) a dedicated transfer queue
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        const auto& qf = queueFamilies[i];
        // Check for graphics support
        if ((qf.queueFlags & vk::QueueFlagBits::eGraphics) && !indices.graphicsFamily.has_value()) {
            indices.graphicsFamily = i;
        }
        // Check for compute support
        if ((qf.queueFlags & vk::QueueFlagBits::eCompute) && !indices.computeFamily.has_value()) {
            indices.computeFamily = i;
        }
        // Check for present support
        if (!indices.presentFamily.has_value() && device.getSurfaceSupportKHR(i, *surface)) {
            indices.presentFamily = i;
        }
        // Prefer a dedicated transfer queue (transfer bit set, but NOT graphics) if available
        if ((qf.queueFlags & vk::QueueFlagBits::eTransfer) && !(qf.queueFlags & vk::QueueFlagBits::eGraphics)) {
            if (!indices.transferFamily.has_value()) {
                indices.transferFamily = i;
            }
        }
        // If all required queue families are found, we can still continue to try find a dedicated transfer queue
        if (indices.isComplete() && indices.transferFamily.has_value()) {
            // Found everything including dedicated transfer
            break;
        }
    }

    // Fallback: if no dedicated transfer queue, reuse graphics queue for transfer
    if (!indices.transferFamily.has_value() && indices.graphicsFamily.has_value()) {
        indices.transferFamily = indices.graphicsFamily;
        std::cout << "no dedicated transfer queue, reuse graphics queue for transfer\n";
    }

    return indices;
}

SwapChainSupportDetails Renderer::querySwapChainSupport(const vk::raii::PhysicalDevice& device) {
	SwapChainSupportDetails details;

	details.capabilities = device.getSurfaceCapabilitiesKHR(*surface);
	details.formats = device.getSurfaceFormatsKHR(*surface);
	details.presentModes = device.getSurfacePresentModesKHR(*surface);

	return details;
}

uint32_t Renderer::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

bool Renderer::checkDeviceExtensionSupport(vk::raii::PhysicalDevice& device) {
    auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();

    // Check if all required extensions are supported
    std::set<std::string> requiredExtensionsSet(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end());

    for (const auto& extension : availableDeviceExtensions) {
        requiredExtensionsSet.erase(extension.extensionName);
    }

    // Print missing required extensions
    if (!requiredExtensionsSet.empty()) {
        std::cout << "Missing required extensions:" << std::endl;
        for (const auto& extension : requiredExtensionsSet) {
            std::cout << "  " << extension << std::endl;
        }
        return false;
    }

    return true;
}

// Choose swap surface format
vk::SurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    // Look for SRGB format
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    // If not found, return first available format
    return availableFormats[0];
}

// Choose swap present mode
vk::PresentModeKHR Renderer::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    // Look for mailbox mode (triple buffering)
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    // If not found, return FIFO mode (guaranteed to be available)
    return vk::PresentModeKHR::eFifo;
}

// Choose swap extent
vk::Extent2D Renderer::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    else {
        // Get framebuffer size
        int width, height;
        platform->GetWindowSize(&width, &height);

        // Create extent
        vk::Extent2D actualExtent = {
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height)
        };

        // Clamp to min/max extent
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

vk::Format Renderer::findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
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

vk::Format Renderer::findDepthFormat() {
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

bool Renderer::hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void Renderer::generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
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
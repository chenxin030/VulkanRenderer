#include <Renderer.h>
#include <set>
#include <fstream>

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
        platform.GetWindowSize(&width, &height);

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

#include "VulkanBase.h"

bool VulkanBase::createSwapChain()
{
    try
    {
        vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

        std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(availableFormats);

        std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(availablePresentModes);

        vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface,
            .minImageCount = minImageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
            .preTransform = surfaceCapabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = nullptr
        };

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::array<uint32_t, 2> queueFamilyIndicesLoc = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndicesLoc.size());
            createInfo.pQueueFamilyIndices = queueFamilyIndicesLoc.data();
        }
        else
        {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        swapChain = vk::raii::SwapchainKHR(device, createInfo);
        swapChainImages = swapChain.getImages();
        swapChainImageLayouts.assign(swapChainImages.size(), vk::ImageLayout::eUndefined);
        swapChainImageFormat = surfaceFormat.format;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create swap chain: " << e.what() << std::endl;
        return false;
    }
}

void VulkanBase::cleanupSwapChain()
{
    swapChainImageViews.clear();
    swapChain = vk::raii::SwapchainKHR(nullptr);
}

void VulkanBase::recreateSwapChain()
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(platform->window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(platform->window, &width, &height);
        glfwWaitEvents();
    }

    // Ensure no pending work still references old swapchain resources.
    device.waitIdle();

    cleanupSwapChain();
    createSwapChain();
    createImageViews();

    // Depth attachment must be recreated to match new swapchain extent.
    depthData.textureImageView = vk::raii::ImageView(nullptr);
    depthData.textureImage = vk::raii::Image(nullptr);
    depthData.textureImageMemory = vk::raii::DeviceMemory(nullptr);
    if (!createDepthResources())
    {
        throw std::runtime_error("failed to recreate depth resources");
    }

    // renderFinishedSemaphores is indexed by acquired swapchain image index.
    // Rebuild it whenever swapchain image count changes after a resize.
    renderFinishedSemaphores.clear();
    renderFinishedSemaphores.reserve(swapChainImages.size());
    vk::SemaphoreCreateInfo semaphoreInfo{};
    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
    }

    // UI buffers are commonly managed per swapchain image.
    uiFrameBuffers.resize(swapChainImages.size());

    currentFrame = 0;
}

bool VulkanBase::createImageViews()
{
    try
    {
        vk::ImageViewCreateInfo createInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainImageFormat,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        swapChainImageViews.clear();
        swapChainImageViews.reserve(swapChainImages.size());
        for (const auto& image : swapChainImages)
        {
            createInfo.image = image;
            swapChainImageViews.emplace_back(device, createInfo);
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create image views: " << e.what() << std::endl;
        return false;
    }
}


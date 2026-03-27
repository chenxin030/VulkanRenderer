#include "VulkanBase.h"

#include <set>
#include <map>

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallbackVkHpp(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    [[maybe_unused]] vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    [[maybe_unused]] void* pUserData)
{
    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
        std::cerr << "Validation layer Warning/Error: " << pCallbackData->pMessage << std::endl << std::endl;
    }
    else
    {
        std::cout << "Validation layer Verbose/Info: " << pCallbackData->pMessage << std::endl << std::endl;
    }
    return vk::False;
}

bool VulkanBase::initVulkan(const std::string& appName) {
    if (!createInstance(appName)) return false;
    if (!setupDebugMessenger()) return false;
    if (!createSurface()) return false;
    if (!pickPhysicalDevice()) return false;
    if (!createLogicalDevice()) return false;
    if (!createSwapChain()) return false;
    if (!createImageViews()) return false;
    if (!createCommandPool()) return false;
    if (!createCommandBuffers()) return false;
    if (!createSyncObjects()) return false;
}

bool VulkanBase::createInstance(const std::string& appName)
{
    try
    {
        vk::ApplicationInfo appInfo{
            .pApplicationName = appName.c_str(),
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "VulkanRenderer",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3
        };

        std::vector<const char*> extensions;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        vk::ValidationFeaturesEXT validationFeatures{};
        std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;

        if (enableValidationLayers)
        {
            if (!checkValidationLayerSupport())
            {
                std::cerr << "Validation layers requested, but not available" << std::endl;
                return false;
            }

            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size());
            validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();

            createInfo.pNext = &validationFeatures;
        }

        instance = vk::raii::Instance(context, createInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create instance: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanBase::setupDebugMessenger()
{
    if (!enableValidationLayers)
    {
        return true;
    }

    try
    {
        vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
        createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;

        createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

        createInfo.pfnUserCallback = &debugCallbackVkHpp;

        debugMessenger = vk::raii::DebugUtilsMessengerEXT(instance, createInfo);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to set up debug messenger: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanBase::createSurface()
{
    try
    {
        VkSurfaceKHR rawSurface{};
        if (!platform->CreateVulkanSurface(*instance, &rawSurface))
        {
            std::cerr << "Failed to create window surface" << std::endl;
            return false;
        }

        surface = vk::raii::SurfaceKHR(instance, rawSurface);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create surface: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanBase::pickPhysicalDevice()
{
    try
    {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        if (devices.empty())
        {
            std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
            return false;
        }

        std::multimap<int, vk::raii::PhysicalDevice> suitableDevices;

        for (auto& candidate : devices)
        {
            vk::PhysicalDeviceProperties props = candidate.getProperties();

            if (props.apiVersion < VK_API_VERSION_1_3)
            {
                continue;
            }

            QueueFamilyIndices indices = findQueueFamilies(candidate);
            if (!indices.isComplete())
            {
                continue;
            }

            if (!checkDeviceExtensionSupport(candidate))
            {
                continue;
            }

            SwapChainSupportDetails swapSupport = querySwapChainSupport(candidate);
            if (swapSupport.formats.empty() || swapSupport.presentModes.empty())
            {
                continue;
            }

            auto features = candidate.getFeatures2<vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan11Features,
                vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
            bool supportsRequiredFeatures =
                features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;
            if (!supportsRequiredFeatures)
            {
                continue;
            }

            int score = 0;
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            {
                score += 1000;
            }
            else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            {
                score += 100;
            }

            vk::PhysicalDeviceMemoryProperties memProps = candidate.getMemoryProperties();
            for (uint32_t i = 0; i < memProps.memoryHeapCount; i++)
            {
                if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                {
                    score += static_cast<int>(memProps.memoryHeaps[i].size / (1024ull * 1024ull * 1024ull));
                    break;
                }
            }

            suitableDevices.emplace(score, candidate);
        }

        if (suitableDevices.empty())
        {
            std::cerr << "Failed to find a suitable GPU" << std::endl;
            return false;
        }

        physicalDevice = suitableDevices.rbegin()->second;
        queueFamilyIndices = findQueueFamilies(physicalDevice);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to pick physical device: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanBase::createLogicalDevice()
{
    try
    {
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value(),
            queueFamilyIndices.computeFamily.value(),
            queueFamilyIndices.transferFamily.value()
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            vk::DeviceQueueCreateInfo queueCreateInfo{
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures supportedFeatures = physicalDevice.getFeatures();
        if (!supportedFeatures.samplerAnisotropy)
        {
            std::cout << "Warning: samplerAnisotropy not supported" << std::endl;
        }

        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
            featureChain = {
                {.features = {.samplerAnisotropy = true}},
                {.shaderDrawParameters = VK_TRUE },
                {.synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE },
                {.extendedDynamicState = VK_TRUE}
        };

        vk::DeviceCreateInfo createInfo{
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = nullptr
        };

        device = vk::raii::Device(physicalDevice, createInfo);

        graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
        presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
        computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);
        transferQueue = vk::raii::Queue(device, queueFamilyIndices.transferFamily.value(), 0);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create logical device: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanBase::checkValidationLayerSupport() const
{
    std::vector<vk::LayerProperties> availableLayers = context.enumerateInstanceLayerProperties();
    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }
        if (!layerFound)
        {
            return false;
        }
    }
    return true;
}


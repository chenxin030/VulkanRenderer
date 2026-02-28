#include <Renderer.h>
#include <set>

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallbackVkHpp(
	vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	[[maybe_unused]] vk::DebugUtilsMessageTypeFlagsEXT messageType,
	const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	[[maybe_unused]] void* pUserData) {
	if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
		std::cerr << "Validation layer Warning/Error: " << pCallbackData->pMessage << std::endl << std::endl;
	}
	else {
		std::cout << "Validation layer Verbose/Info: " << pCallbackData->pMessage << std::endl <<std::endl;
	}
	return vk::False;
}

Renderer::Renderer() {
	deviceExtensions = requiredDeviceExtensions;
}

bool Renderer::createInstance(const std::string& appName) {
	try {
		// Create application info
		vk::ApplicationInfo appInfo{
		  .pApplicationName = appName.c_str(),
		  .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		  .pEngineName = "Simple Engine",
		  .engineVersion = VK_MAKE_VERSION(1, 0, 0),
		  .apiVersion = VK_API_VERSION_1_3
		};

		// Get required extensions
		std::vector<const char*> extensions;

		// Add required extensions for GLFW
#if defined(PLATFORM_DESKTOP)
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
#endif

		// Add debug extension if validation layers are enabled
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		// Create instance info
		vk::InstanceCreateInfo createInfo{
		  .pApplicationInfo = &appInfo,
		  .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
		  .ppEnabledExtensionNames = extensions.data()
		};

		// Enable validation layers if requested
		vk::ValidationFeaturesEXT validationFeatures{};
		std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;

		if (enableValidationLayers) {
			if (!checkValidationLayerSupport()) {
				std::cerr << "Validation layers requested, but not available" << std::endl;
				return false;
			}

			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			// Keep validation output quiet by default (no DebugPrintf feature).
			// Ray Query debugPrintf/printf diagnostics are intentionally removed.

			validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size());
			validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();

			createInfo.pNext = &validationFeatures;
		}

		// Create instance
		instance = vk::raii::Instance(context, createInfo);
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create instance: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::setupDebugMessenger() {
	if (!enableValidationLayers) {
		return true;
	}

	try {
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
	catch (const std::exception& e) {
		std::cerr << "Failed to set up debug messenger: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createSurface() {
	try {
		VkSurfaceKHR _surface;
		if (!platform->CreateVulkanSurface(*instance, &_surface)) {
			std::cerr << "Failed to create window surface" << std::endl;
			return false;
		}

		surface = vk::raii::SurfaceKHR(instance, _surface);
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create surface: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::pickPhysicalDevice() {
	try {
		// Get available physical devices
		std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

		if (devices.empty()) {
			std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
			return false;
		}

		// 优先使用独立显卡（如 NVIDIA RTX 2080），而不是集成显卡（如 Intel UHD Graphics）
		// First, collect all suitable devices with their suitability scores
		std::multimap<int, vk::raii::PhysicalDevice> suitableDevices;

		for (auto& _device : devices) {
			// Print device properties for debugging
			vk::PhysicalDeviceProperties deviceProperties = _device.getProperties();
			std::cout << "Checking device: " << deviceProperties.deviceName
				<< " (Type: " << vk::to_string(deviceProperties.deviceType) << ")" << std::endl;

			// Check if the device supports Vulkan 1.3
			bool supportsVulkan1_3 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
			if (!supportsVulkan1_3) {
				std::cout << "  - Does not support Vulkan 1.3" << std::endl;
				continue;
			}

			// Check queue families
			QueueFamilyIndices indices = findQueueFamilies(_device);
			bool supportsGraphics = indices.isComplete();
			if (!supportsGraphics) {
				std::cout << "  - Missing required queue families" << std::endl;
				continue;
			}

			// Check device extensions
			bool supportsAllRequiredExtensions = checkDeviceExtensionSupport(_device);
			if (!supportsAllRequiredExtensions) {
				std::cout << "  - Missing required extensions" << std::endl;
				continue;
			}

			// Check swap chain support
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_device);
			bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			if (!swapChainAdequate) {
				std::cout << "  - Inadequate swap chain support" << std::endl;
				continue;
			}

			// Check for required features
			auto features = _device.getFeatures2<vk::PhysicalDeviceFeatures2,
				vk::PhysicalDeviceVulkan11Features,
				vk::PhysicalDeviceVulkan13Features,
				vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
			bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
				features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
				features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
				features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;
			if (!supportsRequiredFeatures) {
				std::cout << "  - Does not support required features" << std::endl;
				continue;
			}

			// Calculate suitability score - prioritize discrete GPUs
			int score = 0;

			// Discrete GPUs get the highest priority (NVIDIA RTX 2080, AMD, etc.)
			if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
				score += 1000;
				std::cout << "  - Discrete GPU: +1000 points" << std::endl;
			}
			// Integrated GPUs get lower priority (Intel UHD Graphics, etc.)
			else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
				score += 100;
				std::cout << "  - Integrated GPU: +100 points" << std::endl;
			}

			// Add points for memory size (more VRAM is better)
			vk::PhysicalDeviceMemoryProperties memProperties = _device.getMemoryProperties();
			for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
				if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
					// Add 1 point per GB of VRAM
					score += static_cast<int>(memProperties.memoryHeaps[i].size / (1024 * 1024 * 1024));
					break;
				}
			}

			std::cout << "  - Device is suitable with score: " << score << std::endl;
			suitableDevices.emplace(score, _device);
		}

		if (!suitableDevices.empty()) {
			// Select the device with the highest score (discrete GPU with most VRAM)
			physicalDevice = suitableDevices.rbegin()->second;
			vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
			std::cout << "Selected device: " << deviceProperties.deviceName
				<< " (Type: " << vk::to_string(deviceProperties.deviceType)
				<< ", Score: " << suitableDevices.rbegin()->first << ")" << std::endl;

			// Store queue family indices for the selected device
			queueFamilyIndices = findQueueFamilies(physicalDevice);

			return true;
		}
		std::cerr << "Failed to find a suitable GPU. Make sure your GPU supports Vulkan and has the required extensions." << std::endl;
		return false;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to pick physical device: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createLogicalDevice() {
    try {
        // Create queue create info for each unique queue family
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set uniqueQueueFamilies = {
          queueFamilyIndices.graphicsFamily.value(),
          queueFamilyIndices.presentFamily.value(),
          queueFamilyIndices.computeFamily.value(),
          queueFamilyIndices.transferFamily.value()
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{
              .queueFamilyIndex = queueFamily,
              .queueCount = 1,
              .pQueuePriorities = &queuePriority
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Enable required features (now verified to be supported)
		vk::PhysicalDeviceFeatures supportedFeatures = physicalDevice.getFeatures();
		if (!supportedFeatures.samplerAnisotropy) {  // 如果不需要纹理，这行可以删除
			std::cout << "Warning: samplerAnisotropy not supported" << std::endl;
		}
		// 基础特性配置
		vk::StructureChain<
			vk::PhysicalDeviceFeatures2,
			vk::PhysicalDeviceVulkan11Features,
			vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
			featureChain = {
				{},											// vk::PhysicalDeviceFeatures2
				{.shaderDrawParameters = VK_TRUE },			// vk::PhysicalDeviceVulkan11Features
				{.synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE },				// vk::PhysicalDeviceVulkan13Features
				{.extendedDynamicState = VK_TRUE}				// vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
		};
		//vk::PhysicalDeviceFeatures deviceFeatures{};
		//deviceFeatures.samplerAnisotropy = VK_TRUE;  // 纹理需要，保留
		//deviceFeatures.depthClamp = VK_TRUE;         // 深度测试，3D需要

  //      // Vulkan 1.3特性
		//vk::PhysicalDeviceVulkan13Features vulkan13Features{};
		//vulkan13Features.dynamicRendering = VK_TRUE;     // 简化渲染流程
		//vulkan13Features.synchronization2 = VK_TRUE;     // 简化同步

		//// 特性链
		//auto features = vk::PhysicalDeviceFeatures2()
		//	.setFeatures(deviceFeatures)
		//	.setPNext(&vulkan13Features);

        vk::DeviceCreateInfo createInfo{
          .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),	// 不能直接用.pNext = &featureChain（很神秘）
          .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
          .pQueueCreateInfos = queueCreateInfos.data(),
          .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
          .ppEnabledExtensionNames = deviceExtensions.data(),
          .pEnabledFeatures = nullptr // Using pNext for features
        };

        device = vk::raii::Device(physicalDevice, createInfo);

        // Get queue handles
        graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
        presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
        computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);
        transferQueue = vk::raii::Queue(device, queueFamilyIndices.transferFamily.value(), 0);

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create logical device: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createGraphicsPipeline() {
	try
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto                                     bindingDescription = Vertex::getBindingDescription();
		auto                                     attributeDescriptions = Vertex::getAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo   vertexInputInfo{ 
			.vertexBindingDescriptionCount = 1, 
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()), 
			.pVertexAttributeDescriptions = attributeDescriptions.data() 
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList };
		vk::PipelineViewportStateCreateInfo      viewportState{ .viewportCount = 1, .scissorCount = 1 };

		vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .depthBiasSlopeFactor = 1.0f, .lineWidth = 1.0f };

		vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = vk::False,
																   .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

		vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

		std::vector dynamicStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0 };

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = 2,
				.pStages = shaderStages,
				.pVertexInputState = &vertexInputInfo,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pColorBlendState = &colorBlending,
				.pDynamicState = &dynamicState,
				.layout = pipelineLayout,
				.renderPass = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &swapChainImageFormat
			}
		};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create graphics pipeline: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createSyncObjects()
{
	try {
		// Resize semaphores and fences vectors
		presentCompleteSemaphores.clear();
		renderFinishedSemaphores.clear();
		inFlightFences.clear();

		// Semaphores per swapchain image (indexed by imageIndex from acquireNextImage)
		// The presentation engine holds semaphores until the image is re-acquired, so we need
		// one semaphore per swapchain image to avoid reuse conflicts. See Vulkan spec:
		// https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
		const auto semaphoreCount = static_cast<uint32_t>(swapChainImages.size());
		presentCompleteSemaphores.reserve(semaphoreCount);
		renderFinishedSemaphores.reserve(semaphoreCount);

		// Fences per frame-in-flight for CPU-GPU synchronization (indexed by currentFrame)
		inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

		// Create semaphore info
		vk::SemaphoreCreateInfo semaphoreInfo{};

		// Create semaphores per swapchain image (indexed by imageIndex for presentation sync)
		for (uint32_t i = 0; i < semaphoreCount; i++) {
			presentCompleteSemaphores.emplace_back(device, semaphoreInfo);
			renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
		}

		// Create fences per frame-in-flight (indexed by currentFrame for CPU-GPU pacing)
		vk::FenceCreateInfo fenceInfo{
		  .flags = vk::FenceCreateFlagBits::eSignaled
		};
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			inFlightFences.emplace_back(device, fenceInfo);
		}

		// Ensure uploads timeline semaphore exists (created early in createLogicalDevice)
		// No action needed here unless reinitializing after swapchain recreation.
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create sync objects: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::checkValidationLayerSupport() const {
	// Get available layers
	std::vector<vk::LayerProperties> availableLayers = context.enumerateInstanceLayerProperties();

	// Check if all requested layers are available
	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

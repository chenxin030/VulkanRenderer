#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <map>
#include <Platform.h>

const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation" 
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> computeFamily;
	std::optional<uint32_t> transferFamily; // optional dedicated transfer queue family

	[[nodiscard]] bool isComplete() const {
		return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
	}
};
struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

struct Renderer {

	Renderer(Platform& _platform);

	vk::raii::Context  context;
	vk::raii::Instance instance = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

	vk::raii::PhysicalDevice physicalDevice = nullptr;
	vk::raii::Device         device = nullptr;

	// Queue family indices
	QueueFamilyIndices queueFamilyIndices;
	vk::raii::Queue graphicsQueue = nullptr;
	vk::raii::Queue presentQueue  = nullptr;
	vk::raii::Queue computeQueue  = nullptr;
	vk::raii::Queue transferQueue = nullptr;

	vk::raii::SurfaceKHR surface = nullptr;

	// Required device extensions
	const std::vector<const char*> requiredDeviceExtensions = {
	  VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};
	// All device extensions (required + optional)
	std::vector<const char*> deviceExtensions;

	vk::raii::SwapchainKHR           swapChain = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::Format						 swapChainImageFormat = vk::Format::eUndefined;
	vk::Extent2D                     swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	// Tracked layouts for swapchain images (VVL requires correct oldLayout in barriers).
	// Initialized at swapchain creation and updated as we transition.
	std::vector<vk::ImageLayout> swapChainImageLayouts;

	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline       graphicsPipeline = nullptr;

	vk::raii::CommandPool    commandPool = nullptr;
	vk::raii::CommandBuffer  commandBuffer = nullptr;

	vk::raii::Semaphore presentCompleteSemaphore = nullptr;
	vk::raii::Semaphore renderFinishedSemaphore = nullptr;
	vk::raii::Fence     drawFence = nullptr;

	bool initVulkan()
	{
		if (!createInstance("Vulkan Renderer")) {
			std::cerr << "Failed to create Vulkan instance" << std::endl;
			return false;
		}
		if (!setupDebugMessenger()) {
			std::cerr << "Failed to setup debug messenger" << std::endl;
			return false;
		}
		if (!createSurface()) {
			std::cerr << "Failed to create surface" << std::endl;
			return false;
		}
		if (!pickPhysicalDevice()) {
			std::cerr << "Failed to pick physical device" << std::endl;
			return false;
		}
		if (!createLogicalDevice()) {
			std::cerr << "Failed to create logical device" << std::endl;
			return false;
		}
		if (!createSwapChain()) {
			std::cerr << "Failed to create swap chain" << std::endl;
			return false;
		}
		if (!createImageViews()) {
			std::cerr << "Failed to create image views" << std::endl;
			return false;
		}
		if (!createGraphicsPipeline()) {
			std::cerr << "Failed to create graphics pipeline" << std::endl;
			return false;
		}
		if (!createCommandPool()) {
			std::cerr << "Failed to create command pool" << std::endl;
			return false;
		}
		if (!createCommandBuffers()) {
			std::cerr << "Failed to create command buffers" << std::endl;
			return false;
		}
		if (!createSyncObjects()) {
			std::cerr << "Failed to create sync objects" << std::endl;
			return false;
		}

		return true;
	}

	void render()
	{
		drawFrame();
	}

	bool createInstance(const std::string& appName);
	bool setupDebugMessenger();
	bool createSurface();
	bool pickPhysicalDevice();
	bool createLogicalDevice();
	bool createSwapChain();
	bool createImageViews();
	bool createGraphicsPipeline();
	bool createCommandPool();
	bool createCommandBuffers();
	bool createSyncObjects();

	void recordCommandBuffer(uint32_t imageIndex);

	void transition_image_layout(
		uint32_t                imageIndex,
		vk::ImageLayout         old_layout,
		vk::ImageLayout         new_layout,
		vk::AccessFlags2        src_access_mask,
		vk::AccessFlags2        dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask);

	void drawFrame()
	{
		graphicsQueue.waitIdle();        // NOTE: for simplicity, wait for the queue to be idle before starting the frame
		// In the next chapter you see how to use multiple frames in flight and fences to sync

		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore, nullptr);
		recordCommandBuffer(imageIndex);

		device.resetFences(*drawFence);
		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo   submitInfo{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*presentCompleteSemaphore, .pWaitDstStageMask = &waitDestinationStageMask, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer, .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphore };
		graphicsQueue.submit(submitInfo, *drawFence);
		result = device.waitForFences(*drawFence, vk::True, UINT64_MAX);
		if (result != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		const vk::PresentInfoKHR presentInfoKHR{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphore, .swapchainCount = 1, .pSwapchains = &*swapChain, .pImageIndices = &imageIndex };
		result = presentQueue.presentKHR(presentInfoKHR);
		switch (result)
		{
		case vk::Result::eSuccess:
			break;
		case vk::Result::eSuboptimalKHR:
			std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR !\n";
			break;
		default:
			break;        // an unexpected result is returned!
		}
	}


	std::vector<const char*> getRequiredInstanceExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		auto     glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}

		return extensions;
	}

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT * pCallbackData, void*)
	{
		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
		}

		return vk::False;
	}

	bool checkValidationLayerSupport() const;

	std::vector<char> readFile(const std::string& filename);
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);

	QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device);
	SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device);
	bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice& device);

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

	Platform& platform;

};
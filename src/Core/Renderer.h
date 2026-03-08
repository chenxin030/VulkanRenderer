#pragma once

#include "ResourceManager.h"
#include "Platform.h"

#include <map>

constexpr int MAX_OBJECTS = 3;

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

	Renderer();

	const uint32_t MAX_FRAMES_IN_FLIGHT = 2u;
	// Current frame index
	uint32_t currentFrame = 0;

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

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline       graphicsPipeline = nullptr;

	vk::raii::CommandPool    commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;

	vk::raii::DescriptorPool             descriptorPool = nullptr;

	bool framebufferResized = false;

	TextureData depthData;

	void initialize(Platform* _platform, ResourceManager* _resourceManager) {
		platform = _platform;
		resourceManager = _resourceManager;
		glfwSetWindowUserPointer(platform->window, this);
		glfwSetFramebufferSizeCallback(platform->window, [](GLFWwindow* w, int width, int height) {
			auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(w));
			renderer->framebufferResized = true;
		});
	}

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
		if (!createDescriptorSetLayout()) {
			std::cerr << "Failed to create descriptor set layout" << std::endl;
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
		if (!createDepthResources()) {
			std::cerr << "Failed to create depth resources" << std::endl;
			return false;
		}
		if (!createDescriptorPool()) {
			std::cerr << "Failed to create descriptor pool" << std::endl;
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
	void loadResource() {
		loadModels();
		loadTextures();
	}

	void render()
	{
		try {
			// prepareFrame
			auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
			if (fenceResult != vk::Result::eSuccess)
			{
				throw std::runtime_error("failed to wait for fence!");
			}
			auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
			if (result == vk::Result::eErrorOutOfDateKHR)
			{
				recreateSwapChain();
				return;
			}
			if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
			{
				assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
				throw std::runtime_error("failed to acquire swap chain image!");
			}
			updateUniformBuffer(currentFrame);

			device.resetFences(*inFlightFences[currentFrame]);

			commandBuffers[currentFrame].reset();
			recordCommandBuffer(imageIndex);

			// submitFrame()
			vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
			const vk::SubmitInfo   submitInfo{ .waitSemaphoreCount = 1,
											  .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
											  .pWaitDstStageMask = &waitDestinationStageMask,
											  .commandBufferCount = 1,
											  .pCommandBuffers = &*commandBuffers[currentFrame],
											  .signalSemaphoreCount = 1,
											  .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex] };
			graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

			const vk::PresentInfoKHR presentInfoKHR{ .waitSemaphoreCount = 1,
													.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
													.swapchainCount = 1,
													.pSwapchains = &*swapChain,
													.pImageIndices = &imageIndex };
			result = presentQueue.presentKHR(presentInfoKHR);

			if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
			{
				framebufferResized = false;
				recreateSwapChain();
			}
			else
			{
				// There are no other success codes than eSuccess; on any error code, presentKHR already threw an exception.
				assert(result == vk::Result::eSuccess);
			}
			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}
		catch (const vk::OutOfDateKHRError& e) {
			framebufferResized = false;
			recreateSwapChain();
		}
		
	}

	void cleanup() {
		cleanupUBO();
	}

	bool createInstance(const std::string& appName);
	bool setupDebugMessenger();
	bool createSurface();
	bool pickPhysicalDevice();
	bool createLogicalDevice();
	bool createSwapChain();
	void cleanupSwapChain();
	void recreateSwapChain();
	bool createImageViews();
	bool createDescriptorSetLayout();
	bool createGraphicsPipeline();
	bool createDescriptorPool();
	void createDescriptorSets();
	bool createCommandPool();
	bool createCommandBuffers();
	bool createSyncObjects();
	bool createDepthResources();

	void recordCommandBuffer(uint32_t imageIndex);

	void transition_image_layout(
		vk::Image               image,
		vk::ImageLayout         old_layout,
		vk::ImageLayout         new_layout,
		vk::AccessFlags2        src_access_mask,
		vk::AccessFlags2        dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags    image_aspect_flags);

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

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);

	std::vector<char> readFile(const std::string& filename);
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);

	QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device);
	SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device);
	static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
	bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice& device);

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

	void createVertexBuffer(Mesh& mesh);
	void createIndexBuffer(Mesh& mesh);
	void createUniformBuffers(MeshUniformBuffer& entityResource);
	void updateUniformBuffer(uint32_t currentImage);

	void waitIdle() {
		device.waitIdle();
	}
	void loadModels();
	void loadTextures();
	void LoadTextureFromFile(const std::string& path, TextureData& texData);
	void cleanupUBO();
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData);
	vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);
	vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
	vk::Format findDepthFormat();
	bool hasStencilComponent(vk::Format format);
	void createTextureSampler(vk::raii::Sampler& textureSampler);
	void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels);
	void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height);

	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
	void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);

	void generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

	Platform* platform;
	ResourceManager* resourceManager;

};
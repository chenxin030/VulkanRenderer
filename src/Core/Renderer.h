#pragma once

#include "RenderConfig.h"
#include "ResourceManager.h"
#include "Scene.h"
#include "Platform.h"
#include "Camera.h"

#include <map>
#include <vector>

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

	QueueFamilyIndices queueFamilyIndices;
	vk::raii::Queue graphicsQueue = nullptr;
	vk::raii::Queue presentQueue = nullptr;
	vk::raii::Queue computeQueue = nullptr;
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
	std::vector<vk::ImageLayout> swapChainImageLayouts;

	vk::raii::CommandPool    commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence> inFlightFences;

#if RENDERING_LEVEL == 1
	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::DescriptorPool      descriptorPool = nullptr;
	vk::raii::PipelineLayout      pipelineLayout = nullptr;
	vk::raii::Pipeline            graphicsPipeline = nullptr;
#elif RENDERING_LEVEL == 2
	// Instanced rendering resources
	vk::raii::DescriptorSetLayout instancedDescriptorSetLayout = nullptr;
	vk::raii::DescriptorPool      instancedDescriptorPool = nullptr;
	vk::raii::PipelineLayout      instancedPipelineLayout = nullptr;
	vk::raii::Pipeline            instancedPipeline = nullptr;

	MeshBuffer instancedBufferResources;
	MeshBuffer globalUboResources;
#elif RENDERING_LEVEL == 3
	// PBR Instanced rendering resources
	vk::raii::DescriptorSetLayout pbrDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      pbrPipelineLayout = nullptr;
	vk::raii::Pipeline            pbrPipeline = nullptr;
	vk::raii::DescriptorPool      pbrDescriptorPool = nullptr;

	MeshBuffer pbrInstanceBufferResources;
	MeshBuffer sceneUboResources;
	MeshBuffer lightUboResources;
#elif RENDERING_LEVEL == 4
	// IBL PBR + skybox resources
	vk::raii::DescriptorSetLayout iblPbrDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      iblPbrPipelineLayout = nullptr;
	vk::raii::Pipeline            iblPbrPipeline = nullptr;
	vk::raii::DescriptorPool      iblPbrDescriptorPool = nullptr;

	MeshBuffer pbrInstanceBufferResources;
	MeshBuffer sceneUboResources;
	MeshBuffer lightUboResources;
	MeshBuffer paramsUboResources;
	MeshBuffer skyboxUboResources;

	vk::raii::DescriptorSetLayout skyboxDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      skyboxPipelineLayout = nullptr;
	vk::raii::Pipeline            skyboxPipeline = nullptr;
	vk::raii::DescriptorPool      skyboxDescriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> skyboxDescriptorSets;

	Mesh skyboxTriangleMesh;

	TextureData envCubemapData;
	TextureData irradianceCubemapData;
	TextureData prefilteredEnvMapData;
	TextureData brdfLutData;
#elif RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6 || RENDERING_LEVEL == 7
	// Shadow mapping resources (Level 5) + TAAU base (Level 6) + SSR base (Level 7)
	vk::raii::DescriptorSetLayout shadowDescriptorSetLayout = nullptr;
	vk::raii::DescriptorPool      shadowDescriptorPool = nullptr;
	vk::raii::PipelineLayout      shadowPipelineLayout = nullptr;
	vk::raii::Pipeline            shadowDepthPipeline = nullptr;
	vk::raii::Pipeline            shadowLitPipeline = nullptr;

	MeshBuffer shadowInstanceBufferResources;
	MeshBuffer sceneUboResources;
	MeshBuffer shadowUboResources;
	MeshBuffer shadowParamsUboResources;

	TextureData shadowMapData;
	vk::Extent2D shadowMapExtent{ 2048u, 2048u };
	vk::ImageLayout shadowMapLayout = vk::ImageLayout::eUndefined;


	int shadowFilterMode = 2;
	float pcfRadiusTexels = 2.0f;
	float pcssLightSizeTexels = 25.0f;

	float dirLightIntensity = 0.5f;
	float pointLightIntensity = 3.5f;
	float areaLightIntensity = 2.5f;

	bool uiEnabled = true;

	vk::raii::DescriptorSetLayout uiDescriptorSetLayout = nullptr;
	vk::raii::DescriptorPool uiDescriptorPool = nullptr;
	vk::raii::PipelineLayout uiPipelineLayout = nullptr;
	vk::raii::Pipeline uiPipeline = nullptr;
	vk::raii::DescriptorSets uiDescriptorSets = nullptr;
	TextureData uiFontTexture;

	struct UiFrameBuffers
	{
		vk::raii::Buffer vertexBuffer = nullptr;
		vk::raii::DeviceMemory vertexBufferMemory = nullptr;
		void* vertexMapped = nullptr;
		size_t vertexSize = 0;

		vk::raii::Buffer indexBuffer = nullptr;
		vk::raii::DeviceMemory indexBufferMemory = nullptr;
		void* indexMapped = nullptr;
		size_t indexSize = 0;
	};
	std::vector<UiFrameBuffers> uiFrameBuffers;
#endif
#if RENDERING_LEVEL == 7
	// Screen-space reflection resources
	vk::raii::DescriptorSetLayout ssrDescriptorSetLayout = nullptr;
	vk::raii::DescriptorPool ssrDescriptorPool = nullptr;
	vk::raii::PipelineLayout ssrPipelineLayout = nullptr;
	vk::raii::Pipeline ssrPipeline = nullptr;
	vk::raii::DescriptorSets ssrDescriptorSets = nullptr;
	MeshBuffer ssrSceneUboResources;
	MeshBuffer ssrParamsUboResources;
	TextureData ssrColorData;
	vk::ImageLayout ssrColorLayout = vk::ImageLayout::eUndefined;
	vk::raii::Sampler ssrColorSampler = nullptr;
	vk::raii::Sampler ssrDepthSampler = nullptr;
	int ssrMaxSteps = 64;
	float ssrMaxRayDistance = 16.0f;
	float ssrThickness = 0.12f;
	float ssrStride = 0.25f;
	float ssrIntensity = 0.5f;
	bool ssrEnabled = true;
#endif

	bool framebufferResized = false;

	TextureData depthData;
	vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;
#if RENDERING_LEVEL < 3 || RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6 || RENDERING_LEVEL == 7
	Camera camera = Camera(glm::vec3(0.0f, 0.0f, 5.0f));
#else
	Camera camera = Camera(glm::vec3(0.0f, -1.0f, 13.0f));
#endif

	void initialize(Platform* _platform, ResourceManager* _resourceManager, Scene* _scene) {
		platform = _platform;
		resourceManager = _resourceManager;
		scene = _scene;
		maxInstances = _scene ? _scene->getMaxInstances() : 0;

		platform->resizeCallback = [this](int width, int height) {
			framebufferResized = true;
			};

		// Set mouse callback
		platform->mouseCallback = [this](float xpos, float ypos, uint32_t button) {
			if (platform->rightMouseButtonPressed) {
				if (platform->firstMouse) {
					platform->lastX = xpos;
					platform->lastY = ypos;
					platform->firstMouse = false;
				}

				float xoffset = xpos - (float)platform->lastX;
				float yoffset = (float)platform->lastY - ypos; // reversed since y-coordinates go from bottom to top

				platform->lastX = xpos;
				platform->lastY = ypos;

				camera.ProcessMouseMovement(xoffset, yoffset);
			}
			};

		// Set scroll callback
		platform->scrollCallback = [this](double xoffset, double yoffset) {
			camera.ProcessMouseScroll((float)yoffset);
			};
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
#if RENDERING_LEVEL == 1
		const uint32_t instanceCount = scene ? scene->getActiveInstanceCount() : 0;
		for (auto& meshUniformBuffer : resourceManager->meshUniformBuffer) {
			createUniformBuffers(meshUniformBuffer, sizeof(MVP) * instanceCount);
		}
		if (!createDescriptorSetLayout()) {
			std::cerr << "Failed to create descriptor set layout" << std::endl;
			return false;
		}
		if (!createGraphicsPipeline()) {
			std::cerr << "Failed to create graphics pipeline" << std::endl;
			return false;
		}
		if (!createDescriptorPool()) {
			std::cerr << "Failed to create descriptor pool" << std::endl;
			return false;
		}
#elif RENDERING_LEVEL == 2
		// Initialize instanced rendering
		createInstancedBuffers();
		if (!createInstancedDescriptorSetLayout()) {
			std::cerr << "Failed to create Instanced DescriptorSetLayout" << std::endl;
			return false;
		}
		if (!createInstancedDescriptorPool()) {
			std::cerr << "Failed to create Instanced DescriptorPool" << std::endl;
			return false;
		}
		if (!createInstancedPipeline()) {
			std::cerr << "Failed to create Instanced Pipeline" << std::endl;
			return false;
		}
#elif RENDERING_LEVEL == 3
		// Initialize PBR instanced rendering
		createPBRBuffers();
		if (!createPBRDescriptorSetLayout()) {
			std::cerr << "Failed to create PBR DescriptorSetLayout" << std::endl;
			return false;
		}
		if (!createPBRDescriptorPool()) {
			std::cerr << "Failed to create PBR DescriptorPool" << std::endl;
			return false;
		}
		if (!createPBRPipeline()) {
			std::cerr << "Failed to create PBR Pipeline" << std::endl;
			return false;
		}
#elif RENDERING_LEVEL == 4
		// Initialize IBL PBR rendering
		createIBLPBRBuffers();
		if (!createIBLPBRDescriptorSetLayout()) {
			std::cerr << "Failed to create IBL PBR DescriptorSetLayout" << std::endl;
			return false;
		}
		if (!createIBLPBRDescriptorPool()) {
			std::cerr << "Failed to create IBL PBR DescriptorPool" << std::endl;
			return false;
		}
		if (!createIBLPBRPipeline()) {
			std::cerr << "Failed to create IBL PBR Pipeline" << std::endl;
			return false;
		}
		if (!createSkyboxDescriptorSetLayout()) {
			std::cerr << "Failed to create Skybox DescriptorSetLayout" << std::endl;
			return false;
		}
		if (!createSkyboxDescriptorPool()) {
			std::cerr << "Failed to create Skybox DescriptorPool" << std::endl;
			return false;
		}
		if (!createSkyboxPipeline()) {
			std::cerr << "Failed to create Skybox Pipeline" << std::endl;
			return false;
		}
#elif RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6 || RENDERING_LEVEL == 7
		// Initialize Shadow Mapping rendering (Level 5/6/7 base)
		createShadowBuffers();
		if (!createShadowDescriptorSetLayout()) {
			std::cerr << "Failed to create Shadow DescriptorSetLayout" << std::endl;
			return false;
		}
		if (!createShadowDescriptorPool()) {
			std::cerr << "Failed to create Shadow DescriptorPool" << std::endl;
			return false;
		}
		if (!createShadowMapResources()) {
			std::cerr << "Failed to create ShadowMap resources" << std::endl;
			return false;
		}
		if (!createShadowPipelines()) {
			std::cerr << "Failed to create Shadow pipelines" << std::endl;
			return false;
		}
#endif
		if (!createCommandPool()) {
			std::cerr << "Failed to create command pool" << std::endl;
			return false;
		}
#if RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6 || RENDERING_LEVEL == 7
		if (!initUI()) {
			std::cerr << "Failed to init UI" << std::endl;
			return false;
		}
#endif
		if (!createDepthResources()) {
			std::cerr << "Failed to create depth resources" << std::endl;
			return false;
		}
#if RENDERING_LEVEL == 7
		if (!createSSRResources()) {
			std::cerr << "Failed to create SSR resources" << std::endl;
			return false;
		}
		if (!createSSRDescriptorSetLayout()) {
			std::cerr << "Failed to create SSR DescriptorSetLayout" << std::endl;
			return false;
		}
		if (!createSSRDescriptorPool()) {
			std::cerr << "Failed to create SSR DescriptorPool" << std::endl;
			return false;
		}
		createSSRDescriptorSets();
		if (!createSSRPipeline()) {
			std::cerr << "Failed to create SSR Pipeline" << std::endl;
			return false;
		}
#endif
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
#if RENDERING_LEVEL == 4
		generateIBLResources();
#endif
	}

	void render()
	{
		try {
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
#if RENDERING_LEVEL == 1
			updateUniformBuffer(currentFrame);
#elif RENDERING_LEVEL == 2
			updateInstancedBuffers(currentFrame);
#elif RENDERING_LEVEL == 3
			updatePBRInstanceBuffers(currentFrame);
#elif RENDERING_LEVEL == 4
			updateIBLPBRBuffers(currentFrame);
#elif RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6 || RENDERING_LEVEL == 7
			updateUIFrame();
			updateShadowBuffers(currentFrame);
#endif
#if RENDERING_LEVEL == 7
			updateSSRBuffers(currentFrame);
#endif

			device.resetFences(*inFlightFences[currentFrame]);

			commandBuffers[currentFrame].reset();
			recordCommandBuffer(imageIndex);

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
#if RENDERING_LEVEL > 5
		shutdownUI();
#endif
		cleanupUBO();
	}

	void processInput(float deltaTime) {
		if (platform->rightMouseButtonPressed) {
			if (glfwGetKey(platform->window, GLFW_KEY_W) == GLFW_PRESS)
				camera.ProcessKeyboard(FORWARD, deltaTime);
			if (glfwGetKey(platform->window, GLFW_KEY_S) == GLFW_PRESS)
				camera.ProcessKeyboard(BACKWARD, deltaTime);
			if (glfwGetKey(platform->window, GLFW_KEY_A) == GLFW_PRESS)
				camera.ProcessKeyboard(LEFT, deltaTime);
			if (glfwGetKey(platform->window, GLFW_KEY_D) == GLFW_PRESS)
				camera.ProcessKeyboard(RIGHT, deltaTime);
			if (glfwGetKey(platform->window, GLFW_KEY_Q) == GLFW_PRESS)
				camera.ProcessKeyboard(UP, deltaTime);
			if (glfwGetKey(platform->window, GLFW_KEY_E) == GLFW_PRESS)
				camera.ProcessKeyboard(DOWN, deltaTime);
		}
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

	// Instanced rendering functions
	bool createInstancedDescriptorSetLayout();
	bool createInstancedDescriptorPool();
	void createInstancedDescriptorSets();
	bool createInstancedPipeline();
	void createInstancedBuffers();
	void updateInstancedBuffers(uint32_t currentImage);

	// PBR Instanced rendering functions
	bool createPBRDescriptorSetLayout();
	bool createPBRDescriptorPool();
	void createPBRDescriptorSets();
	bool createPBRPipeline();
	void createPBRBuffers();
	void updatePBRInstanceBuffers(uint32_t currentImage);

	// IBL PBR rendering functions
	bool createIBLPBRDescriptorSetLayout();
	bool createIBLPBRDescriptorPool();
	void createIBLPBRDescriptorSets();
	bool createIBLPBRPipeline();
	void createIBLPBRBuffers();
	void updateIBLPBRBuffers(uint32_t currentImage);

	// Shadow mapping rendering functions (Level 5-7)
	bool createShadowDescriptorSetLayout();
	bool createShadowDescriptorPool();
	void createShadowDescriptorSets();
	bool createShadowMapResources();
	bool createShadowPipelines();
	void createShadowBuffers();
	void updateShadowBuffers(uint32_t currentImage);
	void updateTAAUScene(float deltaTime);
	void updateTAAUHistory(const glm::mat4& currentViewProj);
	void updateTAAUUI();
	bool initUI();
	void shutdownUI();
	void updateUIFrame();
	void recordUI(vk::raii::CommandBuffer& commandBuffer);

	// SSR (Level 7)
	bool createSSRResources();
	bool createSSRDescriptorSetLayout();
	bool createSSRDescriptorPool();
	void createSSRDescriptorSets();
	bool createSSRPipeline();
	void updateSSRBuffers(uint32_t currentImage);
	void recordSSR(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
	void updateSSRUI();

	bool createSkyboxDescriptorSetLayout();
	bool createSkyboxDescriptorPool();
	void createSkyboxDescriptorSets();
	bool createSkyboxPipeline();

	void generateIBLResources();

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

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*)
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
	void createUniformBuffers(MeshBuffer& meshResource, vk::DeviceSize size);
	void createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size);
	void updateUniformBuffer(uint32_t currentImage);

	void waitIdle() {
		device.waitIdle();
	}
	void loadModels();
	void loadTextures();
	void LoadTextureFromFile(const std::string& path, TextureData& texData);
	void LoadHDRTextureFromFile(const std::string& path, TextureData& texData);
	void cleanupUBO();
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData);
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t arrayLayers, vk::ImageCreateFlags flags, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData);
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
	Scene* scene = nullptr;
	uint32_t maxInstances = 0;

};

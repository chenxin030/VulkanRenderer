#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
bool Renderer::createSwapChain() {
	try {
		vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent = chooseSwapExtent(surfaceCapabilities);
		uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

		std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
		vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(availableFormats);

		std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
		vk::PresentModeKHR              presentMode = chooseSwapPresentMode(availablePresentModes);

		// Create swap chain info
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

		// Find queue families
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		std::array<uint32_t, 2> queueFamilyIndicesLoc = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		// ����ͬ����Ķ��з���ͬһ��ͼ��ʱ����Ҫ��������Ȩת��
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = vk::SharingMode::eConcurrent;	// ����ģʽ (eConcurrent)
			createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndicesLoc.size());	// ��2����������
			createInfo.pQueueFamilyIndices = queueFamilyIndicesLoc.data();	// ����ļ��������б�
		}
		else {
			createInfo.imageSharingMode = vk::SharingMode::eExclusive;	// ��ռģʽ (eExclusive),����Ҫ��ʽ����Ȩת��
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		// Create swap chain
		swapChain = vk::raii::SwapchainKHR(device, createInfo);

		// Get swap chain images
		swapChainImages = swapChain.getImages();

		// Swapchain images start in UNDEFINED layout; track per-image layout for correct barriers.
		swapChainImageLayouts.assign(swapChainImages.size(), vk::ImageLayout::eUndefined);

		// Store swap chain format and extent
		swapChainImageFormat = surfaceFormat.format;

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create swap chain: " << e.what() << std::endl;
		return false;
	}
}
void Renderer::cleanupSwapChain() {
	swapChainImageViews.clear();

	depthData.textureImageView = vk::raii::ImageView(nullptr);
	depthData.textureImage = vk::raii::Image(nullptr);
	depthData.textureImageMemory = vk::raii::DeviceMemory(nullptr);

	swapChain = vk::raii::SwapchainKHR(nullptr);
}
void Renderer::recreateSwapChain() {

	int width = 0, height = 0;
	glfwGetFramebufferSize(platform->window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(platform->window, &width, &height);
		glfwWaitEvents();
	}

	device.waitIdle();

	cleanupSwapChain();
	createSwapChain();
	createImageViews();
	createDepthResources();
}
bool Renderer::createImageViews() {
	try {
		assert(swapChainImageViews.empty());

		// Create image view info template (image will be set per iteration)
		vk::ImageViewCreateInfo createInfo{
		  .viewType = vk::ImageViewType::e2D,
		  .format = swapChainImageFormat,
		  .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}
		};

		for (const auto& image : swapChainImages) {
			createInfo.image = image;
			swapChainImageViews.emplace_back(device, createInfo);
		}

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create image views: " << e.what() << std::endl;
		return false;
	}
}
bool Renderer::createCommandPool() {
	try {
		// Find queue families
		QueueFamilyIndices queueFamilyIndicesLoc = findQueueFamilies(physicalDevice);

		// Create command pool info
		vk::CommandPoolCreateInfo poolInfo{
		  .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		  .queueFamilyIndex = queueFamilyIndicesLoc.graphicsFamily.value()
		};

		// Create command pool
		commandPool = vk::raii::CommandPool(device, poolInfo);

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create command pool: " << e.what() << std::endl;
		return false;
	}
}
bool Renderer::createCommandBuffers() {
	try {
		// Resize command buffers vector
		commandBuffers.clear();
		commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

		// Create command buffer allocation info
		vk::CommandBufferAllocateInfo allocInfo{
		  .commandPool = *commandPool,
		  .level = vk::CommandBufferLevel::ePrimary,
		  .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
		};

		// Allocate command buffers
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create command buffers: " << e.what() << std::endl;
		return false;
	}
}
void Renderer::recordCommandBuffer(uint32_t imageIndex)
{
	auto& commandBuffer = commandBuffers[currentFrame];
	commandBuffer.begin({});
	// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
	transition_image_layout(
		swapChainImages[imageIndex],
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},                                                        // srcAccessMask (no need to wait for previous operations)
		vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,         // dstStage
		vk::ImageAspectFlagBits::eColor
	);
	transition_image_layout(
		depthData.textureImage,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);
	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::RenderingAttachmentInfo attachmentInfo = {
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor };
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = depthData.textureImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth };
	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

	auto& mesh = resourceManager->meshes[0];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	for (int i = 0; i < MAX_OBJECTS; ++i) {
		auto& descriptorSets = resourceManager->meshUniformBuffer[i].descriptorSets;
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
		commandBuffer.drawIndexed(mesh.indices.size(), 1, 0, 0, 0);
	}

	commandBuffer.endRendering();
	// After rendering, transition the swapchain image to PRESENT_SRC
	transition_image_layout(
		swapChainImages[imageIndex],
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,                // srcAccessMask
		{},                                                        // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		vk::PipelineStageFlagBits2::eBottomOfPipe,                  // dstStage
		vk::ImageAspectFlagBits::eColor
	);
	commandBuffer.end();
}
void Renderer::updateUniformBuffer(uint32_t currentImage) {
	float deltaTime = platform->frameTimer;

	glm::mat4 view = camera.GetViewMatrix();
	glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom),
		static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
		0.1f, 100.0f);
	proj[1][1] *= -1; // Flip Y for Vulkan

	for (auto& resource : resourceManager->meshUniformBuffer) {
		const float rotationSpeed = 0.5f;                          // Rotation speed in radians per second
		resource.rotation.y += rotationSpeed * deltaTime;        // Slow rotation around Y axis scaled by frame time
		glm::mat4 model = resource.getModelMatrix();

		UniformBufferObject ubo{
			.model = model,
			.view = view,
			.proj = proj
		};

		memcpy(resource.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}
}
bool Renderer::createDepthResources() {
	try {
		vk::Format depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthData);
		depthData.textureImageView = createImageView(depthData.textureImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create depth resources: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::transition_image_layout(
	vk::Image               image,
	vk::ImageLayout         old_layout,
	vk::ImageLayout         new_layout,
	vk::AccessFlags2        src_access_mask,
	vk::AccessFlags2        dst_access_mask,
	vk::PipelineStageFlags2 src_stage_mask,
	vk::PipelineStageFlags2 dst_stage_mask,
	vk::ImageAspectFlags    image_aspect_flags)
{
	vk::ImageMemoryBarrier2 barrier = {
		.srcStageMask = src_stage_mask,
		.srcAccessMask = src_access_mask,
		.dstStageMask = dst_stage_mask,
		.dstAccessMask = dst_access_mask,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			   .aspectMask = image_aspect_flags,
			   .baseMipLevel = 0,
			   .levelCount = 1,
			   .baseArrayLayer = 0,
			   .layerCount = 1} };
	vk::DependencyInfo dependency_info = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier };
	commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
}
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

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = vk::SharingMode::eConcurrent;	// ?????? (eConcurrent)
			createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndicesLoc.size());	// ??2??????????
			createInfo.pQueueFamilyIndices = queueFamilyIndicesLoc.data();	// ????????????????
		}
		else {
			createInfo.imageSharingMode = vk::SharingMode::eExclusive;	// ????? (eExclusive),????????????????
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
	depthImageLayout = vk::ImageLayout::eUndefined;

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
	transition_image_layout(
		swapChainImages[imageIndex],
		swapChainImageLayouts[imageIndex],
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eAllCommands,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor
	);
	swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

	transition_image_layout(
		depthData.textureImage,
		depthImageLayout,
		vk::ImageLayout::eDepthAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eAllCommands,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);
	depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
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
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

	// in recordCommandBuffer function
#if RENDERING_LEVEL == 1
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
	auto& mesh = resourceManager->meshes[0];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	for (int i = 0; i < MAX_OBJECTS; ++i) {
		auto& descriptorSets = resourceManager->meshUniformBuffer[i].descriptorSets;
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
		commandBuffer.drawIndexed(mesh.indices.size(), 1, 0, 0, 0);
	}
#elif RENDERING_LEVEL == 2
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *instancedPipeline);
	auto& mesh = resourceManager->meshes[0];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *instancedPipelineLayout, 0, *instancedBufferResources.descriptorSets[currentFrame], nullptr);
	commandBuffer.drawIndexed(mesh.indices.size(), MAX_OBJECTS, 0, 0, 0);
#elif RENDERING_LEVEL == 3
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);
	auto& mesh = resourceManager->meshes[0];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pbrPipelineLayout, 0, *pbrInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	commandBuffer.drawIndexed(mesh.indices.size(), MAX_OBJECTS, 0, 0, 0);
#elif RENDERING_LEVEL == 4
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *skyboxPipeline);
	commandBuffer.bindVertexBuffers(0, *skyboxTriangleMesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*skyboxTriangleMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(skyboxTriangleMesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *skyboxPipelineLayout, 0, *skyboxDescriptorSets[currentFrame], nullptr);
	commandBuffer.drawIndexed(static_cast<uint32_t>(skyboxTriangleMesh.indices.size()), 1, 0, 0, 0);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *iblPbrPipeline);
	auto& mesh = resourceManager->meshes[0];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *iblPbrPipelineLayout, 0, *pbrInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	commandBuffer.drawIndexed(static_cast<uint32_t>(mesh.indices.size()), MAX_OBJECTS, 0, 0, 0);
#elif RENDERING_LEVEL == 5
	commandBuffer.endRendering();

	transition_image_layout(
		shadowMapData.textureImage,
		shadowMapLayout,
		vk::ImageLayout::eDepthAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eAllCommands,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);
	shadowMapLayout = vk::ImageLayout::eDepthAttachmentOptimal;

	vk::ClearValue shadowClearDepth = vk::ClearDepthStencilValue(1.0f, 0);
	vk::RenderingAttachmentInfo shadowDepthAttachmentInfo = {
		.imageView = shadowMapData.textureImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = shadowClearDepth
	};
	vk::RenderingInfo shadowRenderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = shadowMapExtent},
		.layerCount = 1,
		.colorAttachmentCount = 0,
		.pColorAttachments = nullptr,
		.pDepthAttachment = &shadowDepthAttachmentInfo
	};

	commandBuffer.beginRendering(shadowRenderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(shadowMapExtent.width), static_cast<float>(shadowMapExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), shadowMapExtent));
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowDepthPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);

	{
		auto& cubeMesh = resourceManager->meshes[0];
		commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
		commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
		commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), cubeInstanceCount, 0, 0, 0);
	}
	{
		auto& sphereMesh = resourceManager->meshes[1];
		commandBuffer.bindVertexBuffers(0, *sphereMesh.vertexBuffer, { 0 });
		commandBuffer.bindIndexBuffer(*sphereMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(sphereMesh.indices)::value_type>::value);
		commandBuffer.drawIndexed(static_cast<uint32_t>(sphereMesh.indices.size()), sphereInstanceCount, 0, 0, cubeInstanceCount);
	}

	commandBuffer.endRendering();

	transition_image_layout(
		shadowMapData.textureImage,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eDepth
	);
	shadowMapLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowLitPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);

	{
		auto& cubeMesh = resourceManager->meshes[0];
		commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
		commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
		commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), cubeInstanceCount, 0, 0, 0);
	}
	{
		auto& sphereMesh = resourceManager->meshes[1];
		commandBuffer.bindVertexBuffers(0, *sphereMesh.vertexBuffer, { 0 });
		commandBuffer.bindIndexBuffer(*sphereMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(sphereMesh.indices)::value_type>::value);
		commandBuffer.drawIndexed(static_cast<uint32_t>(sphereMesh.indices.size()), sphereInstanceCount, 0, 0, cubeInstanceCount);
	}

	recordUI(commandBuffer);
#endif

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
	swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;
	commandBuffer.end();
}

#if RENDERING_LEVEL == 1
void Renderer::updateUniformBuffer(uint32_t currentImage) {
	float deltaTime = platform->frameTimer;

	glm::mat4 view = camera.GetViewMatrix();
	glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom),
		static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
		0.1f, 100.0f);
	proj[1][1] *= -1; // Flip Y for Vulkan

	for (size_t i = 0; i < resourceManager->transforms.size(); ++i) {
		auto& resource = resourceManager->meshUniformBuffer[i];
		auto& transform = resourceManager->transforms[i];

		const float rotationSpeed = 0.5f;
		transform.rotation.y += rotationSpeed * deltaTime;
		glm::mat4 model = transform.getModelMatrix();

		MVP ubo{
			.model = model,
			.view = view,
			.proj = proj
		};

		memcpy(resource.BuffersMapped[currentImage], &ubo, sizeof(ubo));
	}
}
#endif

bool Renderer::createDepthResources() {
	try {
		vk::Format depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthData);
		depthData.textureImageView = createImageView(depthData.textureImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
		depthImageLayout = vk::ImageLayout::eUndefined;

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

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
		vk::ImageUsageFlags swapchainUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
#if RENDERING_LEVEL == 6
		// TAAU resolve 会把 swapchain 结果拷贝到 history，需要 TransferSrc。
		swapchainUsage |= vk::ImageUsageFlagBits::eTransferSrc;
#endif

		vk::SwapchainCreateInfoKHR createInfo{
		  .surface = *surface,
		  .minImageCount = minImageCount,
		  .imageFormat = surfaceFormat.format,
		  .imageColorSpace = surfaceFormat.colorSpace,
		  .imageExtent = swapChainExtent,
		  .imageArrayLayers = 1,
		  .imageUsage = swapchainUsage,
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

#if RENDERING_LEVEL == 6
	taauInputColorData.textureImageView = vk::raii::ImageView(nullptr);
	taauInputColorData.textureImage = vk::raii::Image(nullptr);
	taauInputColorData.textureImageMemory = vk::raii::DeviceMemory(nullptr);
	taauInputLayout = vk::ImageLayout::eUndefined;
	for (int i = 0; i < 2; ++i) {
		taauHistoryColorData[i].textureImageView = vk::raii::ImageView(nullptr);
		taauHistoryColorData[i].textureImage = vk::raii::Image(nullptr);
		taauHistoryColorData[i].textureImageMemory = vk::raii::DeviceMemory(nullptr);
		taauHistoryLayouts[i] = vk::ImageLayout::eUndefined;
	}
	taauHistoryValid = false;
#endif
#if RENDERING_LEVEL == 7
	ssrColorData.textureImageView = vk::raii::ImageView(nullptr);
	ssrColorData.textureImage = vk::raii::Image(nullptr);
	ssrColorData.textureImageMemory = vk::raii::DeviceMemory(nullptr);
	ssrColorLayout = vk::ImageLayout::eUndefined;
	ssrNormalData.textureImageView = vk::raii::ImageView(nullptr);
	ssrNormalData.textureImage = vk::raii::Image(nullptr);
	ssrNormalData.textureImageMemory = vk::raii::DeviceMemory(nullptr);
	ssrNormalLayout = vk::ImageLayout::eUndefined;
#endif
#if RENDERING_LEVEL == 8
	cullingDepthTexture.textureImageView = vk::raii::ImageView(nullptr);
	cullingDepthTexture.textureImage = vk::raii::Image(nullptr);
	cullingDepthTexture.textureImageMemory = vk::raii::DeviceMemory(nullptr);
	cullingDepthLayout = vk::ImageLayout::eUndefined;
	cullingHiZMipViews.clear();
	cullingHiZTexture.textureImageView = vk::raii::ImageView(nullptr);
	cullingHiZTexture.textureImage = vk::raii::Image(nullptr);
	cullingHiZTexture.textureImageMemory = vk::raii::DeviceMemory(nullptr);
	cullingHiZLayout = vk::ImageLayout::eUndefined;
#endif

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
#if RENDERING_LEVEL == 6
	createTAAUResources();
	createTAAUDescriptorSets();
#endif
#if RENDERING_LEVEL == 7
	createSSRResources();
	createSSRDescriptorSets();
#endif
#if RENDERING_LEVEL == 8
	createCullingDepthResources();
	createCullingHiZResources();
	createCullingDescriptorSets();
	createCullingHiZDescriptorSets();
#endif
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

	if (depthImageLayout != vk::ImageLayout::eDepthAttachmentOptimal) {
		transition_image_layout(
			depthData.textureImage,
			depthImageLayout,
			vk::ImageLayout::eDepthAttachmentOptimal,
			{},
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::PipelineStageFlagBits2::eTopOfPipe,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::ImageAspectFlagBits::eDepth
		);
		depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
	}

	// in recordCommandBuffer function
#if RENDERING_LEVEL >= 1 && RENDERING_LEVEL <= 4
	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

#if RENDERING_LEVEL == 1
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
	auto& mesh = resourceManager->meshes[scene->cubeMeshIndex];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	const uint32_t instanceCount = scene ? scene->getMeshInstanceCount(MeshTag::Cube) : 0;
	for (uint32_t i = 0; i < instanceCount; ++i) {
		auto& descriptorSets = resourceManager->meshUniformBuffer[i].descriptorSets;
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
		commandBuffer.drawIndexed(mesh.indices.size(), 1, 0, 0, 0);
	}
#elif RENDERING_LEVEL == 2
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *instancedPipeline);
	auto& mesh = resourceManager->meshes[scene->cubeMeshIndex];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *instancedPipelineLayout, 0, *instancedBufferResources.descriptorSets[currentFrame], nullptr);
	const uint32_t instanceCount = scene ? scene->getMeshInstanceCount(MeshTag::Cube) : 0;
	commandBuffer.drawIndexed(mesh.indices.size(), instanceCount, 0, 0, 0);
#elif RENDERING_LEVEL == 3
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);
	auto& mesh = resourceManager->meshes[scene->sphereMeshIndex];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pbrPipelineLayout, 0, *pbrInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	const uint32_t instanceCount = scene ? scene->getMeshInstanceCount(MeshTag::Sphere) : 0;
	commandBuffer.drawIndexed(mesh.indices.size(), instanceCount, 0, 0, 0);
#elif RENDERING_LEVEL == 4
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *skyboxPipeline);
	commandBuffer.bindVertexBuffers(0, *skyboxTriangleMesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*skyboxTriangleMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(skyboxTriangleMesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *skyboxPipelineLayout, 0, *skyboxDescriptorSets[currentFrame], nullptr);
	commandBuffer.drawIndexed(static_cast<uint32_t>(skyboxTriangleMesh.indices.size()), 1, 0, 0, 0);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *iblPbrPipeline);
	auto& mesh = resourceManager->meshes[scene->sphereMeshIndex];
	commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
	commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *iblPbrPipelineLayout, 0, *pbrInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	const uint32_t instanceCount = scene ? scene->getMeshInstanceCount(MeshTag::Sphere) : 0;
	commandBuffer.drawIndexed(static_cast<uint32_t>(mesh.indices.size()), instanceCount, 0, 0, 0);
#endif
#elif RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6
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

	auto drawShadowCubes = [&]() {
		const uint32_t cubeCount = scene ? scene->getMeshInstanceCount(MeshTag::Cube) : 0;
		if (cubeCount == 0) {
			return;
		}
		auto& cubeMesh = resourceManager->meshes[scene->cubeMeshIndex];
		commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
		commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
		commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), cubeCount, 0, 0, 0);
	};

	commandBuffer.beginRendering(shadowRenderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(shadowMapExtent.width), static_cast<float>(shadowMapExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), shadowMapExtent));
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowDepthPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	drawShadowCubes();
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

#if RENDERING_LEVEL == 5
	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowLitPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	drawShadowCubes();
#else RENDERING_LEVEL == 6
	transition_image_layout(
		taauInputColorData.textureImage,
		taauInputLayout,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor
	);
	taauInputLayout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::ClearValue taauClearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	transition_image_layout(
		taauVelocityData.textureImage,
		taauVelocityLayout,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor
	);
	taauVelocityLayout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::RenderingAttachmentInfo taauAttachmentInfo = {
		.imageView = taauInputColorData.textureImageView,
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = taauClearColor
	};
	vk::RenderingAttachmentInfo taauVelocityAttachmentInfo = {
		.imageView = taauVelocityData.textureImageView,
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
	};
	transition_image_layout(
		taauDepthData.textureImage,
		taauDepthLayout,
		vk::ImageLayout::eDepthAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);
	taauDepthLayout = vk::ImageLayout::eDepthAttachmentOptimal;

	const vk::Extent2D taauExtent{
		std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * taauRenderScale)),
		std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * taauRenderScale))
	};
	vk::RenderingAttachmentInfo taauDepthAttachmentInfo = {
		.imageView = taauDepthData.textureImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = vk::ClearDepthStencilValue(1.0f, 0)
	};

	std::array<vk::RenderingAttachmentInfo, 2> taauColorAttachments = { taauAttachmentInfo, taauVelocityAttachmentInfo };
	vk::RenderingInfo taauRenderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = taauExtent},
		.layerCount = 1,
		.colorAttachmentCount = static_cast<uint32_t>(taauColorAttachments.size()),
		.pColorAttachments = taauColorAttachments.data(),
		.pDepthAttachment = &taauDepthAttachmentInfo
	};

	commandBuffer.beginRendering(taauRenderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(taauExtent.width), static_cast<float>(taauExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), taauExtent));
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowLitPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);
	drawShadowCubes();
	commandBuffer.endRendering();

	recordTAAU(commandBuffer, imageIndex);

	vk::RenderingAttachmentInfo uiAttachmentInfo{
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eLoad,
		.storeOp = vk::AttachmentStoreOp::eStore
	};
	vk::RenderingInfo uiRenderingInfo{
		.renderArea = {.offset = {0, 0}, .extent = swapChainExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &uiAttachmentInfo
	};
	commandBuffer.beginRendering(uiRenderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
#endif

	recordUI(commandBuffer);

#elif RENDERING_LEVEL == 7
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
		const uint32_t cubeCount = scene ? scene->getMeshInstanceCount(MeshTag::Cube) : 0;
		if (cubeCount > 0) {
			auto& cubeMesh = resourceManager->meshes[scene->cubeMeshIndex];
			commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
			commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
			commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), cubeCount, 0, 0, 0);
		}
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

	transition_image_layout(
		ssrNormalData.textureImage,
		ssrNormalLayout,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor
	);
	ssrNormalLayout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::RenderingAttachmentInfo normalAttachmentInfo = {
		.imageView = ssrNormalData.textureImageView,
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = vk::ClearColorValue(0.5f, 0.5f, 1.0f, 1.0f)
	};
	std::array<vk::RenderingAttachmentInfo, 2> shadowLitAttachments = { attachmentInfo, normalAttachmentInfo };
	vk::RenderingInfo shadowLitRenderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
		.layerCount = 1,
		.colorAttachmentCount = static_cast<uint32_t>(shadowLitAttachments.size()),
		.pColorAttachments = shadowLitAttachments.data(),
		.pDepthAttachment = &depthAttachmentInfo
	};

	commandBuffer.beginRendering(shadowLitRenderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowLitPipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shadowPipelineLayout, 0, *shadowInstanceBufferResources.descriptorSets[currentFrame], nullptr);

	{
		const uint32_t cubeCount = scene ? scene->getMeshInstanceCount(MeshTag::Cube) : 0;
		if (cubeCount > 0) {
			auto& cubeMesh = resourceManager->meshes[scene->cubeMeshIndex];
			commandBuffer.bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
			commandBuffer.bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
			commandBuffer.drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), cubeCount, 0, 0, 0);
		}
	}
	commandBuffer.endRendering();

	transition_image_layout(
		ssrNormalData.textureImage,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eColor
	);
	ssrNormalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
#elif RENDERING_LEVEL == 8
{
	// culling compute pass 产出的两个关键结果：
	// 1) indirect draw command（instanceCount 由 compute 写入）
	// 2) visible instance index buffer（VS 读取可见实例索引）
	// 在图形阶段消费前，这里补一条内存屏障确保可见性。
	vk::BufferMemoryBarrier indirectBarrier{
		.srcAccessMask = {},
		.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = *cullingIndirectBufferResources.Buffers[currentFrame],
		.offset = 0,
		.size = 20
	};
	vk::BufferMemoryBarrier visibleBarrier{
		.srcAccessMask = {},
		.dstAccessMask = vk::AccessFlagBits::eShaderRead,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = *cullingVisibleBufferResources.Buffers[currentFrame],
		.offset = 0,
		.size = sizeof(uint32_t) * maxInstances
	};
	commandBuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eTopOfPipe,
		vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eVertexShader,
		{}, {}, { indirectBarrier, visibleBarrier }, {}
	);
}
	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
	// 间接绘制：GPU 直接读取 culling 结果，不经 CPU 回传实例列表。
	recordCullingDrawCommands(commandBuffer);
	recordUI(commandBuffer);
#endif

#if RENDERING_LEVEL != 7
	commandBuffer.endRendering();
#endif

#if RENDERING_LEVEL == 7
	recordSSR(commandBuffer, imageIndex);

	vk::RenderingAttachmentInfo uiAttachmentInfo{
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eLoad,
		.storeOp = vk::AttachmentStoreOp::eStore
	};
	vk::RenderingInfo uiRenderingInfo{
		.renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &uiAttachmentInfo
	};

	commandBuffer.beginRendering(uiRenderingInfo);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
	recordUI(commandBuffer);
	commandBuffer.endRendering();
#endif
	// After rendering, transition the swapchain image to PRESENT_SRC
	transition_image_layout(
		swapChainImages[imageIndex],
		swapChainImageLayouts[imageIndex],
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,                
		{},                                                        
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        
		vk::PipelineStageFlagBits2::eBottomOfPipe,                 
		vk::ImageAspectFlagBits::eColor
	);
	swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;
	commandBuffer.end();
}

void Renderer::updateUniformBuffer(uint32_t currentImage) {
	float deltaTime = platform->frameTimer;

	glm::mat4 view = camera.GetViewMatrix();
	glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom),
		static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
		0.1f, 100.0f);
	proj[1][1] *= -1; // Flip Y for Vulkan

	if (scene == nullptr) {
		return;
	}

	std::vector<glm::mat4> models;
	scene->world.collectModels(MeshTag::Cube, models, resourceManager->meshUniformBuffer.size());

	for (size_t i = 0; i < models.size(); ++i) {
		auto& resource = resourceManager->meshUniformBuffer[i];

		const float rotationSpeed = 0.5f;
		glm::mat4 model = glm::rotate(models[i], rotationSpeed * deltaTime, glm::vec3(0.0f, 1.0f, 0.0f));

		MVP ubo{
			.model = model,
			.view = view,
			.proj = proj
		};

		memcpy(resource.BuffersMapped[currentImage], &ubo, sizeof(ubo));
	}
}

bool Renderer::createDepthResources() {
	try {
		vk::Format depthFormat = findDepthFormat();
		vk::ImageUsageFlags depthUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
#if RENDERING_LEVEL == 7
		depthUsage |= vk::ImageUsageFlagBits::eSampled;
#endif
		createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, vk::ImageTiling::eOptimal, depthUsage, vk::MemoryPropertyFlagBits::eDeviceLocal, depthData);
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

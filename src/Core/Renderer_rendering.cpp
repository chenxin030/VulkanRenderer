#include "Renderer.h"
bool Renderer::createImageViews() {
	try {
		swapChainImageViews.clear();
		swapChainImageViews.reserve(swapChainImages.size());

		// Create image view info template (image will be set per iteration)
		vk::ImageViewCreateInfo createInfo{
		  .viewType = vk::ImageViewType::e2D,
		  .format = swapChainImageFormat,
		  .components = {
			.r = vk::ComponentSwizzle::eIdentity,
			.g = vk::ComponentSwizzle::eIdentity,
			.b = vk::ComponentSwizzle::eIdentity,
			.a = vk::ComponentSwizzle::eIdentity
		  },
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
bool Renderer::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
    commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
    return true;
}
//bool Renderer::createCommandBuffers() {
//    try {
//        // Resize command buffers vector
//        commandBuffers.clear();
//        commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
//
//        // Create command buffer allocation info
//        vk::CommandBufferAllocateInfo allocInfo{
//          .commandPool = *commandPool,
//          .level = vk::CommandBufferLevel::ePrimary,
//          .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
//        };
//
//        // Allocate command buffers
//        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
//
//        return true;
//    }
//    catch (const std::exception& e) {
//        std::cerr << "Failed to create command buffers: " << e.what() << std::endl;
//        return false;
//    }
//}
void Renderer::recordCommandBuffer(uint32_t imageIndex)
{
	commandBuffer.begin({});
	// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
	transition_image_layout(
		imageIndex,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},                                                        // srcAccessMask (no need to wait for previous operations)
		vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		vk::PipelineStageFlagBits2::eColorAttachmentOutput         // dstStage
	);
	vk::ClearValue              clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::RenderingAttachmentInfo attachmentInfo = {
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor };
	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo };

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
	commandBuffer.draw(3, 1, 0, 0);
	commandBuffer.endRendering();
	// After rendering, transition the swapchain image to PRESENT_SRC
	transition_image_layout(
		imageIndex,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,                // srcAccessMask
		{},                                                        // dstAccessMask
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
	);
	commandBuffer.end();
}

void Renderer::transition_image_layout(
	uint32_t                imageIndex,
	vk::ImageLayout         old_layout,
	vk::ImageLayout         new_layout,
	vk::AccessFlags2        src_access_mask,
	vk::AccessFlags2        dst_access_mask,
	vk::PipelineStageFlags2 src_stage_mask,
	vk::PipelineStageFlags2 dst_stage_mask)
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
		.image = swapChainImages[imageIndex],
		.subresourceRange = {
			   .aspectMask = vk::ImageAspectFlagBits::eColor,
			   .baseMipLevel = 0,
			   .levelCount = 1,
			   .baseArrayLayer = 0,
			   .layerCount = 1} };
	vk::DependencyInfo dependency_info = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier };
	commandBuffer.pipelineBarrier2(dependency_info);
}
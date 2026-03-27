#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

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

#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <cmath>

#if RENDERING_LEVEL == 3

bool Renderer::createPBRDescriptorSetLayout() {
	try {
		std::vector<vk::DescriptorSetLayoutBinding> bindings = {
			{
				.binding = 0,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
			},
			{
				.binding = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
			},
			{
				.binding = 2,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.descriptorCount = 1,
				.stageFlags = vk::ShaderStageFlagBits::eFragment
			}
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings = bindings.data()
		};

		pbrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create PBR descriptor set layout: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createPBRDescriptorPool() {
	try {
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2 }, // Scene + Light
			{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
		};

		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
			.pPoolSizes = poolSizes.data()
		};

		pbrDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create PBR descriptor pool: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::createPBRDescriptorSets() {
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *pbrDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = *pbrDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};

	pbrInstanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DescriptorBufferInfo sceneBufferInfo{
			.buffer = *sceneUboResources.Buffers[i],
			.offset = 0,
			.range = sizeof(SceneUBO)
		};

		vk::DescriptorBufferInfo instanceBufferInfo{
			.buffer = *pbrInstanceBufferResources.Buffers[i],
			.offset = 0,
			.range = sizeof(PBRInstanceData) * maxInstances
		};

		vk::DescriptorBufferInfo lightBufferInfo{
			.buffer = *lightUboResources.Buffers[i],
			.offset = 0,
			.range = sizeof(LightUBO)
		};

		std::vector<vk::WriteDescriptorSet> descriptorWrites = {
			{
				.dstSet = *pbrInstanceBufferResources.descriptorSets[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo = &sceneBufferInfo
			},
			{
				.dstSet = *pbrInstanceBufferResources.descriptorSets[i],
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &instanceBufferInfo
			},
			{
				.dstSet = *pbrInstanceBufferResources.descriptorSets[i],
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo = &lightBufferInfo
			}
		};

		device.updateDescriptorSets(descriptorWrites, nullptr);
	}
}

bool Renderer::createPBRPipeline() {
	try {
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "pbr.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
		vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

		vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };

		vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = vk::False,
																   .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

		vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

		std::vector dynamicStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*pbrDescriptorSetLayout, .pushConstantRangeCount = 0 };

		pbrPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::Format depthFormat = findDepthFormat();

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = 2,
				.pStages = shaderStages,
				.pVertexInputState = &vertexInputInfo,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pDepthStencilState = &depthStencil,
				.pColorBlendState = &colorBlending,
				.pDynamicState = &dynamicState,
				.layout = pbrPipelineLayout,
				.renderPass = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &swapChainImageFormat,
				.depthAttachmentFormat = depthFormat
			}
		};

		pbrPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create PBR graphics pipeline: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::createPBRBuffers() {
	createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
	createUniformBuffers(lightUboResources, sizeof(LightUBO));
	if (scene != nullptr) {
		maxInstances = scene->getMaxInstances();
	}
	createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * maxInstances);
}

void Renderer::updatePBRInstanceBuffers(uint32_t currentImage) {
	// Scene UBO
	SceneUBO sceneUbo{
		.projection = glm::perspective(glm::radians(camera.Zoom),
			static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
			0.1f, 100.0f),
		.view = camera.GetViewMatrix(),
		.camPos = camera.Position
	};
	sceneUbo.projection[1][1] *= -1;
	memcpy(sceneUboResources.BuffersMapped[currentImage], &sceneUbo, sizeof(sceneUbo));

	if (scene == nullptr || scene->getActiveInstanceCount() == 0) {
		return;
	}

	std::vector<PBRInstance> ecsInstances;
	scene->world.collectPBRInstances(MeshTag::Sphere, ecsInstances, maxInstances);
	if (ecsInstances.empty()) {
		return;
	}

	std::vector<PBRInstanceData> instanceData;
	instanceData.reserve(ecsInstances.size());
	for (const auto& instance : ecsInstances) {
		instanceData.push_back(PBRInstanceData{ instance.model, instance.metallic, instance.roughness, instance.color });
	}

	memcpy(pbrInstanceBufferResources.BuffersMapped[currentImage], instanceData.data(), sizeof(PBRInstanceData) * instanceData.size());

	// Light Animation
	static auto startTime = std::chrono::high_resolution_clock::now();
	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	LightUBO lightUbo;
	lightUbo.lights[0] = { .position = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 400) };

	lightUbo.lights[1] = { .position = glm::vec4(-20.0f, -10.0f, 10.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 50.0f) };

	// Moving along X axis
	lightUbo.lights[2] = { .position = glm::vec4(sin(time * 0.5f) * 12.0f, 5.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 150.0f) };
	// Moving along Y axis
	lightUbo.lights[3] = { .position = glm::vec4(0.0f, cos(time * 0.5f) * 12.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 150.0f) };

	memcpy(lightUboResources.BuffersMapped[currentImage], &lightUbo, sizeof(lightUbo));
}

#elif RENDERING_LEVEL == 4

namespace
{
	struct PushConstMat4
	{
		glm::mat4 mvp;
	};

	struct PushConstIrradiance
	{
		glm::mat4 mvp;
		float deltaPhi;
		float deltaTheta;
		float padding0;
		float padding1;
	};

	struct PushConstPrefilter
	{
		glm::mat4 mvp;
		float roughness;
		uint32_t numSamples;
		float padding0;
		float padding1;
	};

	void transitionImageLayoutCmd(
		vk::raii::CommandBuffer& commandBuffer,
		vk::Image image,
		vk::ImageAspectFlags aspectMask,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		uint32_t baseMipLevel,
		uint32_t levelCount,
		uint32_t baseArrayLayer,
		uint32_t layerCount,
		vk::PipelineStageFlags srcStage,
		vk::PipelineStageFlags dstStage,
		vk::AccessFlags srcAccessMask,
		vk::AccessFlags dstAccessMask)
	{
		vk::ImageMemoryBarrier barrier{
			.srcAccessMask = srcAccessMask,
			.dstAccessMask = dstAccessMask,
			.oldLayout = oldLayout,
			.newLayout = newLayout,
			.srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			.dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			.image = image,
			.subresourceRange = vk::ImageSubresourceRange{
				.aspectMask = aspectMask,
				.baseMipLevel = baseMipLevel,
				.levelCount = levelCount,
				.baseArrayLayer = baseArrayLayer,
				.layerCount = layerCount
			}
		};
		commandBuffer.pipelineBarrier(srcStage, dstStage, {}, {}, {}, barrier);
	}
}

bool Renderer::createIBLPBRDescriptorSetLayout()
{
	try
	{
		std::vector<vk::DescriptorSetLayoutBinding> bindings = {
			{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },	// sceneUbo
			{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },	// instanceData
			{.binding = 2, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },										// lightUbo
			{.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },								// irradianceCubemapSampler
			{.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },								// prefilteredMapSampler
			{.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },								// BRDFLUTsampler
			{.binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },										// paramsUbo
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings = bindings.data()
		};

		iblPbrDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create IBL PBR descriptor set layout: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createIBLPBRDescriptorPool()
{
	try
	{
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
			{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
			{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
		};

		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
			.pPoolSizes = poolSizes.data()
		};

		iblPbrDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create IBL PBR descriptor pool: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::createIBLPBRDescriptorSets()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *iblPbrDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = *iblPbrDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};

	pbrInstanceBufferResources.descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo sceneBufferInfo{ .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
		vk::DescriptorBufferInfo instanceBufferInfo{ .buffer = *pbrInstanceBufferResources.Buffers[i], .offset = 0, .range = sizeof(PBRInstanceData) * maxInstances };
		vk::DescriptorBufferInfo lightBufferInfo{ .buffer = *lightUboResources.Buffers[i], .offset = 0, .range = sizeof(LightUBO) };
		vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *paramsUboResources.Buffers[i], .offset = 0, .range = sizeof(ParamsUBO) };

		vk::DescriptorImageInfo irradianceInfo{ .sampler = irradianceCubemapData.textureSampler, .imageView = irradianceCubemapData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::DescriptorImageInfo prefilteredInfo{ .sampler = prefilteredEnvMapData.textureSampler, .imageView = prefilteredEnvMapData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::DescriptorImageInfo brdfInfo{ .sampler = brdfLutData.textureSampler, .imageView = brdfLutData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

		std::vector<vk::WriteDescriptorSet> writes = {
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &instanceBufferInfo },
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &lightBufferInfo },
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &irradianceInfo },
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &prefilteredInfo },
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &brdfInfo },
			{.dstSet = *pbrInstanceBufferResources.descriptorSets[i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo },
		};
		device.updateDescriptorSets(writes, nullptr);
	}
}

bool Renderer::createIBLPBRPipeline()
{
	try
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "pbribl.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal))
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
		vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = vk::False,
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
		};
		vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

		std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*iblPbrDescriptorSetLayout, .pushConstantRangeCount = 0 };
		iblPbrPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::Format depthFormat = findDepthFormat();
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = 2,
				.pStages = shaderStages,
				.pVertexInputState = &vertexInputInfo,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pDepthStencilState = &depthStencil,
				.pColorBlendState = &colorBlending,
				.pDynamicState = &dynamicState,
				.layout = iblPbrPipelineLayout,
				.renderPass = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &swapChainImageFormat,
				.depthAttachmentFormat = depthFormat
			}
		};

		iblPbrPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create IBL PBR pipeline: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::createIBLPBRBuffers()
{
	createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
	createUniformBuffers(lightUboResources, sizeof(LightUBO));
	createUniformBuffers(paramsUboResources, sizeof(ParamsUBO));
	createUniformBuffers(skyboxUboResources, sizeof(SkyboxUBO));
	if (scene != nullptr) {
		maxInstances = scene->getMaxInstances();
	}
	createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * maxInstances);
}

void Renderer::updateIBLPBRBuffers(uint32_t currentImage)
{
	SceneUBO sceneUbo{
		.projection = glm::perspective(glm::radians(camera.Zoom),
			static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
			0.1f, 100.0f),
		.view = camera.GetViewMatrix(),
		.camPos = camera.Position
	};
	sceneUbo.projection[1][1] *= -1;
	memcpy(sceneUboResources.BuffersMapped[currentImage], &sceneUbo, sizeof(sceneUbo));

	if (scene == nullptr || scene->getActiveInstanceCount() == 0) {
		return;
	}

	std::vector<PBRInstance> ecsInstances;
	scene->world.collectPBRInstances(MeshTag::Sphere, ecsInstances, maxInstances);
	if (ecsInstances.empty()) {
		return;
	}

	std::vector<PBRInstanceData> instanceData;
	instanceData.reserve(ecsInstances.size());
	for (const auto& instance : ecsInstances) {
		instanceData.push_back(PBRInstanceData{ instance.model, instance.metallic, instance.roughness, instance.color });
	}
	memcpy(pbrInstanceBufferResources.BuffersMapped[currentImage], instanceData.data(), sizeof(PBRInstanceData) * instanceData.size());

	static auto startTime = std::chrono::high_resolution_clock::now();
	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	LightUBO lightUbo;
	lightUbo.lights[0] = { .position = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 20.0f) };
	lightUbo.lights[1] = { .position = glm::vec4(-20.0f, -10.0f, 10.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 10.0f) };
	lightUbo.lights[2] = { .position = glm::vec4(sin(time * 0.5f) * 12.0f, 5.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 15.0f) };
	lightUbo.lights[3] = { .position = glm::vec4(0.0f, cos(time * 0.5f) * 12.0f, 8.0f, 1.0f), .color = glm::vec4(1.0f, 1.0f, 1.0f, 15.0f) };
	memcpy(lightUboResources.BuffersMapped[currentImage], &lightUbo, sizeof(lightUbo));

	ParamsUBO params{ .exposure = 4.5f, .gamma = 2.2f };
	memcpy(paramsUboResources.BuffersMapped[currentImage], &params, sizeof(params));

	SkyboxUBO skyboxUbo{
		.invProjection = glm::inverse(sceneUbo.projection),
		.invView = glm::inverse(sceneUbo.view)
	};
	memcpy(skyboxUboResources.BuffersMapped[currentImage], &skyboxUbo, sizeof(skyboxUbo));
}

bool Renderer::createSkyboxDescriptorSetLayout()
{
	try
	{
		std::vector<vk::DescriptorSetLayoutBinding> bindings = {
			{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
			{.binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
			{.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
		};
		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(bindings.size()),
			.pBindings = bindings.data()
		};
		skyboxDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create skybox descriptor set layout: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createSkyboxDescriptorPool()
{
	try
	{
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u },
			{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
		};
		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
			.pPoolSizes = poolSizes.data()
		};
		skyboxDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create skybox descriptor pool: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::createSkyboxDescriptorSets()
{
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *skyboxDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = *skyboxDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};

	skyboxDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vk::DescriptorBufferInfo skyboxBufferInfo{ .buffer = *skyboxUboResources.Buffers[i], .offset = 0, .range = sizeof(SkyboxUBO) };
		vk::DescriptorBufferInfo paramsBufferInfo{ .buffer = *paramsUboResources.Buffers[i], .offset = 0, .range = sizeof(ParamsUBO) };
		vk::DescriptorImageInfo envInfo{ .sampler = prefilteredEnvMapData.textureSampler, .imageView = prefilteredEnvMapData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

		std::vector<vk::WriteDescriptorSet> writes = {
			{.dstSet = *skyboxDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &skyboxBufferInfo },
			{.dstSet = *skyboxDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsBufferInfo },
			{.dstSet = *skyboxDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &envInfo },
		};
		device.updateDescriptorSets(writes, nullptr);
	}
}

bool Renderer::createSkyboxPipeline()
{
	try
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "skybox.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();
		std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions = {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos))
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data()
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
		vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			.depthTestEnable = vk::False,
			.depthWriteEnable = vk::False,
			.depthCompareOp = vk::CompareOp::eAlways,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = vk::False,
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
		};
		vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

		std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*skyboxDescriptorSetLayout, .pushConstantRangeCount = 0 };
		skyboxPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::Format depthFormat = findDepthFormat();
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = 2,
				.pStages = shaderStages,
				.pVertexInputState = &vertexInputInfo,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pDepthStencilState = &depthStencil,
				.pColorBlendState = &colorBlending,
				.pDynamicState = &dynamicState,
				.layout = skyboxPipelineLayout,
				.renderPass = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &swapChainImageFormat,
				.depthAttachmentFormat = depthFormat
			}
		};

		skyboxPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create skybox pipeline: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::generateIBLResources()
{
	if (resourceManager->textures.empty())
	{
		throw std::runtime_error("HDR texture not loaded (resourceManager->textures is empty)");
	}

	const vk::Format envFormat = vk::Format::eR16G16B16A16Sfloat;
	const uint32_t envDim = 512u;
	const uint32_t irradianceDim = 64u;
	const uint32_t prefilterDim = 512u;
	const uint32_t prefilterMipLevels = static_cast<uint32_t>(std::floor(std::log2(prefilterDim))) + 1u;
	const uint32_t brdfDim = 512u;

	createImage(envDim, envDim, 1, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, envCubemapData);
	createImage(irradianceDim, irradianceDim, 1, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, irradianceCubemapData);
	createImage(prefilterDim, prefilterDim, prefilterMipLevels, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, prefilteredEnvMapData);
	createImage(brdfDim, brdfDim, 1, 1, {}, envFormat, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, brdfLutData);

	envCubemapData.mipLevels = 1;
	irradianceCubemapData.mipLevels = 1;
	prefilteredEnvMapData.mipLevels = prefilterMipLevels;
	brdfLutData.mipLevels = 1;

	envCubemapData.textureImageView = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
		.image = envCubemapData.textureImage,
		.viewType = vk::ImageViewType::eCube,
		.format = envFormat,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 }
		});
	irradianceCubemapData.textureImageView = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
		.image = irradianceCubemapData.textureImage,
		.viewType = vk::ImageViewType::eCube,
		.format = envFormat,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 }
		});
	prefilteredEnvMapData.textureImageView = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
		.image = prefilteredEnvMapData.textureImage,
		.viewType = vk::ImageViewType::eCube,
		.format = envFormat,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, prefilterMipLevels, 0, 6 }
		});
	brdfLutData.textureImageView = createImageView(brdfLutData.textureImage, envFormat, vk::ImageAspectFlagBits::eColor, 1);

	vk::SamplerCreateInfo envSamplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.mipLodBias = 0.0f,
		.anisotropyEnable = vk::False,
		.maxAnisotropy = 1.0f,
		.compareEnable = vk::False,			
		.compareOp = vk::CompareOp::eAlways,
		.minLod = 0.0f,
		.maxLod = static_cast<float>(prefilterMipLevels)
	};
	envCubemapData.textureSampler = vk::raii::Sampler(device, envSamplerInfo);
	irradianceCubemapData.textureSampler = vk::raii::Sampler(device, envSamplerInfo);
	prefilteredEnvMapData.textureSampler = vk::raii::Sampler(device, envSamplerInfo);
	brdfLutData.textureSampler = vk::raii::Sampler(device, vk::SamplerCreateInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.minLod = 0.0f,
		.maxLod = 0.0f
		});

	Mesh cubeMesh;
	generateCube(cubeMesh);
	createVertexBuffer(cubeMesh);
	createIndexBuffer(cubeMesh);

	auto cmd = beginSingleTimeCommands();

	transitionImageLayoutCmd(*cmd, envCubemapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		0, 1, 0, 6, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite);
	transitionImageLayoutCmd(*cmd, irradianceCubemapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		0, 1, 0, 6, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite);
	transitionImageLayoutCmd(*cmd, prefilteredEnvMapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		0, prefilterMipLevels, 0, 6, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite);
	transitionImageLayoutCmd(*cmd, brdfLutData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		0, 1, 0, 1, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite);

	vk::raii::ShaderModule filterModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "filtercube.spv"));
	vk::raii::ShaderModule irradianceModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "irradiancecube.spv"));
	vk::raii::ShaderModule prefilterModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "prefilterenvmap.spv"));
	vk::raii::ShaderModule brdfModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "genbrdflut.spv"));

	auto bindingDescription = Vertex::getBindingDescription();
	std::array<vk::VertexInputAttributeDescription, 1> posOnlyAttributeDescriptions = {
		vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos))
	};
	vk::PipelineVertexInputStateCreateInfo posOnlyVertexInputInfo{
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(posOnlyAttributeDescriptions.size()),
		.pVertexAttributeDescriptions = posOnlyAttributeDescriptions.data()
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
	vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
	vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };
	vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };
	vk::PipelineDepthStencilStateCreateInfo depthStencil{ .depthTestEnable = vk::False, .depthWriteEnable = vk::False };
	vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = vk::False, .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
	vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };
	std::vector dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
	vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

	vk::DescriptorSetLayoutBinding equirectBinding{ .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment };
	vk::raii::DescriptorSetLayout equirectSetLayout(device, vk::DescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &equirectBinding });
	vk::PushConstantRange equirectPushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstMat4) };
	vk::raii::PipelineLayout equirectPipelineLayout(device, vk::PipelineLayoutCreateInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*equirectSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &equirectPushConstantRange
		});

	vk::DescriptorPoolSize equirectPoolSize{ .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1 };
	vk::raii::DescriptorPool equirectPool(device, vk::DescriptorPoolCreateInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &equirectPoolSize
		});
	vk::raii::DescriptorSet equirectSet = std::move(device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ .descriptorPool = *equirectPool, .descriptorSetCount = 1, .pSetLayouts = &*equirectSetLayout }).front());
	vk::DescriptorImageInfo hdrInfo{ .sampler = resourceManager->textures[0].textureSampler, .imageView = resourceManager->textures[0].textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
	device.updateDescriptorSets(vk::WriteDescriptorSet{ .dstSet = *equirectSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &hdrInfo }, nullptr);

	std::array<vk::PipelineShaderStageCreateInfo, 2> equirectStages = {
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = filterModule, .pName = "vertMain" },
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = filterModule, .pName = "fragMain" }
	};
	vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> equirectPipelineInfo = {
		{
			.stageCount = 2,
			.pStages = equirectStages.data(),
			.pVertexInputState = &posOnlyVertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = equirectPipelineLayout
		},
		{.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat }
	};
	vk::raii::Pipeline equirectPipeline(device, nullptr, equirectPipelineInfo.get<vk::GraphicsPipelineCreateInfo>());

	vk::DescriptorSetLayoutBinding cubeBinding{ .binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment };
	vk::raii::DescriptorSetLayout cubeSetLayout(device, vk::DescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &cubeBinding });

	vk::PushConstantRange irradiancePushConstantRange{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstIrradiance) };
	vk::raii::PipelineLayout irradiancePipelineLayout(device, vk::PipelineLayoutCreateInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*cubeSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &irradiancePushConstantRange
		});
	vk::PushConstantRange prefilterPushConstantRange{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstPrefilter) };
	vk::raii::PipelineLayout prefilterPipelineLayout(device, vk::PipelineLayoutCreateInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*cubeSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &prefilterPushConstantRange
		});

	vk::DescriptorPoolSize cubePoolSize{ .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 2 };
	vk::raii::DescriptorPool cubePool(device, vk::DescriptorPoolCreateInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 2,
		.poolSizeCount = 1,
		.pPoolSizes = &cubePoolSize
		});
	std::array<vk::DescriptorSetLayout, 2> cubeLayouts = { *cubeSetLayout, *cubeSetLayout };
	vk::raii::DescriptorSets cubeSets(device, vk::DescriptorSetAllocateInfo{ .descriptorPool = *cubePool, .descriptorSetCount = 2, .pSetLayouts = cubeLayouts.data() });
	vk::DescriptorImageInfo envCubeInfo{ .sampler = envCubemapData.textureSampler, .imageView = envCubemapData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
	vk::DescriptorImageInfo envCubeInfo2{ .sampler = envCubemapData.textureSampler, .imageView = envCubemapData.textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
	device.updateDescriptorSets(vk::WriteDescriptorSet{ .dstSet = *cubeSets[0], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &envCubeInfo }, nullptr);
	device.updateDescriptorSets(vk::WriteDescriptorSet{ .dstSet = *cubeSets[1], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &envCubeInfo2 }, nullptr);

	std::array<vk::PipelineShaderStageCreateInfo, 2> irradianceStages = {
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = irradianceModule, .pName = "vertMain" },
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = irradianceModule, .pName = "fragMain" }
	};
	vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> irradiancePipelineInfo = {
		{
			.stageCount = 2,
			.pStages = irradianceStages.data(),
			.pVertexInputState = &posOnlyVertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = irradiancePipelineLayout
		},
		{.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat }
	};
	vk::raii::Pipeline irradiancePipeline(device, nullptr, irradiancePipelineInfo.get<vk::GraphicsPipelineCreateInfo>());

	std::array<vk::PipelineShaderStageCreateInfo, 2> prefilterStages = {
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = prefilterModule, .pName = "vertMain" },
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = prefilterModule, .pName = "fragMain" }
	};
	vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> prefilterPipelineInfo = {
		{
			.stageCount = 2,
			.pStages = prefilterStages.data(),
			.pVertexInputState = &posOnlyVertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = prefilterPipelineLayout
		},
		{.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat }
	};
	vk::raii::Pipeline prefilterPipeline(device, nullptr, prefilterPipelineInfo.get<vk::GraphicsPipelineCreateInfo>());

	vk::raii::PipelineLayout brdfPipelineLayout(device, vk::PipelineLayoutCreateInfo{});
	vk::PipelineVertexInputStateCreateInfo emptyVertexInput{};
	std::array<vk::PipelineShaderStageCreateInfo, 2> brdfStages = {
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = brdfModule, .pName = "vertMain" },
		vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = brdfModule, .pName = "fragMain" }
	};
	vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> brdfPipelineInfo = {
		{
			.stageCount = 2,
			.pStages = brdfStages.data(),
			.pVertexInputState = &emptyVertexInput,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = brdfPipelineLayout
		},
		{.colorAttachmentCount = 1, .pColorAttachmentFormats = &envFormat }
	};

	vk::raii::Pipeline brdfPipeline(device, nullptr, brdfPipelineInfo.get<vk::GraphicsPipelineCreateInfo>());

	glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
	captureProjection[1][1] *= -1;
	std::array<glm::mat4, 6> captureViews = {
		glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	};

	for (uint32_t face = 0; face < 6; face++)
	{
		vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
			.image = envCubemapData.textureImage,
			.viewType = vk::ImageViewType::e2D,
			.format = envFormat,
			.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, face, 1 }
			});

		vk::RenderingAttachmentInfo colorAttachment{
			.imageView = *faceView,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
		};
		vk::RenderingInfo renderingInfo{
			.renderArea = { {0, 0}, {envDim, envDim} },
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachment
		};

		cmd->beginRendering(renderingInfo);
		cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(envDim), static_cast<float>(envDim), 0.0f, 1.0f));
		cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { envDim, envDim }));
		cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *equirectPipeline);
		cmd->bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
		cmd->bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
		cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *equirectPipelineLayout, 0, *equirectSet, {});

		PushConstMat4 pc{ .mvp = captureProjection * captureViews[face] };
		cmd->pushConstants(*equirectPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, vk::ArrayProxy<const PushConstMat4>(1, &pc));
		cmd->drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), 1, 0, 0, 0);
		cmd->endRendering();
	}

	transitionImageLayoutCmd(*cmd, envCubemapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		0, 1, 0, 6, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

	for (uint32_t face = 0; face < 6; face++)
	{
		vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
			.image = irradianceCubemapData.textureImage,
			.viewType = vk::ImageViewType::e2D,
			.format = envFormat,
			.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, face, 1 }
			});

		vk::RenderingAttachmentInfo colorAttachment{
			.imageView = *faceView,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
		};
		vk::RenderingInfo renderingInfo{
			.renderArea = { {0, 0}, {irradianceDim, irradianceDim} },
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachment
		};

		cmd->beginRendering(renderingInfo);
		cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(irradianceDim), static_cast<float>(irradianceDim), 0.0f, 1.0f));
		cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { irradianceDim, irradianceDim }));
		cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *irradiancePipeline);
		cmd->bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
		cmd->bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
		cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *irradiancePipelineLayout, 0, *cubeSets[0], {});

		PushConstIrradiance pc{
			.mvp = captureProjection * captureViews[face],
			.deltaPhi = 0.025f,
			.deltaTheta = 0.025f
		};
		cmd->pushConstants(*irradiancePipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, vk::ArrayProxy<const PushConstIrradiance>(1, &pc));
		cmd->drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), 1, 0, 0, 0);
		cmd->endRendering();
	}

	transitionImageLayoutCmd(*cmd, irradianceCubemapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		0, 1, 0, 6, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

	for (uint32_t mip = 0; mip < prefilterMipLevels; mip++)
	{
		uint32_t mipWidth = std::max(1u, prefilterDim >> mip);
		uint32_t mipHeight = std::max(1u, prefilterDim >> mip);
		float roughness = static_cast<float>(mip) / static_cast<float>(prefilterMipLevels - 1u);

		for (uint32_t face = 0; face < 6; face++)
		{
			vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
				.image = prefilteredEnvMapData.textureImage,
				.viewType = vk::ImageViewType::e2D,
				.format = envFormat,
				.subresourceRange = { vk::ImageAspectFlagBits::eColor, mip, 1, face, 1 }
				});

			vk::RenderingAttachmentInfo colorAttachment{
				.imageView = *faceView,
				.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.loadOp = vk::AttachmentLoadOp::eClear,
				.storeOp = vk::AttachmentStoreOp::eStore,
				.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
			};
			vk::RenderingInfo renderingInfo{
				.renderArea = { {0, 0}, {mipWidth, mipHeight} },
				.layerCount = 1,
				.colorAttachmentCount = 1,
				.pColorAttachments = &colorAttachment
			};

			cmd->beginRendering(renderingInfo);
			cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(mipWidth), static_cast<float>(mipHeight), 0.0f, 1.0f));
			cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { mipWidth, mipHeight }));
			cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *prefilterPipeline);
			cmd->bindVertexBuffers(0, *cubeMesh.vertexBuffer, { 0 });
			cmd->bindIndexBuffer(*cubeMesh.indexBuffer, 0, vk::IndexTypeValue<decltype(cubeMesh.indices)::value_type>::value);
			cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *prefilterPipelineLayout, 0, *cubeSets[1], {});

			PushConstPrefilter pc{
				.mvp = captureProjection * captureViews[face],
				.roughness = roughness,
				.numSamples = 64u
			};
			cmd->pushConstants(*prefilterPipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, vk::ArrayProxy<const PushConstPrefilter>(1, &pc));
			cmd->drawIndexed(static_cast<uint32_t>(cubeMesh.indices.size()), 1, 0, 0, 0);
			cmd->endRendering();
		}
	}

	transitionImageLayoutCmd(*cmd, prefilteredEnvMapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		0, prefilterMipLevels, 0, 6, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

	{
		vk::RenderingAttachmentInfo colorAttachment{
			.imageView = brdfLutData.textureImageView,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
		};
		vk::RenderingInfo renderingInfo{
			.renderArea = { {0, 0}, {brdfDim, brdfDim} },
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachment
		};

		cmd->beginRendering(renderingInfo);
		cmd->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(brdfDim), static_cast<float>(brdfDim), 0.0f, 1.0f));
		cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { brdfDim, brdfDim }));
		cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, *brdfPipeline);
		cmd->draw(3, 1, 0, 0);
		cmd->endRendering();
	}

	transitionImageLayoutCmd(*cmd, brdfLutData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		0, 1, 0, 1, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);

	endSingleTimeCommands(*cmd);
}

#endif

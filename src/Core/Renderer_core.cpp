#include <Renderer.h>
#include <set>
#include <map>

bool Renderer::createGraphicsPipeline() {
	try
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "shader.spv"));

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
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = vk::False };
		vk::PipelineViewportStateCreateInfo      viewportState{ .viewportCount = 1, .scissorCount = 1 };

		vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .lineWidth = 1.0f };

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
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

		vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

		std::vector dynamicStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor 
		};
		vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0 };

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
				.layout = pipelineLayout,
				.renderPass = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &swapChainImageFormat,
				.depthAttachmentFormat = depthFormat
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

bool createDescriptorPool() {
	try
	{
		uint32_t uniformBufferCount = resourceManager->meshUniformBuffer.size();
		std::array poolSize{
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * uniformBufferCount),
			vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * uniformBufferCount)
		};
		vk::DescriptorPoolCreateInfo poolInfo{
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT * uniformBufferCount,
			.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
			.pPoolSizes = poolSize.data()
		};
		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create descriptor pool: " << e.what() << std::endl;
		return false;
	}
}

void createDescriptorSets() {
	for (auto& resource : resourceManager->meshUniformBuffer) {
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data()
		};

		try {
			resource.descriptorSets.clear();
			resource.descriptorSets = device.allocateDescriptorSets(allocInfo);
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to allocate descriptor sets: " << e.what() << std::endl;
			throw;
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo uniformBufferInfo{ .buffer = resource.Buffers[i], .offset = 0, .range = sizeof(MVP) };
			vk::DescriptorImageInfo imageInfo{
				.sampler = resourceManager->textures[0].textureSampler,
				.imageView = resourceManager->textures[0].textureImageView,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = resource.descriptorSets[i],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pBufferInfo = &uniformBufferInfo
				},
				vk::WriteDescriptorSet{
					.dstSet = resource.descriptorSets[i],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo = &imageInfo
				}
			};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}
}
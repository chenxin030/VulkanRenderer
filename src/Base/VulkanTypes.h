#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

struct MeshBuffer {
	std::vector<vk::raii::Buffer> Buffers;
	std::vector<vk::raii::DeviceMemory> BuffersMemory;
	std::vector<void*> BuffersMapped;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	void clear()
	{
		for (size_t i = 0; i < BuffersMapped.size(); ++i)
		{
			if (BuffersMapped[i] != nullptr)
			{
				BuffersMemory[i].unmapMemory();
				BuffersMapped[i] = nullptr;
			}
		}
		descriptorSets.clear();
		BuffersMapped.clear();
		BuffersMemory.clear();
		Buffers.clear();
	}
};

struct TextureData {
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView = nullptr;
	vk::raii::Sampler      textureSampler = nullptr;
	uint32_t mipLevels;
};

struct ShadowUBO {
	glm::mat4 lightViewProj;
	glm::mat4 prevViewProj;
	glm::vec4 dirLightDirIntensity;
	glm::vec4 dirLightColor;
	glm::vec4 pointLightPosIntensity;
	glm::vec4 pointLightColor;
	glm::vec4 areaLightPosIntensity;
	glm::vec4 areaLightColor;
	glm::vec4 areaLightU;
	glm::vec4 areaLightV;
};

struct ShadowParamsUBO {
	int shadowFilterMode;
	float pcfRadiusTexels;
	float pcssLightSizeTexels;
	float shadowBiasMin;

	glm::vec2 invShadowMapSize;
	glm::vec2 padding0;
};

struct CullingParamsUBO {
	glm::vec4 frustumPlanes[6];
	glm::vec4 aabbMin;
	glm::vec4 aabbMax;
	glm::vec4 hiZInfo;
	uint32_t totalInstances;
	uint32_t useCulling;
	uint32_t padding0;
	uint32_t padding1;
};

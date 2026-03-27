#pragma once

#include "Mesh.h"
#include "RenderConfig.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct MeshBuffer {
	std::vector<vk::raii::Buffer> Buffers;
	std::vector<vk::raii::DeviceMemory> BuffersMemory;
	std::vector<void*> BuffersMapped;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
};

struct TextureData {
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView = nullptr;
	vk::raii::Sampler      textureSampler = nullptr;
	uint32_t mipLevels;
};

struct ParamsUBO {
	float exposure;
	float gamma;
};

struct SkyboxUBO {
	glm::mat4 invProjection;
	glm::mat4 invView;
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
	glm::vec4 hiZInfo; // x: width, y: height, z: mipCount, w: depthBias
	uint32_t totalInstances;
	uint32_t useCulling;
	uint32_t padding0;
	uint32_t padding1;
};

struct ResourceManager {
	std::vector<Mesh> meshes;
	std::vector<TextureData> textures;
	std::vector<MeshBuffer> meshUniformBuffer;

	std::vector<std::string> modelPath{
		"viking_room.glb"
	};
	std::vector<std::string> texPath{
		"viking_room.png"
		"newport_loft.hdr"
	};

};

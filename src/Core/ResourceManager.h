#pragma once

#include "Mesh.h"
#include "RenderConfig.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct MVP {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

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

struct PBRInstanceData {
	glm::mat4 model;
	float metallic;
	float roughness;
	alignas(16) glm::vec3 color;
};

struct PointLight {
	glm::vec4 position; // w is intensity or unused
	glm::vec4 color;    // w is intensity
};

struct LightUBO {
	PointLight lights[4];
};

struct SceneUBO {
	glm::mat4 projection;
	glm::mat4 view;
	glm::vec3 camPos;
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

struct ResourceManager {
	std::vector<Mesh> meshes;
	std::vector<TextureData> textures;
	std::vector<MeshBuffer> meshUniformBuffer;

	std::vector<std::string> modelPath{
#if RENDERING_LEVEL < 3
		"viking_room.glb"
#endif
	};
	std::vector<std::string> texPath{
#if RENDERING_LEVEL < 3
		"viking_room.png"
#elif RENDERING_LEVEL == 4
		"newport_loft.hdr"
#endif
	};

	void initResource(unsigned int modelCount) {
#if RENDERING_LEVEL == 3
		meshes.resize(1);
		meshUniformBuffer.resize(modelCount);
#elif RENDERING_LEVEL == 4
		meshes.resize(1);
		textures.resize(texPath.size());
		meshUniformBuffer.resize(modelCount);
#elif RENDERING_LEVEL == 5
		meshes.resize(2);
		meshUniformBuffer.resize(modelCount);
#elif RENDERING_LEVEL == 6
		meshes.resize(2);
		meshUniformBuffer.resize(modelCount);
#elif RENDERING_LEVEL == 7
		meshes.resize(2);
		meshUniformBuffer.resize(modelCount);
#endif
#if RENDERING_LEVEL == 1 || RENDERING_LEVEL == 2
		meshes.resize(modelPath.size());
		textures.resize(texPath.size());
		meshUniformBuffer.resize(modelCount);
#endif
	}
};

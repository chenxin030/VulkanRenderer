#pragma once

#include "Mesh.h"
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct MVP {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct Transform {
	glm::vec3 position = { 0.0f, 0.0f, 0.0f };
	glm::vec3 rotation = { 0.0f, 0.0f, 0.0f };
	glm::vec3 scale = { 1.0f, 1.0f, 1.0f };

	glm::mat4 getModelMatrix() const {
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, position);
		model = glm::rotate(model, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
		model = glm::scale(model, scale);
		return model;
	}
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
	std::vector<Transform> transforms;

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
		transforms.resize(modelCount);
#elif RENDERING_LEVEL == 4
		meshes.resize(1);
		textures.resize(texPath.size());
		meshUniformBuffer.resize(modelCount);
		transforms.resize(modelCount);
#elif RENDERING_LEVEL == 5
		meshes.resize(2);
		meshUniformBuffer.resize(modelCount);
		transforms.resize(modelCount);

		if (modelCount > 0) {
			transforms[0].position = { 0.0f, -1.0f, 0.0f };
			transforms[0].rotation = { 0.0f, 0.0f, 0.0f };
			transforms[0].scale = { 6.0f, 0.05f, 6.0f };
		}
		if (modelCount > 1) {
			transforms[1].position = { -2.0f, 0.25f, 0.0f };
			transforms[1].scale = { 0.25f, 1.20f, 0.25f };
		}
		if (modelCount > 2) {
			transforms[2].position = { -2.0f, 1.95f, 0.0f };
			transforms[2].scale = { 0.50f, 0.50f, 0.50f };
		}
		if (modelCount > 3) {
			transforms[3].position = { 0.0f, -0.92f, -1.5f };
			transforms[3].scale = { 0.30f, 0.03f, 0.30f };
		}
		if (modelCount > 4) {
			transforms[4].position = { 0.0f, -0.43f, -1.5f };
			transforms[4].scale = { 0.50f, 0.50f, 0.50f };
		}
		if (modelCount > 5) {
			transforms[5].position = { 2.0f, -0.45f, 0.8f };
			transforms[5].scale = { 0.50f, 0.50f, 0.50f };
		}
#elif RENDERING_LEVEL == 6
		meshes.resize(2);
		meshUniformBuffer.resize(modelCount);
		transforms.resize(modelCount);

		// TAAU test scene (ghosting / thin-line shimmer / fast motion / edge & high-frequency texture)
		if (modelCount > 0) {
			// Ground plane to expose edge shimmer
			transforms[0].position = { 0.0f, -1.1f, 0.0f };
			transforms[0].rotation = { 0.0f, 0.0f, 0.0f };
			transforms[0].scale = { 7.5f, 0.03f, 7.5f };
		}
		if (modelCount > 1) {
			// Thin pillar group for thin-line flicker
			transforms[1].position = { -3.0f, -0.2f, 0.0f };
			transforms[1].scale = { 0.08f, 2.8f, 0.08f };
		}
		if (modelCount > 2) {
			transforms[2].position = { -2.6f, 0.2f, 0.3f };
			transforms[2].scale = { 0.10f, 2.2f, 0.10f };
		}
		if (modelCount > 3) {
			// Fast moving probe (to be animated in Renderer_TAAU)
			transforms[3].position = { 2.5f, 0.35f, -1.6f };
			transforms[3].scale = { 0.35f, 0.35f, 0.35f };
		}
		if (modelCount > 4) {
			// Edge clipping probe near screen edge
			transforms[4].position = { 3.7f, -0.35f, -2.8f };
			transforms[4].scale = { 0.22f, 1.6f, 0.22f };
		}
		if (modelCount > 5) {
			// High-frequency alternating bars (dense array)
			transforms[5].position = { 0.0f, -0.45f, 2.8f };
			transforms[5].scale = { 0.12f, 0.9f, 0.12f };
		}
		if (modelCount > 6) {
			transforms[6].position = { 0.25f, -0.45f, 2.8f };
			transforms[6].scale = { 0.10f, 0.9f, 0.10f };
		}
		if (modelCount > 7) {
			transforms[7].position = { -0.25f, -0.45f, 2.8f };
			transforms[7].scale = { 0.14f, 0.9f, 0.14f };
		}
		if (modelCount > 8) {
			transforms[8].position = { 0.55f, -0.45f, 2.8f };
			transforms[8].scale = { 0.08f, 0.9f, 0.08f };
		}
#endif
#if RENDERING_LEVEL == 1 || RENDERING_LEVEL == 2
		meshes.resize(modelPath.size());
		textures.resize(texPath.size());
		meshUniformBuffer.resize(modelCount);
		transforms.resize(modelCount);

		// Object 1 - Center
		if (modelCount > 0) {
			transforms[0].position = { 0.0f, 0.0f, 0.0f };
			transforms[0].rotation = { 0.0f, glm::radians(-90.0f), 0.0f };
			transforms[0].scale = { 1.0f, 1.0f, 1.0f };
		}

		// Object 2 - Left
		if (modelCount > 1) {
			transforms[1].position = { -2.0f, 0.0f, -1.0f };
			transforms[1].rotation = { 0.0f, glm::radians(45.0f), .0f };
			transforms[1].scale = { 0.75f, 0.75f, 0.75f };
		}

		// Object 3 - Right
		if (modelCount > 2) {
			transforms[2].position = { 2.0f, 0.0f, -1.0f };
			transforms[2].rotation = { 0.0f, glm::radians(-45.0f), 0.0f };
			transforms[2].scale = { 0.75f, 0.75f, 0.75f };
		}
#endif
	}
};

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

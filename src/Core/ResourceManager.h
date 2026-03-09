#pragma once

#include "Mesh.h"
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct UniformBufferObject {
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

struct MeshUniformBuffer {
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	vk::raii::DescriptorPool descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
};

struct TextureData {
	vk::raii::Image textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView = nullptr;
	vk::raii::Sampler      textureSampler = nullptr;
	uint32_t mipLevels;
};

struct ResourceManager {
	std::vector<Mesh> meshes;
	std::vector<TextureData> textures;
	std::vector<MeshUniformBuffer> meshUniformBuffer;
	std::vector<Transform> transforms;

	std::vector<std::string> modelPath{
		"viking_room.glb"
	};
	std::vector<std::string> texPath{
		"viking_room.png"
	};

	void initResource(unsigned int modelCount) {
		meshes.resize(modelPath.size());
		textures.resize(texPath.size());
		meshUniformBuffer.resize(modelCount);
		transforms.resize(modelCount);

		// Object 1 - Center
		transforms[0].position = { 0.0f, 0.0f, 0.0f };
		transforms[0].rotation = { 0.0f, glm::radians(-90.0f), 0.0f };
		transforms[0].scale = { 1.0f, 1.0f, 1.0f };

		// Object 2 - Left
		transforms[1].position = { -2.0f, 0.0f, -1.0f };
		transforms[1].rotation = { 0.0f, glm::radians(45.0f), .0f };
		transforms[1].scale = { 0.75f, 0.75f, 0.75f };

		// Object 3 - Right
		transforms[2].position = { 2.0f, 0.0f, -1.0f };
		transforms[2].rotation = { 0.0f, glm::radians(-45.0f), 0.0f };
		transforms[2].scale = { 0.75f, 0.75f, 0.75f };
	}
};
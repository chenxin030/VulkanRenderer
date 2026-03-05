#pragma once

#include "Mesh.h"
#include "Texture.h"
#include <unordered_map>

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct MeshResource {
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
};

struct ResourceManager {
	std::vector<Mesh> meshes;
	std::vector<MeshResource> meshResource;
	std::unordered_map<std::string, TextureData> textures;
	uint16_t texCount = 0;

	std::vector<std::string> modelPath{
		"viking_room.glb"
	};
	std::vector<std::string> texPath{
		"viking_room.png"
	};

	void initResource(unsigned int modelCount) {
		meshes.resize(modelPath.size());
		meshResource.resize(modelCount);
		texCount = texPath.size();
	}
};
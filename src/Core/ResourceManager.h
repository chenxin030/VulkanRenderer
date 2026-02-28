#pragma once

#include "Mesh.h"

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct EntityResource {
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
};

struct EntityManager {
	std::vector<Mesh> meshes;
	std::vector<EntityResource> entityResource;
};

struct ResourceManager {
	EntityManager entityManager;
	// models
	// textures

	void initResources() {
		auto& meshes = entityManager.meshes;
		auto& entityResource = entityManager.entityResource;
		meshes.resize(1);
		entityResource.resize(1);

		meshes[0].vertices.resize(4);
		meshes[0].vertices = {
			{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
			{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
		};
		meshes[0].indices.resize(6);
		meshes[0].indices = {0, 1, 2, 2, 3, 0 };
	}
};
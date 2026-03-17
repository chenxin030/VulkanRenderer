#pragma once

#include "RenderConfig.h"
#include "ResourceManager.h"
#include <unordered_map>
#include <vector>

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

enum class MeshTag {
	Cube,
	Sphere
};

struct MeshTagHash {
	size_t operator()(MeshTag tag) const noexcept {
		return static_cast<size_t>(tag);
	}
};

struct InstanceColor {
	glm::vec4 color = glm::vec4(1.0f);
};

struct PBRMaterial {
	float metallic = 0.5f;
	float roughness = 0.5f;
	glm::vec3 color = glm::vec3(1.0f);
};

using Entity = uint32_t;

struct RenderInstance {
	glm::mat4 model;
	glm::vec4 color;
};

struct PBRInstance {
	glm::mat4 model;
	float metallic;
	float roughness;
	glm::vec3 color;
};

struct ECSWorld {
	Entity createEntity() {
		Entity id = nextEntity++;
		entities.push_back(id);
		return id;
	}

	size_t getEntityCount() const {
		return entities.size();
	}

	void clear() {
		entities.clear();
		transforms.clear();
		meshTags.clear();
		colors.clear();
		pbrMaterials.clear();
		nextEntity = 1;
	}

	void addTransform(Entity entity, const Transform& transform) {
		transforms[entity] = transform;
	}

	void addMeshTag(Entity entity, MeshTag tag) {
		meshTags[entity] = tag;
	}

	void addColor(Entity entity, const glm::vec4& color) {
		colors[entity] = InstanceColor{ color };
	}

	void addPBRMaterial(Entity entity, const PBRMaterial& material) {
		pbrMaterials[entity] = material;
	}

	Transform* getTransform(Entity entity) {
		auto it = transforms.find(entity);
		return it == transforms.end() ? nullptr : &it->second;
	}

	const Transform* getTransform(Entity entity) const {
		auto it = transforms.find(entity);
		return it == transforms.end() ? nullptr : &it->second;
	}

	const MeshTag* getMeshTag(Entity entity) const {
		auto it = meshTags.find(entity);
		return it == meshTags.end() ? nullptr : &it->second;
	}

	const InstanceColor* getColor(Entity entity) const {
		auto it = colors.find(entity);
		return it == colors.end() ? nullptr : &it->second;
	}

	const PBRMaterial* getPBRMaterial(Entity entity) const {
		auto it = pbrMaterials.find(entity);
		return it == pbrMaterials.end() ? nullptr : &it->second;
	}

	void collectRenderInstances(MeshTag tag, std::vector<RenderInstance>& out, size_t maxCount) const {
		out.clear();
		out.reserve(maxCount);
		for (Entity entity : entities) {
			const MeshTag* meshTag = getMeshTag(entity);
			if (!meshTag || *meshTag != tag) {
				continue;
			}
			const Transform* transform = getTransform(entity);
			if (!transform) {
				continue;
			}
			RenderInstance instance{
				.model = transform->getModelMatrix(),
				.color = glm::vec4(1.0f)
			};
			if (const InstanceColor* color = getColor(entity)) {
				instance.color = color->color;
			}
			out.push_back(instance);
			if (out.size() >= maxCount) {
				break;
			}
		}
	}

	void collectModels(MeshTag tag, std::vector<glm::mat4>& out, size_t maxCount) const {
		out.clear();
		out.reserve(maxCount);
		for (Entity entity : entities) {
			const MeshTag* meshTag = getMeshTag(entity);
			if (!meshTag || *meshTag != tag) {
				continue;
			}
			const Transform* transform = getTransform(entity);
			if (!transform) {
				continue;
			}
			out.push_back(transform->getModelMatrix());
			if (out.size() >= maxCount) {
				break;
			}
		}
	}

	void collectPBRInstances(MeshTag tag, std::vector<PBRInstance>& out, size_t maxCount) const {
		out.clear();
		out.reserve(maxCount);
		for (Entity entity : entities) {
			const MeshTag* meshTag = getMeshTag(entity);
			if (!meshTag || *meshTag != tag) {
				continue;
			}
			const Transform* transform = getTransform(entity);
			if (!transform) {
				continue;
			}
			PBRInstance instance{
				.model = transform->getModelMatrix(),
				.metallic = 0.5f,
				.roughness = 0.5f,
				.color = glm::vec3(1.0f)
			};
			if (const PBRMaterial* material = getPBRMaterial(entity)) {
				instance.metallic = material->metallic;
				instance.roughness = material->roughness;
				instance.color = material->color;
			}
			out.push_back(instance);
			if (out.size() >= maxCount) {
				break;
			}
		}
	}

	size_t getMeshTagCount(MeshTag tag) const {
		size_t count = 0;
		for (Entity entity : entities) {
			const MeshTag* meshTag = getMeshTag(entity);
			if (meshTag && *meshTag == tag) {
				++count;
			}
		}
		return count;
	}

private:
	Entity nextEntity = 1;
	std::vector<Entity> entities;
	std::unordered_map<Entity, Transform> transforms;
	std::unordered_map<Entity, MeshTag> meshTags;
	std::unordered_map<Entity, InstanceColor> colors;
	std::unordered_map<Entity, PBRMaterial> pbrMaterials;
};

struct Scene {
	ECSWorld world;
	Entity taauMovingProbe = 0;
	Entity taauEdgeProbe = 0;
	uint32_t maxInstances = 0;
	uint32_t cubeMeshIndex = 0;
	uint32_t sphereMeshIndex = 0;

	static constexpr uint32_t getDefaultMaxInstances() {
#if RENDERING_LEVEL < 3 || RENDERING_LEVEL == 7
		return 3;
#elif RENDERING_LEVEL == 3 || RENDERING_LEVEL == 4
		return 49;
#else
		return 9;
#endif
	}

	void initScene(const ResourceManager& resourceManager, unsigned int modelCount);

	uint32_t getMaxInstances() const {
		return maxInstances;
	}

	uint32_t getActiveInstanceCount() const {
		return static_cast<uint32_t>(world.getEntityCount());
	}

	uint32_t getMeshInstanceCount(MeshTag tag) const {
		return static_cast<uint32_t>(world.getMeshTagCount(tag));
	}

	void setCubeMeshIndex(uint32_t meshIndex) {
		cubeMeshIndex = meshIndex;
	}

	void setSphereMeshIndex(uint32_t meshIndex) {
		sphereMeshIndex = meshIndex;
	}

	void animateYaw(Entity entity, float deltaRadians) {
		Transform* transform = world.getTransform(entity);
		if (!transform) {
			return;
		}
		transform->rotation.y += deltaRadians;
	}
};

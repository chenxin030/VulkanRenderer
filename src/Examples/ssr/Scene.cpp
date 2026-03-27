#include "Scene.h"

Entity ECSWorld::createEntity() { Entity id = nextEntity++; entities.push_back(id); return id; }
size_t ECSWorld::getEntityCount() const { return entities.size(); }
void ECSWorld::clear() { entities.clear(); transforms.clear(); meshTags.clear(); colors.clear(); pbrMaterials.clear(); nextEntity = 1; }
void ECSWorld::addTransform(Entity entity, const Transform& transform) { transforms[entity] = transform; }
void ECSWorld::addMeshTag(Entity entity, MeshTag tag) { meshTags[entity] = tag; }
void ECSWorld::addColor(Entity entity, const glm::vec4& color) { colors[entity] = InstanceColor{ color }; }
void ECSWorld::addPBRMaterial(Entity entity, const PBRMaterial& material) { pbrMaterials[entity] = material; }
Transform* ECSWorld::getTransform(Entity entity) { auto it = transforms.find(entity); return it == transforms.end() ? nullptr : &it->second; }
const Transform* ECSWorld::getTransform(Entity entity) const { auto it = transforms.find(entity); return it == transforms.end() ? nullptr : &it->second; }
const MeshTag* ECSWorld::getMeshTag(Entity entity) const { auto it = meshTags.find(entity); return it == meshTags.end() ? nullptr : &it->second; }
const InstanceColor* ECSWorld::getColor(Entity entity) const { auto it = colors.find(entity); return it == colors.end() ? nullptr : &it->second; }
const PBRMaterial* ECSWorld::getPBRMaterial(Entity entity) const { auto it = pbrMaterials.find(entity); return it == pbrMaterials.end() ? nullptr : &it->second; }
void ECSWorld::collectRenderInstances(MeshTag tag, std::vector<RenderInstance>& out, size_t maxCount) const { out.clear(); out.reserve(maxCount); for (Entity entity : entities) { const MeshTag* meshTag = getMeshTag(entity); if (!meshTag || *meshTag != tag) continue; const Transform* transform = getTransform(entity); if (!transform) continue; RenderInstance instance{ .model = transform->getModelMatrix(), .color = glm::vec4(1.0f) }; if (const InstanceColor* color = getColor(entity)) instance.color = color->color; out.push_back(instance); if (out.size() >= maxCount) break; } }
void ECSWorld::collectModels(MeshTag tag, std::vector<glm::mat4>& out, size_t maxCount) const { out.clear(); out.reserve(maxCount); for (Entity entity : entities) { const MeshTag* meshTag = getMeshTag(entity); if (!meshTag || *meshTag != tag) continue; const Transform* transform = getTransform(entity); if (!transform) continue; out.push_back(transform->getModelMatrix()); if (out.size() >= maxCount) break; } }
void ECSWorld::collectPBRInstances(MeshTag tag, std::vector<PBRInstance>& out, size_t maxCount) const { out.clear(); out.reserve(maxCount); for (Entity entity : entities) { const MeshTag* meshTag = getMeshTag(entity); if (!meshTag || *meshTag != tag) continue; const Transform* transform = getTransform(entity); if (!transform) continue; PBRInstance instance{ .model = transform->getModelMatrix(), .metallic = 0.5f, .roughness = 0.5f, .color = glm::vec3(1.0f) }; if (const PBRMaterial* material = getPBRMaterial(entity)) { instance.metallic = material->metallic; instance.roughness = material->roughness; instance.color = material->color; } out.push_back(instance); if (out.size() >= maxCount) break; } }
size_t ECSWorld::getMeshTagCount(MeshTag tag) const { size_t count = 0; for (Entity entity : entities) { const MeshTag* meshTag = getMeshTag(entity); if (meshTag && *meshTag == tag) ++count; } return count; }

void Scene::initScene(const ResourceManager& resourceManager, unsigned int modelCount) {
    world.clear();
    maxInstances = static_cast<uint32_t>(modelCount);
    cubeMeshIndex = 0;
    sphereMeshIndex = 1;
    (void)resourceManager;

    setCubeMeshIndex(0);
    setSphereMeshIndex(1);
    if (modelCount > 0) { Entity ground = world.createEntity(); world.addTransform(ground, Transform{ {0.0f, -0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {7.0f, 0.05f, 7.0f} }); world.addMeshTag(ground, MeshTag::Cube); world.addColor(ground, glm::vec4(0.78f, 0.78f, 0.80f, 1.0f)); }
    if (modelCount > 1) { Entity cube = world.createEntity(); world.addTransform(cube, Transform{ {-1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.35f, 0.35f, 0.35f} }); world.addMeshTag(cube, MeshTag::Cube); world.addColor(cube, glm::vec4(0.20f, 0.25f, 0.90f, 1.0f)); }
    if (modelCount > 2) { Entity cube = world.createEntity(); world.addTransform(cube, Transform{ {1.6f, 0.0f, -1.2f}, {0.0f, 0.0f, 0.0f}, {0.45f, 0.45f, 0.45f} }); world.addMeshTag(cube, MeshTag::Cube); world.addColor(cube, glm::vec4(0.90f, 0.25f, 0.20f, 1.0f)); }
}

uint32_t Scene::getMaxInstances() const { return maxInstances; }
uint32_t Scene::getActiveInstanceCount() const { return static_cast<uint32_t>(world.getEntityCount()); }
uint32_t Scene::getMeshInstanceCount(MeshTag tag) const { return static_cast<uint32_t>(world.getMeshTagCount(tag)); }
void Scene::setCubeMeshIndex(uint32_t meshIndex) { cubeMeshIndex = meshIndex; }
void Scene::setSphereMeshIndex(uint32_t meshIndex) { sphereMeshIndex = meshIndex; }
void Scene::animateYaw(Entity entity, float deltaRadians) { Transform* transform = world.getTransform(entity); if (transform) transform->rotation.y += deltaRadians; }

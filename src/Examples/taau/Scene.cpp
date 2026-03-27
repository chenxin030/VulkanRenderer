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
    if (modelCount > 0) { Entity ground = world.createEntity(); world.addTransform(ground, Transform{ {0.0f, -1.1f, 0.0f}, {0.0f, 0.0f, 0.0f}, {7.5f, 0.03f, 7.5f} }); world.addMeshTag(ground, MeshTag::Cube); world.addColor(ground, glm::vec4(0.78f, 0.78f, 0.80f, 1.0f)); }
    if (modelCount > 1) { Entity pillar = world.createEntity(); world.addTransform(pillar, Transform{ {-3.0f, -0.2f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.08f, 2.8f, 0.08f} }); world.addMeshTag(pillar, MeshTag::Cube); world.addColor(pillar, glm::vec4(0.35f, 0.35f, 0.35f, 1.0f)); }
    if (modelCount > 2) { Entity pillar = world.createEntity(); world.addTransform(pillar, Transform{ {-2.6f, 0.2f, 0.3f}, {0.0f, 0.0f, 0.0f}, {0.10f, 2.2f, 0.10f} }); world.addMeshTag(pillar, MeshTag::Cube); world.addColor(pillar, glm::vec4(0.90f, 0.25f, 0.20f, 1.0f)); }
    if (modelCount > 3) { taauMovingProbe = world.createEntity(); world.addTransform(taauMovingProbe, Transform{ {2.5f, 0.35f, -1.6f}, {0.0f, 0.0f, 0.0f}, {0.35f, 0.35f, 0.35f} }); world.addMeshTag(taauMovingProbe, MeshTag::Cube); world.addColor(taauMovingProbe, glm::vec4(0.20f, 0.75f, 0.95f, 1.0f)); }
    if (modelCount > 4) { taauEdgeProbe = world.createEntity(); world.addTransform(taauEdgeProbe, Transform{ {3.7f, -0.35f, -2.8f}, {0.0f, 0.0f, 0.0f}, {0.22f, 1.6f, 0.22f} }); world.addMeshTag(taauEdgeProbe, MeshTag::Cube); world.addColor(taauEdgeProbe, glm::vec4(0.25f, 0.85f, 0.30f, 1.0f)); }
    if (modelCount > 5) { Entity bar = world.createEntity(); world.addTransform(bar, Transform{ {0.0f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.12f, 0.9f, 0.12f} }); world.addMeshTag(bar, MeshTag::Cube); world.addColor(bar, glm::vec4(0.85f, 0.70f, 0.20f, 1.0f)); }
    if (modelCount > 6) { Entity bar = world.createEntity(); world.addTransform(bar, Transform{ {0.25f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.10f, 0.9f, 0.10f} }); world.addMeshTag(bar, MeshTag::Cube); world.addColor(bar, glm::vec4(0.30f, 0.30f, 0.30f, 1.0f)); }
    if (modelCount > 7) { Entity bar = world.createEntity(); world.addTransform(bar, Transform{ {-0.25f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.14f, 0.9f, 0.14f} }); world.addMeshTag(bar, MeshTag::Cube); world.addColor(bar, glm::vec4(0.40f, 0.40f, 0.40f, 1.0f)); }
    if (modelCount > 8) { Entity bar = world.createEntity(); world.addTransform(bar, Transform{ {0.55f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.08f, 0.9f, 0.08f} }); world.addMeshTag(bar, MeshTag::Cube); world.addColor(bar, glm::vec4(0.50f, 0.50f, 0.50f, 1.0f)); }
}

uint32_t Scene::getMaxInstances() const { return maxInstances; }
uint32_t Scene::getActiveInstanceCount() const { return static_cast<uint32_t>(world.getEntityCount()); }
uint32_t Scene::getMeshInstanceCount(MeshTag tag) const { return static_cast<uint32_t>(world.getMeshTagCount(tag)); }
void Scene::setCubeMeshIndex(uint32_t meshIndex) { cubeMeshIndex = meshIndex; }
void Scene::setSphereMeshIndex(uint32_t meshIndex) { sphereMeshIndex = meshIndex; }
void Scene::animateYaw(Entity entity, float deltaRadians) { Transform* transform = world.getTransform(entity); if (transform) transform->rotation.y += deltaRadians; }

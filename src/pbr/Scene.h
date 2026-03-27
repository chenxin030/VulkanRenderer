#pragma once

#include <Core/ResourceManager.h>
#include <cstdint>
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
    Entity createEntity();
    size_t getEntityCount() const;
    void clear();
    void addTransform(Entity entity, const Transform& transform);
    void addMeshTag(Entity entity, MeshTag tag);
    void addColor(Entity entity, const glm::vec4& color);
    void addPBRMaterial(Entity entity, const PBRMaterial& material);
    Transform* getTransform(Entity entity);
    const Transform* getTransform(Entity entity) const;
    const MeshTag* getMeshTag(Entity entity) const;
    const InstanceColor* getColor(Entity entity) const;
    const PBRMaterial* getPBRMaterial(Entity entity) const;
    void collectRenderInstances(MeshTag tag, std::vector<RenderInstance>& out, size_t maxCount) const;
    void collectModels(MeshTag tag, std::vector<glm::mat4>& out, size_t maxCount) const;
    void collectPBRInstances(MeshTag tag, std::vector<PBRInstance>& out, size_t maxCount) const;
    size_t getMeshTagCount(MeshTag tag) const;

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
        return 49;
    }

    void initScene(const ResourceManager& resourceManager, unsigned int modelCount);
    uint32_t getMaxInstances() const;
    uint32_t getActiveInstanceCount() const;
    uint32_t getMeshInstanceCount(MeshTag tag) const;
    void setCubeMeshIndex(uint32_t meshIndex);
    void setSphereMeshIndex(uint32_t meshIndex);
    void animateYaw(Entity entity, float deltaRadians);
};

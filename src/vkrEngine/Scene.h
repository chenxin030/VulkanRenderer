#pragma once

#include "VkrModel.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>

// ---- Renderable: associates a model with a world transform ----

struct VkrRenderable
{
    VkrModel* model = nullptr;
    glm::mat4 transform = glm::mat4(1.0f);
    std::string debugName;

    void setPosition(const glm::vec3& pos)
    {
        transform = glm::translate(glm::mat4(1.0f), pos);
    }

    void setTransform(const glm::vec3& pos, const glm::vec3& rotEulerDeg, const glm::vec3& scale)
    {
        transform = glm::mat4(1.0f);
        transform = glm::translate(transform, pos);
        transform = glm::rotate(transform, glm::radians(rotEulerDeg.y), glm::vec3(0, 1, 0));
        transform = glm::rotate(transform, glm::radians(rotEulerDeg.x), glm::vec3(1, 0, 0));
        transform = glm::rotate(transform, glm::radians(rotEulerDeg.z), glm::vec3(0, 0, 1));
        transform = glm::scale(transform, scale);
    }
};

// ---- Point light ----

struct VkrPointLight
{
    glm::vec3 position{ 0.0f };
    glm::vec3 color{ 1.0f };
    float     intensity = 1.0f;
    float     radius = 10.0f;
};

// ---- Directional light ----

struct VkrDirectionalLight
{
    glm::vec3 direction{ 0.5f, -0.8f, 0.3f };
    glm::vec3 color{ 1.0f, 0.98f, 0.95f };
    float     intensity = 1.0f;
};

// ---- Scene ----

class VkrScene
{
public:
    // ---- Models ----

    void addModel(VkrModel* model, const glm::mat4& transform, const std::string& debugName = "")
    {
        VkrRenderable r;
        r.model = model;
        r.transform = transform;
        r.debugName = debugName.empty() ? model->name : debugName;
        m_renderables.push_back(r);
    }

    void addModel(VkrModel* model, const glm::vec3& position, const std::string& debugName = "")
    {
        VkrRenderable r;
        r.model = model;
        r.setPosition(position);
        r.debugName = debugName.empty() ? model->name : debugName;
        m_renderables.push_back(r);
    }

    // ---- Lights ----

    void addPointLight(const glm::vec3& pos, const glm::vec3& color, float intensity, float radius = 10.0f)
    {
        m_pointLights.push_back({ pos, color, intensity, radius });
    }

    void setDirectionalLight(const glm::vec3& dir, const glm::vec3& color, float intensity)
    {
        m_dirLight.direction = glm::normalize(dir);
        m_dirLight.color = color;
        m_dirLight.intensity = intensity;
    }

    // ---- Accessors ----

    [[nodiscard]] const std::vector<VkrRenderable>& renderables()   const { return m_renderables; }
    [[nodiscard]] const std::vector<VkrPointLight>& pointLights()   const { return m_pointLights; }
    [[nodiscard]] const VkrDirectionalLight& dirLight()      const { return m_dirLight; }

    // ---- Statistics ----

    [[nodiscard]] uint32_t totalDrawCalls() const
    {
        uint32_t count = 0;
        for (const auto& r : m_renderables)
        {
            if (r.model) count += static_cast<uint32_t>(r.model->subMeshes.size());
        }
        return count;
    }

    [[nodiscard]] uint32_t totalTriangles() const
    {
        uint32_t count = 0;
        for (const auto& r : m_renderables)
        {
            if (r.model) count += r.model->totalTriangles;
        }
        return count;
    }

private:
    std::vector<VkrRenderable>       m_renderables;
    std::vector<VkrPointLight>       m_pointLights;
    VkrDirectionalLight              m_dirLight;
};

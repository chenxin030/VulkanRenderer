#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#if RENDERING_LEVEL == 6

struct ShadowInstanceData
{
    glm::mat4 model;
    glm::vec4 color;
};

struct TAAUParams
{
    float blendFactor;
    float reactiveClamp;
    float antiFlicker;
    float velocityScale;
};

struct TAAUSceneState
{
    glm::mat4 prevViewProj;
    glm::vec3 fastMoveVelocity;
    float time = 0.0f;
    bool freezeHistory = false;
};

static TAAUSceneState gTaauSceneState{};
static TAAUParams gTaauParams{ 0.85f, 0.85f, 0.6f, 1.15f };


void Renderer::updateTAAUScene(float deltaTime)
{
    gTaauSceneState.time += deltaTime;

    if (scene != nullptr)
    {
        if (!gTaauSceneState.freezeHistory && scene->taauMovingProbe != 0)
        {
            float fastPhase = gTaauSceneState.time * 2.4f;
            float zigzag = std::sin(fastPhase);
            float offset = std::sin(fastPhase * 1.5f) * 0.6f;
            Transform* moving = scene->world.getTransform(scene->taauMovingProbe);
            if (moving)
            {
                moving->position.x = 2.5f + zigzag * 1.2f;
                moving->position.z = -1.6f + offset;
            }
        }

        if (scene->taauEdgeProbe != 0)
        {
            Transform* edgeProbe = scene->world.getTransform(scene->taauEdgeProbe);
            if (edgeProbe)
            {
                edgeProbe->position.x = 3.7f + std::sin(gTaauSceneState.time * 0.6f) * 0.25f;
            }
        }
    }

    gTaauSceneState.fastMoveVelocity = glm::vec3(
        std::cos(gTaauSceneState.time * 2.4f),
        0.0f,
        std::sin(gTaauSceneState.time * 1.5f)
    ) * gTaauParams.velocityScale;
}

void Renderer::updateTAAUHistory(const glm::mat4& currentViewProj)
{
    gTaauSceneState.prevViewProj = currentViewProj;
}

void Renderer::updateTAAUUI()
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr)
    {
        return;
    }

    ImGui::Begin("TAAU Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("TAAU coverage goals");
    ImGui::BulletText("Ghosting: blendFactor/antiFlicker");
    ImGui::BulletText("Thin-line shimmer: reactiveClamp");
    ImGui::BulletText("Fast motion: velocityScale");
    ImGui::BulletText("Edge & high-frequency: reactiveClamp + antiFlicker");

    ImGui::Separator();
    ImGui::SliderFloat("BlendFactor", &gTaauParams.blendFactor, 0.2f, 0.98f);
    ImGui::SliderFloat("ReactiveClamp", &gTaauParams.reactiveClamp, 0.2f, 1.2f);
    ImGui::SliderFloat("AntiFlicker", &gTaauParams.antiFlicker, 0.0f, 1.0f);
    ImGui::SliderFloat("VelocityScale", &gTaauParams.velocityScale, 0.2f, 2.5f);
    ImGui::Checkbox("FreezeHistory", &gTaauSceneState.freezeHistory);
    ImGui::End();
}


#endif

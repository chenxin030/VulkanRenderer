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

static void setTAAUColors(std::vector<ShadowInstanceData>& instanceData, uint32_t cubeInstanceCount)
{
    if (cubeInstanceCount > 0) instanceData[0].color = glm::vec4(0.78f, 0.78f, 0.80f, 1.0f);
    if (cubeInstanceCount > 1) instanceData[1].color = glm::vec4(0.35f, 0.35f, 0.35f, 1.0f);
    if (cubeInstanceCount > 2) instanceData[2].color = glm::vec4(0.90f, 0.25f, 0.20f, 1.0f);
    if (cubeInstanceCount > 3) instanceData[3].color = glm::vec4(0.20f, 0.75f, 0.95f, 1.0f);
    if (cubeInstanceCount > 4) instanceData[4].color = glm::vec4(0.25f, 0.85f, 0.30f, 1.0f);
    if (cubeInstanceCount > 5) instanceData[5].color = glm::vec4(0.85f, 0.70f, 0.20f, 1.0f);
    if (cubeInstanceCount > 6) instanceData[6].color = glm::vec4(0.30f, 0.30f, 0.30f, 1.0f);
    if (cubeInstanceCount > 7) instanceData[7].color = glm::vec4(0.40f, 0.40f, 0.40f, 1.0f);
    if (cubeInstanceCount > 8) instanceData[8].color = glm::vec4(0.50f, 0.50f, 0.50f, 1.0f);
}

void Renderer::updateTAAUScene(float deltaTime)
{
    gTaauSceneState.time += deltaTime;

    if (!gTaauSceneState.freezeHistory && resourceManager->transforms.size() > 3)
    {
        float fastPhase = gTaauSceneState.time * 2.4f;
        float zigzag = std::sin(fastPhase);
        float offset = std::sin(fastPhase * 1.5f) * 0.6f;
        auto& moving = resourceManager->transforms[3];
        moving.position.x = 2.5f + zigzag * 1.2f;
        moving.position.z = -1.6f + offset;
    }

    if (resourceManager->transforms.size() > 4)
    {
        auto& edgeProbe = resourceManager->transforms[4];
        edgeProbe.position.x = 3.7f + std::sin(gTaauSceneState.time * 0.6f) * 0.25f;
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

void Renderer::updateShadowBuffers(uint32_t currentImage)
{
    float deltaTime = platform->frameTimer;
    updateTAAUScene(deltaTime);

    SceneUBO sceneUbo{
        .projection = glm::perspective(glm::radians(camera.Zoom),
            static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
            0.1f, 100.0f),
        .view = camera.GetViewMatrix(),
        .camPos = camera.Position
    };
    sceneUbo.projection[1][1] *= -1;
    memcpy(sceneUboResources.BuffersMapped[currentImage], &sceneUbo, sizeof(sceneUbo));

    static float lightAngle = 0.0f;
    lightAngle += deltaTime * 0.35f;
    glm::vec3 lightDir = glm::normalize(glm::vec3(std::cos(lightAngle) * 0.6f, -1.0f, std::sin(lightAngle) * 0.6f));
    glm::vec3 lightPos = -lightDir * 12.0f;
    glm::vec3 target = glm::vec3(0.0f, -0.5f, 0.0f);

    glm::mat4 lightView = glm::lookAt(lightPos, target, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProj = glm::ortho(-7.0f, 7.0f, -7.0f, 7.0f, 0.1f, 30.0f);
    lightProj[1][1] *= -1;

    ShadowUBO shadowUbo{
        .lightViewProj = lightProj * lightView,
        .dirLightDirIntensity = glm::vec4(lightDir, dirLightIntensity),
        .dirLightColor = glm::vec4(1.0f),
        .pointLightPosIntensity = glm::vec4(2.5f, 1.2f, 2.0f, pointLightIntensity),
        .pointLightColor = glm::vec4(1.0f),
        .areaLightPosIntensity = glm::vec4(-2.5f, 2.5f, -1.2f, areaLightIntensity),
        .areaLightColor = glm::vec4(1.0f, 0.95f, 0.85f, 1.0f),
        .areaLightU = glm::vec4(0.9f, 0.0f, 0.0f, 0.0f),
        .areaLightV = glm::vec4(0.0f, 0.0f, 0.9f, 0.0f),
    };
    memcpy(shadowUboResources.BuffersMapped[currentImage], &shadowUbo, sizeof(shadowUbo));

    ShadowParamsUBO shadowParams{
        .shadowFilterMode = shadowFilterMode,
        .pcfRadiusTexels = pcfRadiusTexels,
        .pcssLightSizeTexels = pcssLightSizeTexels,
        .shadowBiasMin = 0.0006f,
        .invShadowMapSize = glm::vec2(1.0f / float(shadowMapExtent.width), 1.0f / float(shadowMapExtent.height)),
        .padding0 = glm::vec2(0.0f)
    };
    memcpy(shadowParamsUboResources.BuffersMapped[currentImage], &shadowParams, sizeof(shadowParams));

    std::vector<ShadowInstanceData> instanceData(MAX_OBJECTS);
    for (uint32_t i = 0; i < cubeInstanceCount; ++i)
    {
        instanceData[i].model = resourceManager->transforms[i].getModelMatrix();
    }

    setTAAUColors(instanceData, cubeInstanceCount);

    memcpy(shadowInstanceBufferResources.BuffersMapped[currentImage], instanceData.data(), sizeof(ShadowInstanceData) * MAX_OBJECTS);

    updateTAAUHistory(sceneUbo.projection * sceneUbo.view);
}

#endif

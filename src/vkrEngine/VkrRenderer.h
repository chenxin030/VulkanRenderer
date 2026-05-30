#pragma once

#include <Base/VulkanBase.h>
#include <Base/VulkanBase_UI.h>

#include "GpuProfiler.h"
#include "Scene.h"
#include "VkrModel.h"

struct VkrRenderer final : VulkanBase
{
public:
    // ---- Lifecycle ----
    void initialize(Platform* _platform);
    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

    // ---- Override for ImGui ----
    void updateUIPanel() override;
    void recreateSwapChain() override;

    // ---- Scene access ----
    VkrScene scene;
    VkrModel  sponzaModel;

private:
    // ---- Constants ----
    static constexpr uint32_t MAX_MATERIALS = 128;

    // ---- UBO ----
    struct SceneUBO
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 camPos;
        float pad0;
    };

    // ---- GPU Resources (ordered for correct RAII destruction) ----

    // Descriptor pool — FIRST: destroyed last, after all descriptor sets
    vk::raii::DescriptorPool descriptorPool = nullptr;

    // Descriptor set layouts
    vk::raii::DescriptorSetLayout sceneDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout materialDescriptorSetLayout = nullptr;

    // Descriptor sets (allocated from pool, RAII destructors run before pool)
    MeshBuffer sceneUboResources;
    std::vector<vk::raii::DescriptorSet> materialDescriptorSets;

    // Pipeline (depends on layouts)
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline       pipeline = nullptr;

    // Per-swapchain-image semaphores (fixes Device Lost from semaphore reuse)
    std::vector<vk::raii::Semaphore> m_imageAcquiredSemaphores;
    std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;

    // Other GPU resources
    VkrGpuProfiler gpuProfiler;
    vk::raii::Sampler materialSampler = nullptr;
    TextureData dummyWhiteTexture;

    // ---- Stats (updated each frame) ----
    uint32_t statDrawCalls = 0;
    uint32_t statTriangles = 0;
    float    statCpuMs = 0.0f;
    bool     showStatsWindow = true;
    bool     showProfilerWindow = true;

    // ---- Internal Methods ----

    /** Create GPU buffers for a model (vertex + index buffer). */
    bool createModelGpuResources(VkrModel& model);

    /** Create GPU resources (image, image view, sampler) for a material's textures. */
    bool createMaterialGpuResources(VkrMaterial& mat, const std::string& textureBasePath);

    /** Create descriptor set layouts and pool. */
    bool createDescriptors();

    /** Create per-material descriptor sets. */
    bool createMaterialDescriptorSets();

    /** Create the graphics pipeline. */
    bool createPipeline();

    /** Update scene UBO for the current frame. */
    void updateSceneUBO(uint32_t frameIndex);

    /** Main command buffer recording. */
    void recordCommandBuffer(uint32_t imageIndex);

    /** Render a single model (all sub-meshes). */
    void renderModel(vk::CommandBuffer cmd, const VkrModel& model, const glm::mat4& transform);

    /** Initialize ImGui. */
    bool initUI();

    /** Load a texture from file into the TextureCache-like system. */
    bool loadMaterialTexture(VkrMaterialTexture& tex, const std::string& fullPath);

    /** Create a default 1x1 white texture for fallback. */
    bool createDummyWhiteTexture();
};

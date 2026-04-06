#pragma once

#include <Base/VulkanBase.h>

#include "FrameGraph.h"
#include "GpuProfiler.h"
#include "RenderBatcher.h"
#include "ThreadPool.h"

#include <future>
#include <string>
#include <vector>

struct MultithreadedRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    struct GlobalUBO
    {
        glm::mat4 view;
        glm::mat4 proj;
    };

    struct InstanceData
    {
        glm::mat4 model;
    };

    struct WorkerRecordStats
    {
        uint32_t drawCalls = 0;
        uint32_t staticDrawCalls = 0;
        uint32_t dynamicDrawCalls = 0;
        float recordMs = 0.0f;
    };

    struct PercentileStats
    {
        float avg = 0.0f;
        float p95 = 0.0f;
        float p99 = 0.0f;
    };

    struct BenchmarkStats
    {
        PercentileStats frameMs;
        PercentileStats recordMs;
        uint32_t sampleCount = 0;
        bool valid = false;
    };

    enum class BenchmarkMode : int
    {
        None = 0,
        SingleThread = 1,
        MultiThread = 2,
        Done = 3
    };

    enum class ScenePreset : int
    {
        Small = 0,
        Medium = 1,
        Large = 2
    };

    struct ScenePresetConfig
    {
        uint32_t staticInstanceCount = 0;
        uint32_t dynamicInstanceCount = 0;
        uint32_t particleCount = 0;
        uint32_t pointLightCount = 0;
        float particleEmissionRate = 0.0f;
    };

    static constexpr uint32_t DEFAULT_WORKER_THREADS = 4;

    bool enableMultiThreadRecording = true;
    bool enableAsyncCompute = true;
    uint32_t workerThreadCount = DEFAULT_WORKER_THREADS;

    ScenePreset currentPreset = ScenePreset::Medium;
    ScenePresetConfig currentPresetConfig{ .staticInstanceCount = 20000, .dynamicInstanceCount = 5000, .particleCount = 100000, .pointLightCount = 256, .particleEmissionRate = 6000.0f };
    uint32_t sceneInstanceCount = 25000;

    uint32_t activeParticleCapacity = 100000;
    uint32_t activeLightCount = 256;
    float activeParticleEmissionRate = 6000.0f;

    bool sceneBuffersCreated = false;
    bool presetResourcesDirty = false;

    float frameMs = 0.0f;
    float fps = 0.0f;

    bool benchmarkAutoRun = false;
    bool benchmarkRestoreMultiThreadState = true;
    BenchmarkMode benchmarkMode = BenchmarkMode::None;
    float benchmarkDurationSeconds = 10.0f;
    float benchmarkElapsedSeconds = 0.0f;

    std::vector<float> benchmarkSingleFrameSamples;
    std::vector<float> benchmarkSingleRecordSamples;
    std::vector<float> benchmarkMultiFrameSamples;
    std::vector<float> benchmarkMultiRecordSamples;

    BenchmarkStats benchmarkSingleStats{};
    BenchmarkStats benchmarkMultiStats{};

    Mesh mesh;
    TextureData texture;

    MeshBuffer globalUboResources;
    MeshBuffer instanceBufferResources;
    MeshBuffer particleBufferResources;
    MeshBuffer lightBufferResources;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;

    std::vector<vk::raii::CommandPool> secondaryCommandPools;
    std::vector<vk::raii::CommandBuffer> secondaryStaticCommandBuffers;
    std::vector<vk::raii::CommandBuffer> secondaryDynamicCommandBuffers;

    ThreadPool threadPool;
    FrameGraph frameGraph;
    GpuProfiler gpuProfiler;

    std::vector<WorkerRecordStats> workerStats;

    bool createSceneBuffers();
    bool recreatePresetDrivenBuffers();
    bool createSecondaryCommandResources();
    bool createDescriptors();
    void updateDescriptorSets();
    bool createPipeline();
    bool initFrameGraph();
    bool initThreading();
    bool initUI();

    void updateFrameData(uint32_t frameIndex);
    void updateUIFrame();
    void updateProfilerUI();
    void updateInstanceBuffer(uint32_t frameIndex);

    void dispatchWorkerRecording(uint32_t frameIndex);
    WorkerRecordStats recordWorkerRange(uint32_t frameIndex, RenderBatch batch, bool dynamicBatch);

    void recordPrimaryCommandBuffer(uint32_t imageIndex);
    void recordUI(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex);

    static PercentileStats calcPercentiles(const std::vector<float>& samples);
    static BenchmarkStats buildBenchmarkStats(const std::vector<float>& frameSamples, const std::vector<float>& recordSamples);
    void resetBenchmarkSamples();
    void startAutoBenchmark();
    void updateBenchmarkFlow(float dt, float cpuRecordMs);

    static const char* presetLabel(ScenePreset preset);
    static ScenePresetConfig presetConfig(ScenePreset preset);
    void applyPreset(ScenePreset preset);

    void rebuildThreadPoolIfNeeded();
};

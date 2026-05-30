#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

/**
 * GPU Profiler using Vulkan Timestamp Queries.
 *
 * Usage per frame:
 *   profiler.beginFrame();
 *   profiler.beginPass(cmd, "Shadow");
 *   ... draw commands ...
 *   profiler.endPass(cmd);
 *   profiler.beginPass(cmd, "GBuffer");
 *   ... draw commands ...
 *   profiler.endPass(cmd);
 *   profiler.endFrame();
 *
 * After endFrame(), getTimings() returns the latest pass durations in ms.
 */
class VkrGpuProfiler
{
public:
    static constexpr uint32_t MAX_TIMESTAMPS = 64; // supports up to 32 passes with begin+end
    static constexpr uint32_t MAX_PASSES = 32;

    struct PassTiming
    {
        std::string name;
        float       durationMs = 0.0f;
    };

    VkrGpuProfiler() = default;

    /**
     * Initialize the query pool. Must be called after device creation.
     * @param device       Vulkan logical device
     * @param timestampPeriod  physicalDeviceProperties.limits.timestampPeriod (in nanoseconds)
     */
    void init(vk::raii::Device& device, float timestampPeriodNs);

    /** Call at the START of a frame (before any draw commands). */
    void beginFrame(vk::CommandBuffer cmd);

    /** Insert a timestamp marking the START of a named pass. */
    void beginPass(vk::CommandBuffer cmd, const std::string& name);

    /** Insert a timestamp marking the END of a named pass. */
    void endPass(vk::CommandBuffer cmd);

    /**
     * Call at the START of render() (before recordCommandBuffer).
     * Reads back the PREVIOUS frame's GPU timestamps.
     */
    void endFrame();

    /** Get the latest frame's pass timings. */
    [[nodiscard]] const std::vector<PassTiming>& getTimings() const { return m_timings; }

    /** Total GPU frame time (sum of all passes), in ms. */
    [[nodiscard]] float totalGpuMs() const
    {
        float total = 0.0f;
        for (const auto& t : m_timings) total += t.durationMs;
        return total;
    }

private:
    vk::raii::QueryPool m_queryPool = nullptr;
    float               m_timestampPeriodNs = 1.0f;
    uint32_t            m_queryIndex = 0;
    uint32_t            m_passCount = 0;
    bool                m_initialized = false;
    bool                m_hasPendingFrame = false; // true after first beginFrame(), enables endFrame readback

    // Pending pass names for the current frame
    std::string m_pendingNames[MAX_PASSES];

    // Latest results
    std::vector<PassTiming> m_timings;
};

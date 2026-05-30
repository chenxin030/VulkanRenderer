#include "GpuProfiler.h"

#include <cassert>
#include <cstring>
#include <iostream>

void VkrGpuProfiler::init(vk::raii::Device& device, float timestampPeriodNs)
{
    if (m_initialized) return;

    m_timestampPeriodNs = timestampPeriodNs;

    vk::QueryPoolCreateInfo poolInfo{};
    poolInfo.queryType = vk::QueryType::eTimestamp;
    poolInfo.queryCount = MAX_TIMESTAMPS;

    m_queryPool = vk::raii::QueryPool(device, poolInfo);
    m_initialized = true;

    std::cout << "[GpuProfiler] Initialized (timestamp period: "
        << timestampPeriodNs << " ns)" << std::endl;
}

void VkrGpuProfiler::beginFrame(vk::CommandBuffer cmd)
{
    assert(m_initialized);
    m_queryIndex = 0;
    m_passCount = 0;
    m_hasPendingFrame = true;
    cmd.resetQueryPool(*m_queryPool, 0, MAX_TIMESTAMPS);

    // Write the very first timestamp as "frame start"
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *m_queryPool, m_queryIndex++);

    // Store as a dummy pass for frame start
    m_pendingNames[0] = "__frame_start";
}

void VkrGpuProfiler::beginPass(vk::CommandBuffer cmd, const std::string& name)
{
    assert(m_initialized);
    assert(m_passCount < MAX_PASSES);
    assert(m_queryIndex + 1 < MAX_TIMESTAMPS);

    m_pendingNames[m_passCount] = name;
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *m_queryPool, m_queryIndex++);
}

void VkrGpuProfiler::endPass(vk::CommandBuffer cmd)
{
    assert(m_initialized);
    assert(m_queryIndex < MAX_TIMESTAMPS);

    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *m_queryPool, m_queryIndex++);
    ++m_passCount;
}

void VkrGpuProfiler::endFrame()
{
    assert(m_initialized);

    // First frame: no previous results to read back
    if (!m_hasPendingFrame) return;

    // Read back timestamps using RAII QueryPool template API (no WAIT_BIT — results should be ready)
    auto [result, timestamps] = m_queryPool.getResults<uint64_t>(
        0, m_queryIndex,
        sizeof(uint64_t) * m_queryIndex, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);

    if (result != vk::Result::eSuccess && result != vk::Result::eNotReady)
    {
        std::cerr << "[GpuProfiler] Failed to read timestamp results: " << vk::to_string(result) << std::endl;
        return;
    }

    if (timestamps.empty()) return;

    uint64_t rawTimestamps[MAX_TIMESTAMPS] = {};
    std::copy_n(timestamps.begin(), std::min(timestamps.size(), size_t(MAX_TIMESTAMPS)), rawTimestamps);

    // Convert to ms and build timings
    m_timings.clear();
    m_timings.reserve(m_passCount);

    // rawTimestamps[0] = frame_start
    // For each pass i: rawTimestamps[1 + 2*i] = begin, rawTimestamps[1 + 2*i + 1] = end
    for (uint32_t i = 0; i < m_passCount; ++i)
    {
        uint64_t begin = rawTimestamps[1 + 2 * i];     // beginPass wrote this
        uint64_t end = rawTimestamps[1 + 2 * i + 1]; // endPass wrote this

        float durationNs = static_cast<float>(end - begin) * m_timestampPeriodNs;
        float durationMs = durationNs * 1e-6f;

        PassTiming t;
        t.name = m_pendingNames[i];
        t.durationMs = durationMs;
        m_timings.push_back(t);
    }
}

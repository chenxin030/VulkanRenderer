#include "GpuProfiler.h"

#include <algorithm>

void GpuProfiler::beginFrame()
{
    timings_.clear();
}

void GpuProfiler::endFrame()
{
    // Skeleton placeholder. Real implementation should resolve Vulkan timestamp queries here.
}

void GpuProfiler::setPassTiming(const std::string& passName, float timeMs)
{
    auto it = std::find_if(timings_.begin(), timings_.end(), [&](const GpuPassTiming& item)
    {
        return item.name == passName;
    });

    if (it != timings_.end())
    {
        it->timeMs = timeMs;
    }
    else
    {
        timings_.push_back({ passName, timeMs });
    }
}

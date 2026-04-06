#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GpuPassTiming
{
    std::string name;
    float timeMs = 0.0f;
};

class GpuProfiler
{
public:
    void beginFrame();
    void endFrame();

    void setPassTiming(const std::string& passName, float timeMs);
    [[nodiscard]] const std::vector<GpuPassTiming>& timings() const { return timings_; }

private:
    std::vector<GpuPassTiming> timings_;
};

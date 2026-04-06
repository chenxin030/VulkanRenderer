#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct FramePass
{
    std::string name;
    bool enabled = true;
};

class FrameGraph
{
public:
    void reset();
    void addPass(const std::string& passName, bool enabled = true);
    [[nodiscard]] const std::vector<FramePass>& passes() const { return passes_; }

private:
    std::vector<FramePass> passes_;
};

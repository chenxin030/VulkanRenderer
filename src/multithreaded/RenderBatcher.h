#pragma once

#include <cstdint>
#include <vector>

struct RenderBatch
{
    uint32_t begin = 0;
    uint32_t end = 0;
};

class RenderBatcher
{
public:
    static std::vector<RenderBatch> splitEvenly(uint32_t itemCount, uint32_t batchCount);
};

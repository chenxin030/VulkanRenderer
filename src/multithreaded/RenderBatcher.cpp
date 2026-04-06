#include "RenderBatcher.h"

std::vector<RenderBatch> RenderBatcher::splitEvenly(uint32_t itemCount, uint32_t batchCount)
{
    std::vector<RenderBatch> batches;
    if (itemCount == 0 || batchCount == 0)
    {
        return batches;
    }

    batchCount = (batchCount > itemCount) ? itemCount : batchCount;
    batches.reserve(batchCount);

    const uint32_t baseSize = itemCount / batchCount;
    const uint32_t remainder = itemCount % batchCount;

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < batchCount; ++i)
    {
        const uint32_t currentSize = baseSize + (i < remainder ? 1u : 0u);
        RenderBatch batch{};
        batch.begin = cursor;
        batch.end = cursor + currentSize;
        batches.push_back(batch);
        cursor = batch.end;
    }

    return batches;
}

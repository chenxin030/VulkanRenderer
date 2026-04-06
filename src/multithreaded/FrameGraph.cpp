#include "FrameGraph.h"

void FrameGraph::reset()
{
    passes_.clear();
}

void FrameGraph::addPass(const std::string& passName, bool enabled)
{
    passes_.push_back({ passName, enabled });
}

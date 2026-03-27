#pragma once

#include <Core/Renderer.h>

// Culling renderer wrapper (legacy core path, to be extracted later).
class CullingRenderer : public Renderer
{
public:
    CullingRenderer() = default;
    ~CullingRenderer() = default;
};


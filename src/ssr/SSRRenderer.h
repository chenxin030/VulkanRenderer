#pragma once

#include <Core/Renderer.h>

// SSR renderer wrapper (legacy core path, to be extracted later).
class SSRRenderer : public Renderer
{
public:
    SSRRenderer() = default;
    ~SSRRenderer() = default;
};


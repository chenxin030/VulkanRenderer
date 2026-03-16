# Instanced (Level 2)

This document describes the Level 2 instanced rendering path: render multiple instances in a single draw call and provide per-instance transforms via SSBO/UBO.

Back to root: [README.md](../../README.md)

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## Goals

- Render N instances using `vkCmdDrawIndexed(..., instanceCount=MAX_OBJECTS, ...)`
- Put camera matrices in a global UBO; put per-instance model matrices in an SSBO
- Use Dynamic Rendering (no classic RenderPass required)

## Runtime Flow

1. Create buffers
   - `GlobalUBO`: view/proj
   - `InstanceData[]`: an array of model matrices
2. Create descriptors
   - binding 0: global UBO
   - binding 1: instance SSBO
   - binding 2: texture sampler (the Level 2 demo uses a texture)
3. Per-frame update
   - update camera UBO
   - update instance SSBO (each instance's model)
4. Record commands
   - bind pipeline / descriptor set
   - `drawIndexed(indexCount, instanceCount, ...)`

## Data & Bindings

### C++

Main implementation files:

- `src/Core/Renderer_instanced.cpp`
- `src/Core/Renderer_rendering.cpp`
- `src/Core/Renderer.h`

Core data:

- `GlobalUBO`: view/proj
- `InstanceData`: model

### Shaders (Slang)

- `shaders/instanced.slang`

Conventions:

- The vertex shader uses `SV_InstanceID` to index the SSBO
- `globalUbo` provides view/proj
- `instanceBuffer[instanceID].model` provides model

## Troubleshooting

### Validation: Vertex attribute not consumed

If the pipeline's vertex attribute declarations (locations 0/1/2) don't match the shader inputs, validation will report "not consumed" warnings.

Recommendation:

- Provide dedicated vertex attribute layouts via `Vertex::getXXXAttributeDescriptions()` and avoid declaring unused attributes for a given pipeline.

See `Vertex` in `src/Core/Mesh.h`.

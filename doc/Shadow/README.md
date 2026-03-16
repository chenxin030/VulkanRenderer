# Shadow (Level 5)

This document describes the shadow-related rendering path:

- Back to root: [README.md](../../README.md)

- Chinese version: [README.zh-CN.md](README.zh-CN.md)

- **Level 5: Shadow Map (hard shadows)**

In this project, these techniques are implemented in one path and can be switched at runtime via UI:

- Hard / PCF / PCSS

> Note: Only `RENDERING_LEVEL == 5` is the shadow path. `RENDERING_LEVEL >= 6` is reserved for TAAU.

## Test Scene

The scene is designed to highlight how "near-ground vs off-ground" affects shadow quality:

- High pillar: a tall pillar with a cube floating high above the ground
- Low pillar: a very short pillar with a cube almost touching the ground
- Static cube: a cube placed directly on the ground

Transforms are configured in `src/Core/ResourceManager.h` under the `RENDERING_LEVEL == 5` branch.

## Light Model (3 lights)

The scene places one light of each type and allows adjusting intensity:

- Directional light (casts shadows)
- Point light (no shadows, basic attenuation)
- Area light (rectangular area light, approximated with 4 sample points)

Lighting data is packed in `ShadowUBO` (see `src/Core/ResourceManager.h`).

## Shadow Map Pass

### Goals

- Render depth from the light's view into a depth texture (shadow map)
- In the main pass, transform world position into light clip space to obtain shadow UV and receiver depth

### Key Points

- Shadow map image usage: `DepthAttachment | Sampled`
- After the shadow pass, transition the shadow map layout to `ShaderReadOnlyOptimal`

## PCF

### Core Idea

Replace a single comparison with multiple samples and average the visibility:

- Sample multiple offsets (e.g., Poisson disk)
- For each sample, compare `currentDepth - bias` against `closestDepth`
- Average to obtain visibility

### Parameters

- `pcfRadiusTexels`: filter radius (in texels)

## PCSS

### Core Idea

Two-stage approach:

1. **Blocker Search**
   - Sample a larger search radius and estimate average blocker depth (`avgBlockerDepth`)
2. **Penumbra Size**
   - Estimate penumbra size based on receiver depth and blocker depth (farther ?? softer)
3. **Filtering**
   - Perform PCF using the estimated filter radius

### Parameters

- `pcssLightSizeTexels`: baseline "light size" (in texels) used for blocker search and filter radius estimation

## UI (Runtime Switching & Tuning)

A lightweight UI panel provides runtime switching and tuning:

- Shadow Filter: Hard / PCF / PCSS
- PCF Radius (texels)
- PCSS Light Size (texels)
- Three light intensities (Dir / Point / Area)

UI implementation lives in `src/Core/Renderer_instanced.cpp`:

- `initUI()`
- `updateUIFrame()`
- `recordUI()`

UI shader: `shaders/imgui.slang`

## Shader Binding Contract

See `shaders/shadow_lit.slang` (explicit `[[vk::binding(x, 0)]]`):

- binding 0: `SceneUBO`
- binding 1: `InstanceData[]`
- binding 2: `ShadowUBO`
- binding 3: `shadowMap` (Sampler2D)
- binding 4: `ShadowParamsUBO` (filter mode and parameters)

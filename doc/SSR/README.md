# SSR (Screen-Space Reflections)
Back to root: [README.md](../../README.md)

Chinese version: [README.zh-CN.md](README.zh-CN.md)

This renderer implements a lightweight SSR pass for `RENDERING_LEVEL == 7`. The goal is to provide view-dependent reflections using the current frame’s depth and color without requiring extra reflection probes or ray tracing.

## Pipeline Overview

1. **Lighting pass** renders into the swapchain color image (opaque pass).
2. **SSR input capture** copies the swapchain color into an offscreen `ssrColor` texture.
3. **SSR composite** performs ray marching in view space using depth, then blends reflections back onto the swapchain image.
4. **UI pass** is rendered after SSR so it is not reflected.

## Data Inputs

- **Depth**: current frame depth buffer (sampled as `DEPTH_READ_ONLY_OPTIMAL`).
- **Color**: captured swapchain color (sampled as `SHADER_READ_ONLY_OPTIMAL`).
- **Scene UBO**: projection, view, inverse projection, camera position, near/far planes.
- **SSR Params**: ray distance, thickness, stride, max steps, intensity.

## Algorithm Highlights

- **View-space reconstruction** uses the inverse projection matrix.
- **Ray marching** in view space, followed by a short binary search refinement.
- **Edge fade** to reduce artifacts near screen borders.

## Key Files

- `src/Core/Renderer_SSR.cpp`
- `shaders/ssr.slang`

## Notes

SSR is inherently limited by screen visibility. Off-screen or occluded geometry will not appear in reflections.

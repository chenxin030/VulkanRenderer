# TAAU (Level 6)

Back to docs index: [README.md](../../README.md)

This level provides a **TAAU test path** built on top of the existing shadow rendering pipeline. It focuses on exposing typical temporal accumulation weaknesses so you can iterate on TAAU heuristics without changing the whole renderer.

## Goals (Weakness Coverage)

- **Ghosting**: history blending can leave trails on moving objects.
- **Thin-line shimmer**: sub-pixel geometry tends to flicker when reprojected.
- **Fast motion**: large motion vectors cause history rejection and stability issues.
- **Screen edge / high-frequency textures**: edge disocclusion and dense patterns produce jittered instability.

## Entry Points

- `src/Core/Renderer_TAAU.cpp` (main TAAU logic)
- `src/Core/Renderer.h` (`RENDERING_LEVEL == 6` switch)
- `src/Core/ResourceManager.h` (test scene transforms)

## Runtime Flow

1. **Resource initialization**
   - Uses the same shadow descriptor set layout and pipelines as Level 5.
   - `createShadowMapResources()` is executed before `createShadowDescriptorSets()` so the shadow map image view is valid.

2. **Frame update**
   - `updateShadowBuffers()` (Level 6 override) updates:
     - camera UBO
     - light UBO
     - shadow params UBO
     - per-instance transforms and colors
   - `updateTAAUScene()` animates stress-test objects.

3. **UI**
   - `updateUIFrame()` calls `ImGui::NewFrame()` then `updateTAAUUI()`.
   - `updateTAAUUI()` exposes TAAU debug controls in a separate window.

## Test Scene Layout

Defined in `ResourceManager.h` under `RENDERING_LEVEL == 6`:

- Ground plane (edge shimmer)
- Thin pillars (thin-line flicker)
- Fast moving probe
- Edge-aligned probe
- Dense bar array (high-frequency stress)

## UI Controls

The `TAAU Debug` window provides:

- **BlendFactor**: history mix weight (ghosting sensitivity)
- **ReactiveClamp**: clamp for high-frequency and disocclusion
- **AntiFlicker**: stabilizer for fine details
- **VelocityScale**: stress fast motion
- **FreezeHistory**: stop history updates to inspect trails

## Notes

This level currently **reuses the shadow pipelines** and does not yet implement a full temporal resolve shader. It is intended as a scaffolding path where you can add velocity buffers, jitter, history textures, and resolve passes progressively.

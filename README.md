# VulkanRenderer

A small Vulkan renderer project (C++ / Vulkan-Hpp RAII) that demonstrates multiple rendering paths controlled by `RENDERING_LEVEL`.

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## Rendering Levels

`RENDERING_LEVEL` is defined in `src/Core/Renderer.h`.

- **Level 1 (Multi-draw)**: basic multi-draw rendering (textured mesh).
- **Level 2 (Instanced)**: instanced rendering (per-instance transform data).
- **Level 3 (PBR Instanced)**: instanced PBR shading (direct lighting).
- **Level 4 (IBL PBR + Skybox)**: HDR IBL precomputation + IBL PBR + skybox.
- **Level 5 (Shadow / PCF / PCSS)**: shadow map + PCF + PCSS + adjustable lights + runtime UI (current default).

## Quick Start (Windows / Visual Studio)

### Prerequisites

- Vulkan SDK (loader / validation layers / tools)
- CMake >= 3.28
- Visual Studio 2022 (MSVC)

### Configure & Build

From the repository root:

```powershell
cmake -S . -B build_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Debug -j
```

Executable:

`build_x64/Debug/vulkanRenderer.exe`

## Asset & Path Conventions

Paths are compiled in via preprocessor definitions in `CMakeLists.txt`:

- `VK_MODEL_DIR = <repo>/assets/models/`
- `VK_TEXTURE_DIR = <repo>/assets/textures/`
- `VK_SHADERS_DIR = <repo>/shaders/`

## Shaders

All `.slang` files under `shaders/` are compiled to SPIR-V during the build and copied to `shaders/*.spv`:

- Source: `shaders/*.slang`
- Output: `shaders/*.spv`

## Docs

- Instanced rendering (Level 2): [doc/Instanced/README.md](doc/Instanced/README.md)
- PBR / IBL (Level 3/4): [doc/PBR/README.md](doc/PBR/README.md)
- Shadow / PCF / PCSS / lights & UI (Level 5/6/7): [doc/Shadow/README.md](doc/Shadow/README.md)

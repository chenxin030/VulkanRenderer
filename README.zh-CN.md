# VulkanRenderer

一个使用 C++ / Vulkan-Hpp RAII 编写的小型 Vulkan 渲染器项目，通过 `RENDERING_LEVEL` 切换不同的渲染路径。

English version: [README.md](README.md)

## 渲染级别

`RENDERING_LEVEL` 在 `src/Core/Renderer.h` 中定义。

- **Level 1 (Multi-draw)**：基础多物体绘制（纹理/网格）。
- **Level 2 (Instanced)**：实例化渲染（每实例变换数据）。
- **Level 3 (PBR Instanced)**：实例化 PBR（直接光照）。
- **Level 4 (IBL PBR + Skybox)**：HDR IBL 预计算 + IBL PBR + 天空盒。
- **Level 5 (Shadow / PCF / PCSS)**：ShadowMap + PCF + PCSS + 可调光源与运行时 UI（当前默认）。
- **Level 6 (TAAU)**：TAAU 测试路径（拖影 / 细线闪烁 / 快速运动 / 边缘与高频压力测试）。

## 快速开始（Windows / Visual Studio）

### 前提条件

- Vulkan SDK（loader / validation layers / tools）
- CMake >= 3.28
- Visual Studio 2022（MSVC）

### 配置与构建

在仓库根目录执行：

```powershell
cmake -S . -B build_x64 -G "Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Debug -j
```

可执行文件：

`build_x64/Debug/vulkanRenderer.exe`

## 资源与路径约定

路径通过 `CMakeLists.txt` 中的预处理器定义编译到代码中：

- `VK_MODEL_DIR = <repo>/assets/models/`
- `VK_TEXTURE_DIR = <repo>/assets/textures/`
- `VK_SHADERS_DIR = <repo>/shaders/`

## 着色器

`shaders/` 下所有 `.slang` 文件在构建时会被编译为 SPIR-V，并复制到 `shaders/*.spv`：

- 源文件：`shaders/*.slang`
- 输出：`shaders/*.spv`

## 文档

- 实例化渲染（Level 2）：[doc/Instanced/README.zh-CN.md](doc/Instanced/README.zh-CN.md)
- PBR / IBL（Level 3/4）：[doc/PBR/README.zh-CN.md](doc/PBR/README.zh-CN.md)
- 阴影 / PCF / PCSS / 光源与 UI（Level 5）：[doc/Shadow/README.zh-CN.md](doc/Shadow/README.zh-CN.md)
- TAAU（Level 6）：[doc/TAAU/README.zh-CN.md](doc/TAAU/README.zh-CN.md)

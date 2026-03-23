# VulkanRenderer

一个vulkan新手使用 C++ / Vulkan-Hpp RAII 编写的小型 Vulkan 渲染器项目，实现几个常见功能，功能之间独立，通过 `RENDERING_LEVEL` 切换不同的功能。


## 渲染级别

`RENDERING_LEVEL` 在 `src/Core/Renderer.h` 中定义。

- **Level 1 (Multi-draw)**：基础的绘制3个物体（带纹理 & 网格）。
- **Level 2 (Instanced)**：把Level 1改成实例化渲染（不同实例有不同的transform）。
- **Level 3 (PBR Instanced)**：实例化 PBR（直接光照）。
- **Level 4 (IBL PBR + Skybox)**：HDR IBL 预计算 + IBL PBR + 天空盒。
- **Level 5 (Shadow / PCF / PCSS)**：ShadowMap + PCF + PCSS + 可调光源与运行时 UI（当前默认）。
- **Level 6 (TAAU)**：TAAU （拖影 / 细线闪烁 / 快速运动 / 边缘与高频压力测试）。
- **Level 7（SSR）**：屏幕空间反射

## 构建（Windows / Visual Studio）

### 前提条件

- Vulkan SDK（loader / validation layers / tools）
- CMake >= 3.28
- Visual Studio 2022（MSVC）

### 配置与构建

在仓库根目录执行：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug -j
```

可执行文件：

`build/Debug/vulkanRenderer.exe`

## 资源与路径约定

资源路径通过 `CMakeLists.txt` 中的预处理器定义编译到代码中：

- `VK_MODEL_DIR = <repo>/assets/models/`
- `VK_TEXTURE_DIR = <repo>/assets/textures/`
- `VK_SHADERS_DIR = <repo>/shaders/`

## 着色器

文件都在`shaders/` 下，所有 `.slang` 文件在构建时会被编译为 SPIR-V，并复制到 `shaders/*.spv`：

- 源文件：`shaders/*.slang`
- 输出：`shaders/*.spv`

## 文档

- [Level 1：基础渲染](doc/BasicRender/README.zh-CN.md)
- [Level 2：实例化渲染](doc/Instanced/README.zh-CN.md)
- [Level 3/4：PBR / IBL](doc/PBR/README.zh-CN.md)
- [Level 5：阴影 / PCF / PCSS / 光源与 UI](doc/Shadow/README.zh-CN.md)
- [Level 6：TAAU](doc/TAAU/README.zh-CN.md)
- [Level 7：SSR](doc/SSR/README.zh-CN.md)
- [ECS 概述](doc/ECS/README.zh-CN.md)


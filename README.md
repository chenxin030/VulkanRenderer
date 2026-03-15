# VulkanRenderer

A small Vulkan renderer project (C++ / Vulkan-Hpp RAII) that demonstrates multiple rendering paths controlled by `RENDERING_LEVEL`.

## Rendering Levels

`RENDERING_LEVEL` is defined in `Renderer.h`.

- **Level 1 (Multi-draw)**: Basic multi-draw rendering path (textured model/mesh rendering).
- **Level 2 (Instanced)**: Instanced rendering path (many instances with per-instance transform data).
- **Level 3 (PBR Instanced)**: Instanced PBR shading (direct lights, instancing).
- **Level 4 (IBL PBR + Skybox)**: HDR environment map based IBL precomputation + PBR shading with IBL + skybox rendering.

## Quick Start (Windows / Visual Studio)

### Prerequisites

- **Vulkan SDK** installed (loader, validation layers, and tools).
- **CMake** (>= 3.28).
- **Visual Studio 2022** with MSVC toolchain.

### Configure & Build

From the repository root:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug -j
```

The executable will be at:

`build/Debug/vulkanRenderer.exe`

## Asset & Path Conventions

Paths are compiled in via preprocessor definitions in `CMakeLists.txt`:

- `VK_MODEL_DIR = <repo>/assets/models/`
- `VK_TEXTURE_DIR = <repo>/assets/textures/`
- `VK_SHADERS_DIR = <repo>/shaders/`

### Required assets for Level 4

- HDR equirectangular environment:  
  `assets/textures/newport_loft.hdr`

## Shaders

All `.slang` files under `shaders/` are compiled to SPIR-V during the build and copied to `shaders/*.spv`:

- Source: `shaders/*.slang`
- Output: `shaders/*.spv`

Level 4 expects (at least) the following SPIR-V outputs to exist:

- `filtercube.spv`
- `irradiancecube.spv`
- `prefilterenvmap.spv`
- `genbrdflut.spv`
- `pbribl.spv`
- `skybox.spv`

## Level 4: IBL PBR + Skybox 

Level 4 renders a grid of **instanced spheres** with PBR shading, lit by:

- **Direct lighting** (a few point lights)
- **Image Based Lighting (IBL)** from an HDR environment map:
  - Diffuse IBL via irradiance cubemap
  - Specular IBL via prefiltered environment cubemap + BRDF LUT

It also renders a **skybox** using the same environment map.

### High-level runtime flow

1. **Load resources**
   - Load sphere mesh.
   - Load HDR `newport_loft.hdr` as a 2D float texture.
2. **Generate IBL resources** `generateIBLResources`
   - Equirectangular (2D HDR) → **environment cubemap**
   - Environment cubemap → **irradiance cubemap** (diffuse convolution)
   - Environment cubemap → **prefiltered cubemap** (specular prefilter, mip chain)
   - Generate **BRDF LUT** (2D)
3. **Create descriptor set layouts / pools / sets**
4. **Create pipelines**
   - `pbribl` graphics pipeline (instanced spheres)
   - `skybox` graphics pipeline
5. **Per-frame update**
   - Update camera UBO, lights UBO, instance SSBO, tone-mapping parameters.
6. **Record & submit**
   - Draw skybox
   - Draw instanced spheres

### IBL precomputation details

IBL generation uses dynamic rendering and renders into offscreen color attachments:

- `envCubemapData` (cube-compatible image, 6 layers)
- `irradianceCubemapData` (cube-compatible image, 6 layers)
- `prefilteredEnvMapData` (cube-compatible image, 6 layers, mip chain)
- `brdfLutData` (2D image)

Each cubemap face is rendered by creating a temporary 2D view into one layer (and mip level where applicable), then calling `beginRendering()` / `endRendering()`.

### Descriptor bindings (PBR IBL)

See `createIBLPBRDescriptorSetLayout` and shader bindings in `pbribl.slang`.

- **binding 0**: `SceneUBO` (projection/view/camPos)
- **binding 1**: `PBRInstanceData[]` SSBO (model/metallic/roughness/color)
- **binding 2**: `LightUBO` (4 point lights)
- **binding 3**: `Irradiance` cubemap (diffuse IBL)
- **binding 4**: `Prefiltered` cubemap (specular IBL)
- **binding 5**: `BRDF LUT` (2D)
- **binding 6**: `ParamsUBO` (exposure/gamma)

### Descriptor bindings (Skybox)

See `createSkyboxDescriptorSetLayout` and shader bindings in `skybox.slang`.

- **binding 0**: `SkyboxUBO` (`invProjection`, `invView`)
- **binding 1**: `ParamsUBO` (exposure/gamma)
- **binding 2**: Environment cubemap sampler (currently uses the prefiltered cubemap)

---
# VulkanRenderer

一个使用 C++ / Vulkan-Hpp RAII 编写的小型 Vulkan 渲染器项目，通过 `RENDERING_LEVEL` 控制多种渲染路径。

## 渲染级别

`RENDERING_LEVEL` 在 `Renderer.h` 中定义。

- **级别 1 (多物体绘制)**：基础的多物体绘制渲染路径（带纹理的模型/网格渲染）。
- **级别 2 (实例化)**：实例化渲染路径（许多实例，每个实例包含独立的变换数据）。
- **级别 3 (PBR 实例化)**：实例化 PBR 着色（直接光源，实例化）。
- **级别 4 (IBL PBR + 天空盒)**：基于 HDR 环境贴图的 IBL 预计算 + 带 IBL 的 PBR 着色 + 天空盒渲染。

## 快速开始 (Windows / Visual Studio)

### 前提条件

- 已安装 **Vulkan SDK**（包括加载器、验证层和工具）。
- **CMake** (>= 3.28)。
- **Visual Studio 2022** 及 MSVC 工具链。

### 配置与构建

在仓库根目录下执行：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug -j
```

可执行文件位于：

`build/Debug/vulkanRenderer.exe`

## 资源与路径约定

路径通过 `CMakeLists.txt` 中的预处理器定义编译到代码中：

- `VK_MODEL_DIR = <仓库路径>/assets/models/`
- `VK_TEXTURE_DIR = <仓库路径>/assets/textures/`
- `VK_SHADERS_DIR = <仓库路径>/shaders/`

### Level 4 所需的资源

- HDR 等距柱状图环境贴图：
  `assets/textures/newport_loft.hdr`

## 着色器

`shaders/` 目录下所有 `.slang` 文件在构建过程中会被编译为 SPIR-V，并复制到 `shaders/*.spv`：

- 源文件：`shaders/*.slang`
- 输出文件：`shaders/*.spv`

Level 4 需要（至少）存在以下 SPIR-V 输出文件：

- `filtercube.spv`
- `irradiancecube.spv`
- `prefilterenvmap.spv`
- `genbrdflut.spv`
- `pbribl.spv`
- `skybox.spv`

## Level 4：IBL PBR + 天空盒

Level 4 渲染一个**实例化的球体网格**，使用 PBR 着色，光照来源包括：

- **直接光照**（数个点光源）
- 来自 HDR 环境贴图的**基于图像的光照 (IBL)**：
  - 漫反射 IBL 通过辐照度立方体贴图实现
  - 镜面反射 IBL 通过预过滤的环境立方体贴图 + BRDF 查找表实现

同时，它使用相同的环境贴图渲染一个**天空盒**。

### 高级运行时流程

1.  **加载资源**
    - 加载球体网格。
    - 将 HDR 文件 `newport_loft.hdr` 作为 2D 浮点纹理加载。
2.  **生成 IBL 资源** (generateIBLResources)
    - 等距柱状图 (2D HDR) → **环境立方体贴图**
    - 环境立方体贴图 → **辐照度立方体贴图**（漫反射卷积）
    - 环境立方体贴图 → **预过滤立方体贴图**（镜面反射预过滤，包含 mip 链）
    - 生成 **BRDF 查找表** (2D)
3.  **创建描述符集布局 / 池 / 集**
4.  **创建管线**
    - `pbribl` 图形管线（实例化球体）
    - `skybox` 图形管线
5.  **每帧更新**
    - 更新相机 UBO、光源 UBO、实例 SSBO、色调映射参数。
6.  **录制与提交**
    - 绘制天空盒
    - 绘制实例化球体

### IBL 预计算细节

IBL 生成使用动态渲染，渲染到离屏颜色附件中：

- `envCubemapData`（CUBE_COMPATIBLE的图像，6 层）
- `irradianceCubemapData`（CUBE_COMPATIBLE的图像，6 层）
- `prefilteredEnvMapData`（CUBE_COMPATIBLE的图像，6 层，包含 mipmap）
- `brdfLutData`（2D 图像）

每个立方体贴图面都是通过创建一个指向单层（以及适用的 mip 级别）的临时 2D 视图，然后调用 `beginRendering()` / `endRendering()` 来渲染的。

### 描述符绑定 (PBR IBL)

参见 `createIBLPBRDescriptorSetLayout` 和 `pbribl.slang` 着色器中的绑定。

- **binding 0**：`SceneUBO` (投影/视图/相机位置)
- **binding 1**：`PBRInstanceData[]` SSBO (模型矩阵/金属度/粗糙度/颜色)
- **binding 2**：`LightUBO` (4 个点光源)
- **binding 3**：`Irradiance` 立方体贴图 (漫反射 IBL)
- **binding 4**：`Prefiltered` 立方体贴图 (镜面反射 IBL)
- **binding 5**：`BRDF LUT` (2D)
- **binding 6**：`ParamsUBO` (曝光/伽马值)

### 描述符绑定 (天空盒)

参见 `createSkyboxDescriptorSetLayout` 和 `skybox.slang` 着色器中的绑定。

- **binding 0**：`SkyboxUBO` (`invProjection`, `invView`)
- **binding 1**：`ParamsUBO` (曝光/伽马值)
- **binding 2**：环境立方体贴图采样器（当前使用的是预过滤立方体贴图）

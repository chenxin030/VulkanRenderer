# VulkanRenderer

基于 Vulkan-Hpp RAII API 构建的模块化渲染器，采用 Slang 作为 Shader 语言，CMake 构建，C++23 标准。

各模块彼此独立，编译为独立可执行文件。

---

## 子项目一览

| 编号 | 模块 | 源码目录 | Shader 目录 | 说明 |
| :---: | ---- | -------- | ----------- | ---- |
| 1 | **Instanced** | `src/Instanced/` | `shaders/1_Instanced/` | 多物体实例化渲染，不同实例携带独立 Transform |
| 2 | **PBR** | `src/pbr/` | `shaders/2_pbr/` | 直接光照 PBR 材质渲染 |
| 3 | **IBL PBR** | `src/IBL_pbr/` | `shaders/3_pbr_ibl/` | HDR IBL 预计算（辐照度图 + 预过滤环境图 + BRDF LUT）+ 天空盒 |
| 4 | **Shadow** | `src/shadow/` | `shaders/4_shadow/` | 阴影映射 + PCF + PCSS + 运行时光源与 UI 控制 |
| 5 | **TAAU** | `src/taau/` | `shaders/5_taau/` | Temporal Anti-Aliasing Upsampling（时序抗锯齿上采样） |
| 6 | **SSR** | `src/ssr/` | `shaders/6_ssr/` | 屏幕空间反射（Screen Space Reflections） |
| 7 | **Culling** | `src/culling/` | `shaders/7_culling/` | GPU-Driven 可见性剔除：Frustum Culling + Hi-Z Occlusion Culling + 间接绘制 |
| 8 | **Deferred Shading** | `src/deferredShading/` | `shaders/8_deferredShading/` | 延迟渲染：GBuffer 填充 + 光照 Pass |
| 9 | **PostFX** | `src/postfx/` | `shaders/9_postfx/` | 后处理：Bloom (Extract + Blur + Composite) + 色调映射 |
| 10 | **GI** | `src/gi/` | `shaders/10_gi/` | 环境光照：SSAO（Screen Space Ambient Occlusion） |
| 11 | **Clustered** | `src/clustered/` | `shaders/11_clustered/` | Clustered Shading：2048 动态点光源，Compute Shader 预构建 Cluster Grid |
| 12 | **Particles** | `src/particles/` | `shaders/12_particles/` | GPU 粒子系统：Compute Shader 更新粒子状态，Vertex Shader 渲染 |
| 13 | **Multithreaded** | `src/multithreaded/` | — | 多线程渲染框架：Frame Graph + Render Batcher + 线程池 |

---

## 公共基础设施

- **`Base`**（静态库，`vkr_base`）：所有 Renderer 必须链接的核心库
  - `VulkanBase` — Vulkan 设备初始化、SwapChain、命令缓冲、同步对象
  - `VulkanTypes` — Buffer / Texture / UBO 数据结构（`MeshBuffer`、`TextureData`、`ShadowUBO` 等）
  - `Mesh` — glTF 模型加载、网格数据结构
  - `Camera` — WASD + Q/E 自由相机控制

---

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

每个子项目编译为独立可执行文件，输出目录：

```
build/Debug/
├── 1_Instanced.exe
├── 2_pbr.exe
├── 3_pbr_ibl.exe
├── 4_shadow.exe
├── 5_taau.exe
├── 6_ssr.exe
├── 7_culling.exe
├── 8_deferredShading.exe
├── 9_postfx.exe
├── 10_gi.exe
├── 11_clustered.exe
├── 12_particles.exe
├── 13_multithreaded.exe
└── glfw3.dll
```

---

## 资源与路径约定

资源路径通过 CMake 预处理器定义注入：

| 宏 | 展开路径 |
| -- | -------- |
| `VK_MODEL_DIR` | `assets/models/` |
| `VK_TEXTURE_DIR` | `assets/textures/` |
| `VK_SHADERS_DIR` | `shaders/` |

---

## Shader 开发

所有 Shader 使用 Slang 语言编写（`.slang`），构建时自动编译为 SPIR-V（`.spv`）：

- 源文件：`shaders/<编号>_<名称>/*.slang`
- 输出：`shaders/*.spv`（同时复制到各可执行文件输出目录）
- 入口函数约定：`vertMain`（顶点）/ `fragMain`（片元）/ `compMain`（计算）
- Slang 编译器：`external/slang/bin/slangc.exe`（需提前下载）

---

## 文档

| 章节 | 文档 |
| ---- | ---- |
| Level 0：基础渲染 | [doc/BasicRender/README.md](doc/BasicRender/README.md) |
| Level 1：Instanced | [doc/01_Instanced/README.md](doc/01_Instanced/README.md) |
| Level 2：PBR | [doc/02_PBR/README.md](doc/02_PBR/README.md) |
| Level 3：IBL PBR | [doc/02_PBR/README.md](doc/02_PBR/README.md) |
| Level 4：Shadow | [doc/04_Shadow/README.md](doc/04_Shadow/README.md) |
| Level 5：TAAU | [doc/05_TAAU/README.md](doc/05_TAAU/README.md) |
| Level 6：SSR | [doc/06_SSR/README.md](doc/06_SSR/README.md) |
| Level 7：Culling | [doc/07_Culling/README.md](doc/07_Culling/README.md) |
| Level 8：Deferred Shading | [doc/08_Deferred/README.md](doc/08_Deferred/README.md) |
| Level 9：PostFX | [doc/09_PostFX/README.md](doc/09_PostFX/README.md) |
| Level 10：GI | [doc/10_gi/README.md](doc/10_gi/README.md) |
| Level 11：Clustered | [doc/11_clustered/README.md](doc/11_clustered/README.md) |
| Level 12：Particles | [doc/12_particles/README.md](doc/12_particles/README.md) |
| Level 13：Multithreaded | [doc/13_multithreaded/README.md](doc/13_multithreaded/README.md) |

# Shadow（Level 5/6/7）

English version: [README.md](README.md)

本文描述阴影相关渲染路径：

- **Level 5：Shadow Map（硬阴影）**
- **Level 6：PCF（Percentage Closer Filtering）**
- **Level 7：PCSS（Percentage Closer Soft Shadows）**

当前项目把这些能力整合在同一条路径中，并通过 UI 运行时切换：

- Hard / PCF / PCSS

## 测试场景

场景用于突出“贴地/离地”对阴影质量的影响：

- 高支柱：柱子 + 顶部悬空立方体（离地较远）
- 低支柱：极矮柱子 + 顶部立方体（几乎贴地）
- 静态立方体：直接放在地面

Transforms 配置在 `src/Core/ResourceManager.h` 的 `RENDERING_LEVEL == 5` 分支。

## 光源模型（3 种光源）

场景中放置 3 种光源各 1 个，并支持调节强度：

- Directional light（带阴影）
- Point light（无阴影，基础衰减）
- Area light（矩形面积光，简化为 4 个采样点叠加）

光照数据打包在 `ShadowUBO`（见 `src/Core/ResourceManager.h`）。

## Shadow Map Pass

### 目标

- 从光源视角渲染深度到一张深度贴图（ShadowMap）
- 主渲染 pass 中把世界坐标变换到 light clip space，得到 shadow UV 与 receiver depth

### 关键点

- ShadowMap image usage：`DepthAttachment | Sampled`
- shadow pass 结束后将 shadow map layout 转为 `ShaderReadOnlyOptimal`

## PCF（Level 6）

### 核心思路

将一次比较扩展为多次采样求平均：

- 采样多个 offset（例如 Poisson disk）
- 对每个 sample 做 `currentDepth - bias` 与 `closestDepth` 比较
- 平均得到 visibility

### 参数

- `pcfRadiusTexels`：滤波半径（texel）

## PCSS（Level 7）

### 核心思路

两段式：

1. **Blocker Search**
   - 以更大搜索半径采样，估计平均 blocker depth（`avgBlockerDepth`）
2. **Penumbra Size**
   - 根据 receiver depth 与 blocker depth 估算半影大小（越远越软）
3. **Filtering**
   - 用估算出的半径做一次 PCF

### 参数

- `pcssLightSizeTexels`：基准“光源尺寸”（texel），用于 blocker search 与滤波半径估算

## UI（运行时切换与调参）

UI 面板提供运行时切换与调参：

- Shadow Filter：Hard / PCF / PCSS
- PCF Radius（texels）
- PCSS Light Size（texels）
- 三种光源强度（Dir / Point / Area）

UI 实现位于 `src/Core/Renderer_instanced.cpp`：

- `initUI()`
- `updateUIFrame()`
- `recordUI()`

UI shader：`shaders/imgui.slang`

## Shader 绑定约定（Level 5/6/7）

以 `shaders/shadow_lit.slang` 为准（显式 `[[vk::binding(x, 0)]]`）：

- binding 0：`SceneUBO`
- binding 1：`InstanceData[]`
- binding 2：`ShadowUBO`
- binding 3：`shadowMap`（Sampler2D）
- binding 4：`ShadowParamsUBO`（滤波模式与参数）

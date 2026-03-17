# Shadow（Level 5）

返回目录：[README.zh-CN.md](../../README.zh-CN.md)

英文版本：[README.md](README.md)

本章节描述阴影渲染路径：

- **Level 5：Shadow Map（硬阴影）**

本项目中这些阴影策略统一在一条路径中，并可通过 UI 运行时切换：

- Hard / PCF / PCSS

> 注意：只有 `RENDERING_LEVEL == 5` 是 Shadow 路径。`RENDERING_LEVEL >= 6` 用于 TAAU。

## 测试场景

场景用于对比“贴地 vs 离地”对阴影质量的影响：

- 高柱：立柱上方有高悬浮方块
- 低柱：立柱高度极低，方块几乎贴地
- 静态方块：直接放置在地面上

Transforms 在 `src/Core/ResourceManager.h` 的 `RENDERING_LEVEL == 5` 分支中配置。

## 光照模型（3 种光源）

场景包含三种光源，并可调节强度：

- Directional light（投射阴影）
- Point light（不投射阴影，含距离衰减）
- Area light（矩形面积光，使用 4 个采样点近似）

光照数据打包在 `ShadowUBO` 中（见 `src/Core/ResourceManager.h`）。

## Shadow Map Pass

### 目标

- 从光源视角渲染深度图（shadow map）
- 主渲染 pass 中将世界坐标变换到 light clip space，得到 shadow UV 与 receiver depth

### 关键点

- Shadow map image usage：`DepthAttachment | Sampled`
- Shadow pass 结束后将 shadow map layout 切换为 `ShaderReadOnlyOptimal`

## PCF

### 核心思路

将单次比较扩展为多点采样并求平均：

- 采样多个 offset（例如 Poisson disk）
- 对每个 sample 比较 `currentDepth - bias` 与 `closestDepth`
- 取平均得到可见度

### 参数

- `pcfRadiusTexels`：滤波半径（texel）

## PCSS

### 核心思路

两阶段：

1. **Blocker Search**
   - 使用更大的搜索半径采样，估算 blocker depth 平均值 `avgBlockerDepth`
2. **Penumbra Size**
   - 根据 receiver depth 与 blocker depth 估算阴影软化范围
3. **Filtering**
   - 使用估算出的半径执行 PCF

### 参数

- `pcssLightSizeTexels`：基准“光源大小”（texel），用于 blocker 搜索与滤波半径估算

## UI（运行时切换与调参）

UI 面板支持运行时切换与调参：

- Shadow Filter：Hard / PCF / PCSS
- PCF Radius（texels）
- PCSS Light Size（texels）
- 三种光源强度（Dir / Point / Area）

UI 实现位置：`src/Core/Renderer_Shadow.cpp`

- `initUI()`
- `updateUIFrame()`
- `recordUI()`

UI shader：`shaders/imgui.slang`

## Shader 绑定约定（Level 5）

见 `shaders/shadow_lit.slang`（显式 `[[vk::binding(x, 0)]]`）：

- binding 0：`SceneUBO`
- binding 1：`InstanceData[]`
- binding 2：`ShadowUBO`
- binding 3：`shadowMap`（Sampler2D）
- binding 4：`ShadowParamsUBO`（滤波模式与参数）

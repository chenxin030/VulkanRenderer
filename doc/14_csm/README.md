# 14_csm — Cascaded Shadow Maps（级联阴影映射）

[返回目录](../../README.md)

## 1. 项目目标

实现 **CSM (Cascaded Shadow Maps)** 级联阴影映射系统，通过将相机视锥分割为多个级联，为每个级联渲染独立分辨率的阴影贴图，解决大场景中近处阴影精度不足和远处阴影浪费的问题。

### 技术目标
- **视锥分割**：Practical Split Scheme（对数/线性混合），λ 可调
- **4 级联纹理数组**：单个 `VK_IMAGE_TYPE_2D` 纹理数组，每层 2048×2048 D32
- **逐级联光源 VP 计算**：子视锥 AABB → 光源空间正交投影
- **多级过滤模式**：Hard Shadow / PCF / PCSS 三种模式
- **级联可视化**：不同颜色标识各 cascad 覆盖区域

### 求职定位

> "实现了基于纹理数组的级联阴影映射 (CSM)，包含 Practical Split Scheme 视锥分割、逐级联光源 VP 矩阵计算、Hard/PCF/PCSS 三种阴影过滤模式，有效解决大场景阴影精度与性能的平衡问题。"

---

## 2. 核心技术原理

### 2.1 渲染管线

```
┌──────────────────────────────────────────────────────┐
│              CsmRenderer::recordCommandBuffer()        │
│                                                       │
│  1. [CSM Depth Pass ×4] → csmDepthPipeline            │
│     for each cascade (0..3):                          │
│       pushConstants(cascadeIndex)                     │
│       beginRendering(csmLayerView[cascadeIndex])      │
│       draw scene (depth only)                         │
│     → csmTextureArray (texture2DArray, 4 layers)      │
│                                                       │
│  2. transition → ShaderReadOnlyOptimal                │
│                                                       │
│  3. [Lit Pass] → csmLitPipeline                       │
│     depth comparison → select cascade                 │
│     sample shadowMapArray[cascadeIdx]                 │
│     Hard / PCF / PCSS filtering                       │
│                                                       │
│  4. [UI Pass] → imgui pipeline                        │
└──────────────────────────────────────────────────────┘
```

### 2.2 视锥分割 — Practical Split Scheme

$$split_i = \lambda \cdot near \cdot \left(\frac{far}{near}\right)^{i/N} + (1-\lambda) \cdot \left(near + (far-near) \cdot \frac{i}{N}\right)$$

- $\lambda = 0$：纯线性分割
- $\lambda = 1$：纯对数分割
- $\lambda = 0.75$：推荐值，兼顾近处精度和远处覆盖

```
Camera Near                                                       Camera Far
  |-------- Cascade 0 --------|----- Cascade 1 -----|-- C2 --|- C3 -|
  0                     split1              split2      split3   far
```

### 2.3 光源 VP 矩阵计算

对每个级联：

1. 提取子视锥 8 个角点（view space）
2. 变换到世界空间
3. 变换到光源空间（light view）
4. 计算 AABB 包围盒
5. 用 AABB 构建正交投影矩阵

```
Camera Frustum          Light Space
    ┌──────┐              ┌──────────┐
   ╱        ╲             │ AABB     │
  ╱  sub-    ╲   ──▶     │  ┌──┐   │
 ╱   frustum  ╲           │  │  │   │
╱              ╲          │  └──┘   │
─────────────────         └──────────┘
```

### 2.4 级联选择（Fragment Shader）

```glsl
// 根据当前片段在相机空间的深度选择级联
uint cascadeIdx = 0;
for (int i = 0; i < 3; i++) {
    if (viewDepth > cascadeSplitDepths[i])
        cascadeIdx = uint(i + 1);
}

// 用对应级联的 VP 计算 shadow position
float4 shadowPos = mul(csmUbo.cascadeViewProj[cascadeIdx], worldPos);

// 采样对应层级
float mapDepth = shadowMapArray.SampleLevel(sampler, float3(uv, cascadeIdx), 0).r;
```

---

## 3. 资源与 Descriptor Set

### Descriptor Set Layout（`csmDescriptorSetLayout`）

| binding | 类型                   | 用途                           | Stage       |
| ------- | ---------------------- | ------------------------------ | ----------- |
| 0       | `UniformBuffer`        | `SceneUBO`（相机 VP）          | Vert + Frag |
| 1       | `StorageBuffer`        | `InstanceData[]`（实例列表）   | Vert + Frag |
| 2       | `UniformBuffer`        | `CsmUBO`（4 个 VP + 分割深度） | Vert + Frag |
| 3       | `CombinedImageSampler` | `shadowMapArray`（纹理数组）   | Frag        |
| 4       | `UniformBuffer`        | `ShadowParamsUBO`（过滤参数）  | Frag        |

### CSM UBO（自定义结构）

```cpp
struct CsmUBO {
    glm::mat4 cascadeViewProj[4];  // 4 个级联的光源 VP 矩阵
    glm::vec4 cascadeSplitDepths;  // [0]=split1, [1]=split2, [2]=split3, [3]=far
};
```

### 纹理数组

| 属性   | 值                                        |
| ------ | ----------------------------------------- |
| 格式   | `VK_FORMAT_D32_SFLOAT`                    |
| 分辨率 | 2048 × 2048 × 4 layers                    |
| 类型   | `VK_IMAGE_TYPE_2D`                        |
| Flags  | `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT` |
| 采样器 | Linear, ClampToBorder, no compare         |

---

## 4. UI 参数

| 参数               | 默认值 | 说明                               |
| ------------------ | ------ | ---------------------------------- |
| Split Lambda       | 0.75   | 视锥分割混合因子（0=线性，1=对数） |
| Visualize Cascades | Off    | 级联区域颜色可视化                 |
| Shadow Filter      | PCSS   | Hard / PCF / PCSS 三种模式         |
| PCF Radius         | 2.0    | PCF 采样核半径（texels）           |
| Light Size         | 25.0   | PCSS 光源尺寸（texels）            |
| Light Intensity    | 0.5    | 方向光强度                         |

---

## 5. 与 04_Shadow 的对比

| 特性         | 04_Shadow            | 14_csm                 |
| ------------ | -------------------- | ---------------------- |
| 阴影贴图数量 | 1 张                 | 4 张（纹理数组）       |
| 光源类型     | 方向光 + 点光 + 面光 | 仅方向光               |
| 阴影精度     | 全局统一             | 近处精细，远处粗糙     |
| 视锥分割     | 无                   | Practical Split Scheme |
| 过滤模式     | Hard / PCF / PCSS    | Hard / PCF / PCSS      |
| 级联可视化   | 无                   | 支持                   |

---

## 6. 参考资料

| 类型     | 资源                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------- |
| 论文     | "Parallel-Split Shadow Maps for Large-scale Virtual Environments"                                             |
| GPU Gems | Chapter 10: "Parallel-Split Shadow Maps on Programmable GPUs"                                                 |
| 博客     | [Cascaded Shadow Maps - MSDN](https://docs.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps) |
| 开源     | https://github.com/SaschaWillems/Vulkan (shadowmappingcascade 示例)                                           |
| 教程     | https://learnopengl.com/Guest-Articles/2021/CSM                                                               |

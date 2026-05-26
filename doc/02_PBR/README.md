# IBL PBR（Level 3/4）

[返回目录](../../README.md)

## 概述

`IBLPBRRenderer` 基于 PBR 直接光照， Image-Based Lighting（IBL）实现间接光照，渲染 49 个球体组成 7×7 grid，展示不同 metallic / roughness 参数组合下的材质效果。天空盒使用全屏三角形渲染 HDR 环境贴图。

## 渲染流程总览

```
initialize()
└── VulkanBase::initialize()
initVulkan()
└── VulkanBase::initVulkan("VulkanRenderer - 2_pbr")
prepareResource()
├── generateSphere(sphereMesh, 1.0f, 100)       // PBR 球体网格
├── createVertexBuffer / createIndexBuffer(sphereMesh)
├── 全屏三角形天空盒网格 (skyboxTriangleMesh)
├── LoadHDRTextureFromFile("newport_loft.hdr")   // 载入 HDR 环境贴图
├── createPBRDescriptorSetLayout / Pool / Pipeline
├── createSkyboxDescriptorSetLayout / Pool / Pipeline
├── generateIBLResources()                      // 重点： 预计算 IBL 贴图（只需做一次）
├── createPBRDescriptorSets()
└── createSkyboxDescriptorSets()

每帧 render():
├── waitForFences → acquireNextImage
├── updatePBRInstanceBuffers(currentFrame)     // CPU 更新所有 UBO/SSBO
├── recordCommandBuffer(imageIndex)            // 录制渲染命令
├── graphicsQueue.submit → presentQueue.presentKHR
```

---

## 数据结构

```cpp
struct PBRInstanceData {
    glm::mat4 model;
    float metallic;
    float roughness;
    alignas(16) glm::vec3 color;
};

struct SceneUBO {
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec3 camPos;
};

struct PointLight {
    glm::vec4 position; // w: intensity/unused
    glm::vec4 color;    // w: intensity
};

struct LightUBO {
    PointLight lights[4]; // 4 个点光源
};

struct ParamsUBO {
    float exposure;
    float gamma;
};

struct SkyboxUBO {
    glm::mat4 invProjection;
    glm::mat4 invView;
};
```

---

## 初始化阶段（一次性资源）

### 几何体

| 网格                 | 类型                       | 用途                          |
| -------------------- | -------------------------- | ----------------------------- |
| `sphereMesh`         | VertexBuffer + IndexBuffer | 球体网格，radius=1.0, 100 段  |
| `skyboxTriangleMesh` | VertexBuffer + IndexBuffer | 全屏三角形（顶点位置 NDC.xy） |
| `cubeMesh`（临时）   | VertexBuffer + IndexBuffer | IBL 预计算用完后销毁          |

### HDR 纹理

```cpp
LoadHDRTextureFromFile("newport_loft.hdr", hdrEquirectData);
```

### IBL 预计算 (`generateIBLResources`)

**只执行一次**，生成 4 张 IBL 贴图：

| 资源                    | 维度    | 层数/Mip      | 格式                | 用途                         |
| ----------------------- | ------- | ------------- | ------------------- | ---------------------------- |
| `envCubemapData`        | 512×512 | 6 层 cubemap  | eR16G16B16A16Sfloat | HDR → cubemap（filtercube）  |
| `irradianceCubemapData` | 64×64   | 6 层 cubemap  | eR16G16B16A16Sfloat | 漫反射 IBL（irradiancecube） |
| `prefilteredEnvMapData` | 512×512 | 6 层 + Mipmap | eR16G16B16A16Sfloat | 镜面 IBL（prefilterenvmap）  |
| `brdfLutData`           | 512×512 | 2D            | eR16G16B16A16Sfloat | BRDF 积分近似（genbrdflut）  |

**预计算步骤：**

1. `filtercube.spv` — 将 2D equirectangular HDR 转换为 cubemap（6 个面分别渲染）
2. `irradiancecube.spv` — 对 cubemap 做漫反射卷积，生成 irradiance map（模糊版环境贴图）
3. `prefilterenvmap.spv` — 对 cubemap 做镜面预滤波，按 roughness 级别生成多 Mip 层
4. `genbrdflut.spv` — 渲染 BRDF LUT 全屏三角形到 2D 贴图

每张 cubemap 通过临时 `ImageView`（选中的 face/mip 层）作为 color attachment 渲染，6 面 × Mip 层循环执行。

### Descriptor Set Layouts

**PBR Pass** (`createPBRDescriptorSetLayout`)：

```cpp
std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer,           .stageFlags = eVertex|eFragment }, // sceneUbo
    {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,           .stageFlags = eVertex|eFragment }, // instanceData
    {.binding = 2, .descriptorType = vk::DescriptorType::eUniformBuffer,           .stageFlags = eFragment },        // lightUbo
    {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler,    .stageFlags = eFragment },        // irradianceCubemap
    {.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler,    .stageFlags = eFragment },        // prefilteredEnvMap
    {.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler,    .stageFlags = eFragment },        // brdfLut
    {.binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer,           .stageFlags = eFragment },        // paramsUbo
};
```

**Skybox Pass** (`createSkyboxDescriptorSetLayout`)：

```cpp
std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer,        .stageFlags = eFragment }, // SkyboxUBO
    {.binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,        .stageFlags = eFragment }, // paramsUbo
    {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler,  .stageFlags = eFragment }, // prefilteredEnvMap
};
```

### Descriptor Pool Sizes

**PBR：** `MAX_FRAMES_IN_FLIGHT × 3` UniformBuffer + `MAX_FRAMES_IN_FLIGHT` StorageBuffer + `MAX_FRAMES_IN_FLIGHT × 3` CombinedImageSampler

**Skybox：** `MAX_FRAMES_IN_FLIGHT × 2` UniformBuffer + `MAX_FRAMES_IN_FLIGHT` CombinedImageSampler

### Buffers

```cpp
createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
createUniformBuffers(lightUboResources, sizeof(LightUBO));
createUniformBuffers(paramsUboResources, sizeof(ParamsUBO));
createUniformBuffers(skyboxUboResources, sizeof(SkyboxUBO));
createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * instanceCount); // instanceCount = 49
```

### Pipelines

| Pipeline         | Shader       | 顶点输入             | 深度测试       | 背面剔除 |
| ---------------- | ------------ | -------------------- | -------------- | -------- |
| `iblPbrPipeline` | `pbribl.spv` | pos + normal         | 开启（Less）   | Back     |
| `skyboxPipeline` | `skybox.spv` | pos only（全屏三角） | 关闭（Always） | None     |

---

## 每帧 CPU 数据准备

**`updatePBRInstanceBuffers(currentFrame)`** 逐帧执行：

### SceneUBO（CPU→GPU）

```cpp
sceneUbo.projection = glm::perspective(fov, aspect, 0.1f, 100.0f)
sceneUbo.view = camera.GetViewMatrix()
sceneUbo.camPos = camera.Position
```

### PBRInstanceData（CPU→GPU）

7×7 grid，49 个实例，每个实例：

```cpp
model = translate(mat4(1), vec3(x*2.2f, y*2.2f, 0))
metallic  = (x + 3) / 6.0f        // 0.0 ~ 1.0
roughness = clamp((y + 3) / 6.0f, 0.04f, 1.0f)
color     = vec3(1.0f, 0.86f, 0.57f)
```

### LightUBO（CPU→GPU）

4 个点光源，随时间移动：

```cpp
lights[0]: pos=(20,20,20),  intensity=400
lights[1]: pos=(-20,-10,10), intensity=50
lights[2]: pos=(sin(t*0.5)*12, 5, 8), intensity=150
lights[3]: pos=(0, cos(t*0.5)*12, 8), intensity=150
```

### ParamsUBO & SkyboxUBO

```cpp
params: { exposure=4.5f, gamma=2.2f }
skyboxUbo: { invProjection=inverse(proj), invView=inverse(view) }
```

---

## 每帧命令录制 (`recordCommandBuffer`)

```
cmdBuffer.begin()
│
│ transition: swapChainImages → ColorAttachmentOptimal
│ transition: depthImage → DepthAttachmentOptimal
│
│ beginRendering(color + depth, clear):
│ setViewport / setScissor
│
│ ★ Skybox Pass（先绘制，无深度写入）
│   bindPipeline(skyboxPipeline)
│   bindVertexBuffers(skyboxTriangleMesh)
│   bindDescriptorSets(skyboxDescriptorSets[currentFrame])
│   drawIndexed(3 indices)  // 全屏三角形
│
│ ★ PBR Pass（实例化球体）
│   bindPipeline(iblPbrPipeline)
│   bindVertexBuffers(sphereMesh)
│   bindIndexBuffer(sphereMesh)
│   bindDescriptorSets(pbrInstanceBufferResources.descriptorSets[currentFrame])
│   drawIndexed(indexCount, 49 instances)  // 49 球体
│
│ endRendering()
│
│ transition: swapChainImages → PresentSrcKHR
│
cmdBuffer.end()
```

**绘制顺序**：Skybox → PBR spheres。先绘制天空盒（深度测试关闭），再绘制场景，无需深度测试，直接覆盖深度。

---

## Shader 关键逻辑（IBL）

### PBR 直接光照（Cook-Torrance BRDF）

镜面 BRDF 三项：
- **D（GGX）**：法线分布，控制高光锐度，roughness 越小高光越集中
- **G（Smith-GGX）**：几何遮蔽，roughness 越大遮蔽越强，高光在掠射角变暗
- **F（Schlick）**：菲涅尔，掠射角反射率提升，F0 由 metallic 控制

### IBL 间接光照

**漫反射 IBL**：用表面法线采样 `irradianceCubemap`，得到环境漫反射贡献

**镜面 IBL**：Epic Games 近似，将积分拆分为两部分：
- 环境部分：用反射方向 R = reflect(-V, N) 采样 `prefilteredEnvMap`，根据 roughness 选择 Mip 级别
- BRDF 部分：`BRDF_LUT.xy` 存储预计算的 scale 和 bias，最终贡献 = `F * LUT.x + LUT.y`

最终颜色 + Reinhard Tone Mapping

### Skybox Shader

用 `invProjection * invView` 从 NDC 反推出世界空间方向，采样 `prefilteredEnvMap`（mip=0）显示 HDR 环境。

---

## UI 面板

```cpp
void IBLPBRRenderer::updateUIPanel()
{
    ImGui::Begin("IBL PBR", nullptr, ImGuiWindowFlags_NoCollapse);
    ImGui::Text("Instance Count: %d", instanceCount); // 49 固定
    ImGui::Text("Exposure: %.1f", 4.5f);
    ImGui::End();
}
```

当前 UI 仅展示信息，无交互控制。

---

## 数据读写关系速查

| 方向    | 资源                         | 操作                                 |
| ------- | ---------------------------- | ------------------------------------ |
| CPU→GPU | `sceneUboResources`          | 每帧 `memcpy` → UBO                  |
| CPU→GPU | `pbrInstanceBufferResources` | 每帧 `memcpy` → SSBO（49 实例）      |
| CPU→GPU | `lightUboResources`          | 每帧 `memcpy` → UBO（4 光源）        |
| CPU→GPU | `paramsUboResources`         | 每帧 `memcpy` → UBO                  |
| CPU→GPU | `skyboxUboResources`         | 每帧 `memcpy` → UBO                  |
| GPU 读  | `irradianceCubemapData`      | Fragment Shader（漫反射 IBL）        |
| GPU 读  | `prefilteredEnvMapData`      | Fragment Shader（镜面 IBL + Skybox） |
| GPU 读  | `brdfLutData`                | Fragment Shader（BRDF LUT）          |

---

## Image Layout 转换汇总

本文件使用两套 barrier 封装：

- **`transitionImageLayoutCmd`**：用于 `generateIBLResources` 初始化阶段，手动传参
- **`transition_image_layout`**：用于 `recordCommandBuffer` 每帧录制，基于 `VulkanBase` 维护的 `swapChainImageLayouts[imageIndex]` / `depthImageLayout` 状态

### 初始化阶段 — `generateIBLResources` 中的 `transitionImageLayoutCmd`

发生在一次性 IBL 预计算 pass 序列中，共 7 次 barrier：

| #   | 资源                    | 范围               | `oldLayout`               | `newLayout`               | `srcStage`               | `dstStage`               | 目的                        |
| --- | ----------------------- | ------------------ | ------------------------- | ------------------------- | ------------------------ | ------------------------ | --------------------------- |
| 1   | `envCubemapData`        | 0..1, 6 层         | `eUndefined`              | `eColorAttachmentOptimal` | `eTopOfPipe`             | `eColorAttachmentOutput` | 为 filtercube 渲染准备      |
| 2   | `irradianceCubemapData` | 0..1, 6 层         | `eUndefined`              | `eColorAttachmentOptimal` | `eTopOfPipe`             | `eColorAttachmentOutput` | 为 irradiancecube 渲染准备  |
| 3   | `prefilteredEnvMapData` | 0..mipLevels, 6 层 | `eUndefined`              | `eColorAttachmentOptimal` | `eTopOfPipe`             | `eColorAttachmentOutput` | 为 prefilterenvmap 渲染准备 |
| 4   | `brdfLutData`           | 0..1, 1 层         | `eUndefined`              | `eColorAttachmentOptimal` | `eTopOfPipe`             | `eColorAttachmentOutput` | 为 genbrdflut 渲染准备      |
| 5   | `envCubemapData`        | 0..1, 6 层         | `eColorAttachmentOptimal` | `eShaderReadOnlyOptimal`  | `eColorAttachmentOutput` | `eFragmentShader`        | 渲染完成后转为 shader 采样  |
| 6   | `irradianceCubemapData` | 0..1, 6 层         | `eColorAttachmentOptimal` | `eShaderReadOnlyOptimal`  | `eColorAttachmentOutput` | `eFragmentShader`        | 渲染完成后转为 shader 采样  |
| 7   | `prefilteredEnvMapData` | 0..mipLevels, 6 层 | `eColorAttachmentOptimal` | `eShaderReadOnlyOptimal`  | `eColorAttachmentOutput` | `eFragmentShader`        | 渲染完成后转为 shader 采样  |
| 8   | `brdfLutData`           | 0..1, 1 层         | `eColorAttachmentOptimal` | `eShaderReadOnlyOptimal`  | `eColorAttachmentOutput` | `eFragmentShader`        | 渲染完成后转为 shader 采样  |

时序：1-4 → (进入渲染循环) → 5-8。即 cubemap 各面/mip 渲染完毕后，再统一做一轮 layout 转换。

### 每帧录制 — `recordCommandBuffer` 中的 `transition_image_layout`

共 3 次 barrier：

| #   | 图像                 | `oldLayout`（上帧遗留）    | `newLayout`               | `srcAccess`             | `dstAccess`                    | `srcStage`               | `dstStage`                                  | 目的                   |
| --- | -------------------- | -------------------------- | ------------------------- | ----------------------- | ------------------------------ | ------------------------ | ------------------------------------------- | ---------------------- |
| 1   | `swapChainImages[i]` | `swapChainImageLayouts[i]` | `eColorAttachmentOptimal` | `{}`                    | `eColorAttachmentWrite`        | `eAllCommands`           | `eColorAttachmentOutput`                    | 为 beginRendering 准备 |
| 2   | `depthImage`         | `depthImageLayout`         | `eDepthAttachmentOptimal` | `{}`                    | `eDepthStencilAttachmentWrite` | `eAllCommands`           | `eEarlyFragmentTests \| eLateFragmentTests` | 为深度测试准备         |
| 3   | `swapChainImages[i]` | `eColorAttachmentOptimal`  | `ePresentSrcKHR`          | `eColorAttachmentWrite` | `{}`                           | `eColorAttachmentOutput` | `eBottomOfPipe`                             | 呈现前转 present 布局  |

> 注意：#1 和 #2 中 `srcAccess` 为空，`srcStage` 使用 `eAllCommands`，表示不关心上一阶段的内存可见性（因为紧接着是 `clear` loadOp，理论上不需要等待）。

---

## IBL 与光照探针：相同点与不同点

### 相同点

**本质都是"对环境光照做离线采样并存储为贴图，运行时直接采样"**：

1. **采样位置固定**：探针在场景中某个位置"拍摄"周围光照，生成 cubemap 或 irradiance map
2. **离线预计算**：环境光照积分（漫反射 SH 卷积 / 镜面重要性采样）不在运行时逐帧计算，而是离线生成
3. **运行时插值光照**：运行时 shader 根据片元位置采样最近探针的数据，作为间接光照输入
4. **都是"全局光照近似"**：都是对场景中光线弹射（multi-bounce）的经验性简化

### 不同点

| 维度             | IBL（本项目）                                  | 光照探针（工程实践）                           |
| ---------------- | ---------------------------------------------- | ---------------------------------------------- |
| **探针数量**     | 全局 1 张（整个场景共享同一张 irradiance map） | 场景中放置多个（如网格排列），运行时插值       |
| **存储内容**     | 单张 global irradiance cubemap + prefiltered   | 每探针独立 irradiance cubemap                  |
| **插值方式**     | 无插值，所有物体采样同一张图                   | 三线性插值（TRILINEAR）或球谐函数（SH）插值    |
| **覆盖范围**     | 大场景远处细节丢失（全局图无法反映局部光照）   | 每个探针覆盖局部区域，细节保留                 |
| **动态物体支持** | 静止场景可用；动态物体仍只能采样 global 图     | 静止探针对动态物体仍有效（探针本身不移动）     |
| **实现复杂度**   | 一次性预计算，无需运行时管理                   | 需编辑器工具摆放探针、运行时插值逻辑           |
| **典型应用**     | 全局环境光照、天空反射                         | 室内/关卡光照 bake、物体在局部光照下的正确着色 |

### 关键理解

> **IBL 是光照探针的最简形式——"单探针、全局覆盖"**。

光照探针的本质和 IBL 完全相同：预计算 + 采样。只是工程中把单个 global 探针扩展为**空间中的一组探针网格**，每个探针存储自己位置的环境辐照度，物体着色时根据世界坐标在三线性插值周围探点的数据。这样既保留了离线预计算的性能优势，又解决了大场景中局部光照细节丢失的问题。

镜面反射接缝问题导致**动态物体通常只插值漫反射 irradiance**，而不插值镜面 prefiltered cubemap——因为粗糙度不同的表面采样不同 mip 层级，插值会在面与面交界处产生明显色带和法线跳变。在实际工程中的使用方式有：

- **漫反射 irradiance**：多探针三线性插值，质量稳定
- **镜面反射**：每个区域放置独立的反射探针（Reflection Probe），运行时根据片元位置直接采样最近探针的 prefiltered cubemap，不做插值；如果需要混合，则用 blend radius 等方式在探针边缘做渐变过渡
- **反射探针单独烘焙**：渲染管线在特定时机（如光照变化、物件进入区域）对特定区域重新"拍摄"Cubemap，将结果烘焙到对应探针，实现局部光照细节（局部光源反射、遮挡等）

# Phase 3: CSM 级联阴影映射 — 完整代码梳理

## 1. 整体架构

```
每帧渲染流程 (recordCommandBuffer):
  ┌─────────────────────────────────────────────────────────────┐
  │ Pass 0: CSM Depth Pass (4 passes, one per cascade)         │
  │   写: csmTextureArrays[currentFrame] (2048×2048×4 layers)  │
  │   Barrier: DepthAttachment —— ShaderReadOnly                │
  │                                                            │
  │ Pass 1: Main Color Pass (PBR + IBL + CSM shadow)           │
  │   读: csmTextureArrays[currentFrame] (via Set 0 Binding 6) │
  │                                                            │
  │ Pass 2: ImGui Pass                                         │
  └─────────────────────────────────────────────────────────────┘
```

关键设计:
- **双缓冲阴影贴图**: `csmTextureArrays[2]`，每个飞行帧（`MAX_FRAMES_IN_FLIGHT=2`）独立拥有，消除帧间竞争
- **独立 CSM 深度管线**: `csmDepthPipeline`，仅输出深度，无颜色附件
- **复用主管线的 Descriptor Set 0**: Binding 5=CsmUBO, 6=shadowMapArray, 7=ShadowParamsUBO

---

## 2. 数据结构

### 2.1 SceneUBO（Binding 0）— 扩展了光源方向

```cpp
struct SceneUBO {
    glm::mat4 projection;    // 相机投影矩阵
    glm::mat4 view;          // 相机视图矩阵
    glm::vec3 camPos;        // 相机世界位置
    float     pad0;
    glm::vec3 lightDir;     // ★ 方向光方向（归一化，CSM 与 PBR 共用）
    float     pad1;
    glm::vec3 lightColor;   // 方向光颜色 (来自 Scene.dirLight())
    float     pad2;
};
// sizeof = 64 + 64 + 16 + 16 + 16 = 176 bytes
```

### 2.2 CsmUBO（Binding 5）

```cpp
struct CsmUBO {
    glm::mat4 cascadeViewProj[4];  // 4 个级联的 lightView * lightProj
    glm::vec4 cascadeSplitDepths;  // x=split1, y=split2, z=split3, w=farPlane
};
// sizeof = 4*64 + 16 = 272 bytes
```

### 2.3 ShadowParamsUBO（Binding 7）

```cpp
struct ShadowParamsUBO {
    int   shadowFilterMode;    // 0=Hard, 1=PCF, 2=PCSS, 3=Visualize（内部覆写）
    float pcfRadiusTexels;     // PCF 采样半径（纹素）
    float pcssLightSizeTexels; // PCSS 光源大小（纹素）
    float shadowBiasMin;       // 基础偏移 (0.0006)
    glm::vec2 invShadowMapSize; // 1/2048, 1/2048
    glm::vec2 padding0;
};
// sizeof = 32 bytes
```

### 2.4 CSM 双缓冲资源

```cpp
// 每个 in-flight frame 拥有一套独立的阴影贴图资源
TextureData csmTextureArrays[2];            // 2048×2048 D32Sfloat × 4 layers
vector<vk::raii::ImageView> csmLayerViewsArray[2]; // 4 per-layer views each
vector<vk::raii::ImageView> csmArrayViews;         // 1 full-array view each
vector<vk::raii::Sampler>   csmSamplers;           // ClampToBorder, compareEnable=false
vk::ImageLayout csmArrayLayouts[2];                // pipeline barrier tracking
```

VRAM 占用: 2 × 2048² × 4 layers × 4 bytes = **128 MB**（D32Sfloat）

---

## 3. 级联分割算法

### 3.1 Practical Split Scheme

```
split_i = λ × near × (far/near)^(i/N)  +  (1-λ) × (near + (far-near) × i/N)
          └─── 对数项 ────┘              └────── 线性项 ──────┘
```

- `λ = 0.45`（默认）: 0=纯线性均匀，1=纯对数近处密集
- `near = 0.1m`, `far = 2000m`

### 3.2 当前分割（λ=0.45, far=2000m）

```
Cascade 0 (Red):    0m ->  276m
Cascade 1 (Green): 276m ->  556m
Cascade 2 (Blue):  556m ->  901m
Cascade 3 (Yellow):901m -> 2000m
```

### 3.3 实现（calculateCascadeSplits）

```cpp
for (uint32_t i = 1; i < CASCADE_COUNT; ++i) {
    float ratio = (float)i / CASCADE_COUNT;
    float logSplit = near * pow(far/near, ratio);
    float linSplit = near + (far-near) * ratio;
    cascadeSplitDepths[i] = λ * logSplit + (1-λ) * linSplit;
}
```

---

## 4. 级联视锥体 AABB 计算

### 4.1 computeCascadeViewProj 

对每个级联 `[nearZ, farZ]`:

1. **计算子视锥体的 8 个角点**（view space）
2. **变换到世界空间**: `cornerWS = invViewMatrix * cornerVS`
3. **变换到光源空间**: `cornerLS = lightViewMatrix * cornerWS`
4. **计算 AABB**: `minAABB = min(cornerLS), maxAABB = max(cornerLS)`
5. **扩展 Z 范围**: `minAABB.z -= 50m, maxAABB.z += 50m`（捕获视锥体外的遮挡物）
6. **构建正交投影**: `lightProj = ortho(minAABB, maxAABB)` + Vulkan Y 翻转
7. **最终矩阵**: `cascadeViewProj[i] = lightProj * lightView`

### 4.2 光源视图矩阵

```cpp
// 动态计算光源距离，确保光源始终在模型（整个场景）的 AABB 外部
lightDist = length(sceneMax - sceneMin) * 1.5f;
lightPos = sceneCenter - lightDir * lightDist;
lightView = lookAt(lightPos, sceneCenter, lightUp);
```

---

## 5. Shader 管线

### 5.1 CSM 深度通道（csm_depth.slang）

```
Pipeline Layout:
  Set 0: CsmUBO (binding 0) — cascade VP matrices
  Push Constant: mat4 modelMatrix (64 bytes) + uint cascadeIndex (4 bytes)

Vertex Shader:
  1. worldPos = modelMatrix * position
  2. output.pos = cascadeViewProj[cascadeIndex] * worldPos

Fragment Shader:
  void — depth written automatically by GPU
```

顶点输入: position only (offset 0, stride 48 = sizeof(VkrVertex))  
光栅化: CullMode=Back, 深度偏移 enable (1.25 const + 1.75 slope)

### 5.2 主通道（scene_vert.slang + scene_frag.slang）

**Vertex Shader 新增输出:**
```hlsl
float4 shadowPos[4] : TEXCOORD5; // 每个级联的 shadow map UV
float  viewDepth    : TEXCOORD9; // 视空间深度，用于级联选择
```

**Fragment Shader 阴影采样管线:**

```
csmShadowFactor(worldPos, normal, lightDir, viewDepth)
  │
  ├─ selectCascade(viewDepth)
  │    └─ 比较 viewDepth 与 cascadeSplitDepths.xyz
  │
  ├─ perspectiveDivide + NDC->UV 转换
  │    └─ uv = shadowPos[cascade].xy / shadowPos[cascade].w * 0.5 + 0.5
  │    └─ refDepth = shadowPos[cascade].z / shadowPos[cascade].w
  │       └─ GLM_FORCE_DEPTH_ZERO_TO_ONE: 已在 [0,1]，无需再映射
  │
  ├─ 斜率缩放偏移: bias = max(biasMin * (1-dot(N,L)), 0.0001)
  │
  └─ 滤波模式:
       ├─ Hard:  单点采样比较
       ├─ PCF:   (2r+1)² 采样点均匀 PCF
       └─ PCSS:  blocker search —— penumbra estimation —— adaptive PCF radius
```

---

## 6. 每帧 CPU 更新流程

```
render():
  │
  ├─ waitForFences(inFlightFences[currentFrame])  ← 等待 2 帧前的提交完成
  ├─ acquireNextImage()
  ├─ vkrUpdateUIFrame()
  │
  ├─ updateSceneUBO(currentFrame)
  │    ├─ 计算 projection * view
  │    ├─ 计算 lightDir（可选旋转: lightAngle += dt*0.15）
  │    └─ memcpy to sceneUboResources
  │
  ├─ updateCsmBuffers(currentFrame)
  │    ├─ 从 sceneUboResources 读取 lightDir（保证一致性）
  │    ├─ calculateCascadeSplits()
  │    ├─ for each cascade: computeCascadeViewProj()
  │    ├─ memcpy CsmUBO —— csmUboResources
  │    └─ memcpy ShadowParamsUBO —— shadowParamsUboResources
  │
  └─ recordCommandBuffer(imageIndex)
```

---

## 7. Descriptor Set 布局

### Set 0（Scene, 每帧 1 套）

| Binding | Type                 | 内容                                       | Stage |
| ------- | -------------------- | ------------------------------------------ | ----- |
| 0       | UBO                  | SceneUBO (projection, view, camPos, light) | V+F   |
| 1       | CombinedImageSampler | Irradiance cubemap                         | F     |
| 2       | CombinedImageSampler | Prefiltered env map                        | F     |
| 3       | CombinedImageSampler | BRDF LUT                                   | F     |
| 4       | UBO                  | ParamsUBO (exposure, gamma, mode)          | F     |
| 5       | UBO                  | CsmUBO (4 cascade VPs + splits)            | V+F   |
| 6       | CombinedImageSampler | shadowMapArray (双缓冲，per-frame)         | F     |
| 7       | UBO                  | ShadowParamsUBO (filter, bias)             | F     |

### Set 1（Material，每材质 1 套，静态）

| Binding | Type         | 内容                     |
| ------- | ------------ | ------------------------ |
| 0       | SampledImage | baseColorTexture         |
| 1       | SampledImage | normalTexture            |
| 2       | SampledImage | metallicRoughnessTexture |
| 3       | Sampler      | 共享采样器               |
| 4       | UBO          | MaterialUBO              |

---

## 8. 累积 Bug 修复清单

| #   | 症状                                                 | 根因                                                                                                                                                  | 修复                                                                                                                                 |
| --- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Device Lost                                          | Push constant struct padding: `glm::mat4` align=16 导致 80 字节写入 68 字节 range                                                                     | 拆为两次 `pushConstants` (offset 0 —— mat4, offset 64 —— uint)                                                                       |
| 2   | 方形阴影漂移                                         | CSM 旋转光源 vs PBR shader 硬编码光源不一致                                                                                                           | SceneUBO 新增 lightDir/lightColor，统一光源                                                                                          |
| 3   | 多帧阴影贴图竞争                                     | MAX_FRAMES_IN_FLIGHT=2，帧N写&帧N-1读同一贴图                                                                                                         | 双缓冲 csmTextureArrays[2]                                                                                                           |
| 4   | csmFence 首帧死锁                                    | Fence 创建时 unsignaled，waitForFences 永久阻塞                                                                                                       | 已移除，改用双缓冲                                                                                                                   |
| 5   | RenderDoc 中前 3 个级联阴影贴图错误（仅第 4 个正确） | `computeCascadeViewProj` 视锥体角点 Z 坐标符号反了：`glm::lookAt` 视图空间中相机看 -Z，但角点用了正 Z（放在相机背后），导致 AABB / lightProj 全部算错 | 角点 Z 改为负值：`-nearZ` / `-farZ`（`CsmRenderer.cpp` 和 `VkrRenderer.cpp` 两处）                                                   |  |
| 6   | 转动视角时随机卡退（TDR -> Device Lost）             | PCSS shader 中 `avgBlockerDepth` 接近 0 时 `penumbraRatio` 爆炸 —— `pcfRadius` 溢出为巨大值 —— PCF 循环数百万次 —— GPU 超时                           | 加 `max(avgBlockerDepth, 0.001)` 防除零；`clamp(pcfRadius, 1, pcssLightSizeTexels)` 上限保护（`scene_frag.slang` / `csm_lit.slang`） |
| 7   | 14_csm shader shadowPos 读取自身                     | `computeShadow` 中 `shadowPos = shadowPos[0]` 从自身 swizzle 读取而非 `input.shadowPos[cascadeIdx]`                                                   | 增加 `float4 shadowPositions[4]` 参数，`shadowPositions[cascadeIdx]` 直接索引                                                        |
| 8   | 光源在 Sponza AABB 内部                              | `lightDist = 50` 对 ~3700 单位的 Sponza 不够，光源在模型内部 —— 部分几何体 z_view>0 —— NDC Z<0 —— 全亮                                                | `lightDist = glm::length(sceneMax - sceneMin) * 1.5f`                                                                                |

## 9. 调试可视化模式 (shadowFilterMode)

| 模式 | UI 名称       | 颜色编码                                     |
| ---- | ------------- | -------------------------------------------- |
| 0    | Hard          | 正常硬阴影                                   |
| 1    | PCF           | 正常 PCF 软阴影                              |
| 2    | PCSS          | 正常 PCSS 软阴影                             |
| 3    | CascadeVis    | Red:C0 / Green:C1 / Blue:C2 / Yellow:C3      |
| 4    | DBG:ShadowMap | Red:UV越界 / Blue:深度越界 / Purple:有效深度 |
| 5    | DBG:ShadowFac | Black:阴影 / White:照亮                      |
| 6    | DBG:NDC-Z     | Red:z<0 / Green:z∈[0,1] / Blue:z>1           |
| 7    | DBG:w         | Green:w≈1 / Orange:w<1 / Cyan:w>1            |

## 10. UI 控制面板

```
CSM Shadows
├── Filter Mode:  [Hard/PCF/PCSS/CascadeVis/DBG:ShadowMap/DBG:ShadowFac/DBG:NDC-Z/DBG:w]
├── Split Lambda: [0.0 ────●──── 1.0]  (0=linear, 1=log)
├── ☑ Visualize Cascades   ☐ Rotate Light
├── PCF Radius:   [1 ──●── 5] texels  (仅 PCF 模式显示)
├── Light Size:   [5 ──●── 50] texels (仅 PCSS 模式显示)
├── Split Depths:
│     Cascade 0~3: (动态计算, far=2000m)
└── Color Legend (Visualize 模式):
      Red=Near  Green  Blue  Yellow=Far
```

---

## 11. 关键参数速查

| 参数             | 值                                      | 位置                   |
| ---------------- | --------------------------------------- | ---------------------- |
| CASCADE_COUNT    | 4                                       | VkrRenderer.h:33       |
| SHADOW_MAP_SIZE  | 2048                                    | VkrRenderer.h:34       |
| 阴影贴图格式     | D32Sfloat (fallback D16Unorm)           | createCsmResources     |
| nearPlane        | 0.1f                                    | calculateCascadeSplits |
| farPlane         | 2000.0f                                 | calculateCascadeSplits |
| splitLambda 默认 | 0.45f                                   | VkrRenderer.h:162      |
| 光源方向         | (0.5+cos(t)*0.3, -0.55, 0.3+sin(t)*0.3) | updateSceneUBO         |
| 光源颜色         | 来自 Scene.dirLight().color             | updateSceneUBO         |
| 光源距离         | length(sceneMax-sceneMin)*1.5 (动态)    | computeCascadeViewProj |
| 旋转速度         | 0.15 rad/s                              | updateSceneUBO         |
| 深度偏移         | const=1.25, slope=1.75                  | createCsmDepthPipeline |
| 阴影偏移         | 0.0006 * slopeScale                     | scene_frag.slang       |
| Z 扩展           | 50.0f                                   | computeCascadeViewProj |
| IBL ambient 强度 | 0.35                                    | scene_frag.slang       |

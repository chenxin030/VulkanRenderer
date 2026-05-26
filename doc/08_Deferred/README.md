# 8_deferredShading — 延迟渲染

## 1. 目标渲染效果

延迟渲染（Deferred Shading）将几何阶段和光照阶段解耦，在不增加 draw call 复杂度的情况下高效支持大量光源的 PBR 渲染。

**渲染画面**：
- 场景包含 49 个球体（7×7 排列），每个球体具有不同的金属度（0→1）和粗糙度（0.04→1），形成材质渐变网格。
- 4 个动态点光源（2 个固定白色高亮、2 个动画运动彩色光源）照射场景。
- 支持 5 种 Debug View 切换：Lit（最终结果）、Albedo（基础颜色）、Normal（法线编码值）、Material（金属度/粗糙度/AO）、Depth（线性化深度）。

---

## 2. 渲染流程总览

```
┌──────────────────────────────────────────────────────────────┐
│                     Pass 1: GBuffer Pass                      │
│  输入: 球体顶点/索引 + 实例数据 (model/color/material)         │
│  输出: Albedo RT + Normal RT + Material RT + Depth RT        │
│        → 四路 MRT 同步写入                                    │
│                                                              │
│                     Pass 2: Lighting Pass                     │
│  输入: GBuffer 四个 RT + SceneUBO + LightUBO + SettingsUBO  │
│  输出: SwapChain Image (最终颜色)                              │
│        → 全屏三角形 + PBR 光照                                │
│                                                              │
│                     Pass 3: UI Pass                           │
│  输入: ImDrawData                                             │
│  输出: SwapChain Image (叠加 ImGui 面板)                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 流程需要用到什么东西，用在哪里

### 3.1 GBuffer Pass

**输入资源**：

| 资源 | 类型 | 绑定位置 | 说明 |
|---|---|---|---|
| `sphereMesh` | VertexBuffer + IndexBuffer | `bindVertexBuffers` | 球体几何体（100 段），`drawIndexed` 49 个实例 |
| `sceneUboResources` | UniformBuffer | binding=0 | VP 矩阵 + 相机位置 |
| `gbufferInstanceBufferResources` | StorageBuffer (StructuredBuffer) | binding=1 | 49 个 `DeferredInstanceData`（model 矩阵 + baseColor + material） |

**GBuffer 附件（MRT 输出）**：

| Attachment | Format | 用途 |
|---|---|---|
| `gbufferAlbedo` | `R16G16B16A16Sfloat` | RGB=漫反射基础颜色，A=1.0 |
| `gbufferNormal` | `R16G16B16A16Sfloat` | RGB=(N×0.5+0.5) 编码世界法线，A=1.0 |
| `gbufferMaterial` | `R16G16B16A16Sfloat` | R=金属度，G=粗糙度，B=AO，A=0.0 |
| `gbufferDepth` | `D32Sfloat` | 场景深度 |

### 3.2 Lighting Pass

**输入资源**（全部通过 `deferredDescriptorSet` 绑定）：

| Binding | 类型 | 内容 |
|---|---|---|
| 0 | `CombinedImageSampler` | `gbufferAlbedo` → `gAlbedo` |
| 1 | `CombinedImageSampler` | `gbufferNormal` → `gNormal` |
| 2 | `CombinedImageSampler` | `gbufferMaterial` → `gMaterial` |
| 3 | `CombinedImageSampler` | `gbufferDepth` → `gDepth` |
| 4 | `UniformBuffer` | `sceneUboResources[currentFrame]`（VP+逆矩阵+camPos） |
| 5 | `UniformBuffer` | `lightUboResources[currentFrame]`（4 个 `PointLight`） |
| 6 | `UniformBuffer` | `deferredSettingsUboResources[currentFrame]`（ambient/exposure/gamma/debugView） |

**几何输入**：`draw(3, 1, 0, 0)` — 全屏三角形（`vertMain` 按 `SV_VertexID` 硬编码三个顶点位置），无需任何 VertexBuffer。

### 3.3 数据结构

```cpp
// C++ 端
struct DeferredInstanceData {
    glm::mat4 model;        // 模型矩阵
    glm::vec4 baseColor;    // 基础颜色
    glm::vec4 material;     // x=金属度, y=粗糙度, z=AO, w=unused
};

struct PointLight {
    glm::vec4 position;    // xyz=位置, w=unused
    glm::vec4 color;        // rgb=颜色, w=强度
};

struct SceneUBO {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 invProjection; // 重建世界坐标用
    glm::mat4 invView;
    glm::vec4 camPos;
};

struct DeferredSettingsUBO {
    glm::vec4 params0;     // x=ambient, y=exposure, z=gamma, w=lightScale
    glm::ivec4 debug;       // x=debugView (0~4)
};
```

---

## 4. 东西的初始化过程

### 4.1 调用链

```
StandaloneMain
 └─ DeferredRenderer::initialize(platform)
     └─ VulkanBase::initialize(_platform)

 DeferredRenderer::initVulkan()
  └─ VulkanBase::initVulkan("VulkanRenderer - 5_deferredPBR")
     → 创建 device / commandPool / swapChain / sync objects

 DeferredRenderer::prepareResource()
  ├─ generateSphere(sphereMesh, 1.0f, 100)
  ├─ createVertexBuffer / createIndexBuffer(sphereMesh)
  ├─ createDeferredBuffers()
  │   ├─ createUniformBuffers(sceneUboResources)
  │   ├─ createUniformBuffers(lightUboResources)
  │   ├─ createUniformBuffers(deferredSettingsUboResources)
  │   └─ createStorageBuffers(gbufferInstanceBufferResources)
  │
  ├─ createGBufferDescriptorSetLayout()
  │   → binding=0: UniformBuffer (sceneUbo)
  │   → binding=1: StorageBuffer (instanceData)
  │
  ├─ createGBufferDescriptorPool()
  │   → MAX_FRAMES_IN_FLIGHT × { UniformBuffer, StorageBuffer }
  │
  ├─ createGBufferResources()
  │   → 创建 4 个 GBuffer Image + ImageView
  │   → 创建 gbufferSampler (Linear, ClampToEdge)
  │
  ├─ createGBufferPipeline()
  │   → vertMain / fragMain ← deferred_gbuffer.spv
  │   → 3 路 MRT color blend (blendEnable=false)
  │   → DepthTest=Yes, DepthWrite=Yes
  │
  ├─ createGBufferDescriptorSets()
  │   → 每帧绑定 sceneUbo + instanceData
  │
  ├─ createDeferredDescriptorSetLayout()
  │   → binding=0~3: CombinedImageSampler (GBuffer×4)
  │   → binding=4: UniformBuffer (sceneUbo)
  │   → binding=5: UniformBuffer (lightUbo)
  │   → binding=6: UniformBuffer (settingsUbo)
  │
  ├─ createDeferredDescriptorPool()
  │   → MAX_FRAMES_IN_FLIGHT × { 4×CombinedImageSampler, 3×UniformBuffer }
  │
  ├─ createDeferredPipeline()
  │   → vertMain / fragMain ← deferred_lighting.spv
  │   → DepthTest=No, DepthWrite=No（全屏三角不需要深度）
  │
  ├─ createDeferredDescriptorSets()
  │   → 每帧绑定 gbufferSampler + 4 个 GBuffer ImageView + 3 个 UBO
  │
  └─ initUI()
      → ImGui context / 字体纹理 / uiPipeline ← imgui.spv
```

### 4.2 GBuffer 资源回收与重建

`recreateDeferredSizedResources()` 在 SwapChain 尺寸变化时被调用：

1. `destroyGBufferResources()` — 释放 4 个 GBuffer Image/Memory/View
2. 重建 Descriptor Pool 和 Descriptor Sets（旧的 set 随旧 pool 一起销毁）

---

## 5. 渲染循环

### 5.1 `render()` 主循环

```cpp
while (platform.processEvents()) {
    renderer.processInput(platform.frameTimer); // WASD/Q/E 相机
    renderer.render();                            // 每帧渲染
    platform.endFrame();
}
```

### 5.2 单帧 `render()` 内部流程

```
1. device.waitForFences(inFlightFences[currentFrame])     // CPU 等待 GPU 完成上帧
2. swapChain.acquireNextImage()                             // 获取当前帧 swapchain image index
3. device.resetFences() / commandBuffers[currentFrame].reset()

4. updateUIFrame()                                          // ImGui NewFrame + updateDeferredUI
5. updateDeferredBuffers(currentFrame)                      // CPU 数据准备
   ├─ SceneUBO: projection / view / invProj / invView / camPos
   ├─ DeferredInstanceData[]: 49 个球的 model/color/material
   │   └── 7×7 网格，metallic=x/6, roughness=clamp(y/6,0.04,1)
   └─ LightUBO: 4 个点光源
       ├── lights[0]: (20,20,20) 白, intensity=400×lightScale
       ├── lights[1]: (-20,-10,10) 暖白, intensity=80×lightScale
       ├── lights[2]: (animSin×12, 5, 8) 青色, intensity=180×lightScale
       └── lights[3]: (0, animCos×12, 8) 紫红, intensity=180×lightScale
   └─ DeferredSettingsUBO: ambient / exposure / gamma / lightScale / debugView

6. recordCommandBuffer(imageIndex)                           // 命令录制（见 5.3）

7. graphicsQueue.submit(wait=presentCompleteSem, signal=renderFinishedSem)
8. presentQueue.presentKHR(wait=renderFinishedSem)
9. currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT
```

### 5.3 `recordCommandBuffer` — 命令录制详细顺序

```
cmdBuffer.begin()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 1: GBuffer Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// GBuffer Image → ColorAttachmentOptimal（4 路）
transition_image_layout(gbufferAlbedo.image,  Undefined → ColorAttachmentOptimal)
transition_image_layout(gbufferNormal.image, Undefined → ColorAttachmentOptimal)
transition_image_layout(gbufferMaterial.image,Undefined → ColorAttachmentOptimal)
transition_image_layout(gbufferDepth.image,  Undefined → DepthAttachmentOptimal)

beginRendering(colorAttachments={Albedo,Normal,Material}, depthAttachment={Depth})
  cmdBuffer.setViewport / setScissor
  cmdBuffer.bindPipeline(gbufferPipeline)
  cmdBuffer.bindVertexBuffers(sphereMesh)     // vertex buffer
  cmdBuffer.bindIndexBuffer(sphereMesh)       // index buffer
  cmdBuffer.bindDescriptorSets(gbufferDescriptorSet[currentFrame])
  cmdBuffer.drawIndexed(indexCount, instanceCount=49, 0, 0, 0)
  // → 49 个实例的球体写入 4 个 MRT，Depth 写入 gbufferDepth
endRendering()

// GBuffer Image → ShaderReadOnlyOptimal（供 Lighting Pass 采样）
transition_image_layout(gbufferAlbedo.image,   ColorAttachmentOptimal → ShaderReadOnlyOptimal)
transition_image_layout(gbufferNormal.image,   ColorAttachmentOptimal → ShaderReadOnlyOptimal)
transition_image_layout(gbufferMaterial.image, ColorAttachmentOptimal → ShaderReadOnlyOptimal)
transition_image_layout(gbufferDepth.image,     DepthAttachmentOptimal → ShaderReadOnlyOptimal)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 2: Lighting Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// SwapChain Image → ColorAttachmentOptimal
transition_image_layout(swapChainImages[imageIndex], PresentSrc → ColorAttachmentOptimal)

beginRendering(colorAttachment={SwapChainImage}, loadOp=Clear(0.03,0.03,0.03,1))
  cmdBuffer.setViewport / setScissor
  cmdBuffer.bindPipeline(deferredPipeline)
  cmdBuffer.bindDescriptorSets(deferredDescriptorSet[currentFrame])
  cmdBuffer.draw(3, 1, 0, 0)  // 全屏三角形
endRendering()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 3: UI Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

beginRendering(colorAttachment={SwapChainImage}, loadOp=Load)
  cmdBuffer.setViewport / setScissor
  recordUI(cmdBuffer)  // ImDrawData → cmdBuffer.drawIndexed
endRendering()

// SwapChain Image → PresentSrcKHR
transition_image_layout(swapChainImages[imageIndex], ColorAttachmentOptimal → PresentSrcKHR)

cmdBuffer.end()
```

### 5.4 Lighting Fragment Shader 逻辑（`deferred_lighting.fragMain`）

```
1. Sample GBuffer: gAlbedo / gNormal / gMaterial / gDepth

2. Debug Switch:
   debugView==1 → return albedo
   debugView==2 → return normal
   debugView==3 → return material
   debugView==4 → return linearizeDepth(depth)
   debugView==0 (默认):
     ↓

3. 重建世界坐标:
   clip = float4(uv*2-1, depth, 1)
   viewPos = invProjection × clip
   worldPos = invView × viewPos
   V = normalize(camPos - worldPos)

4. PBR 光照循环 (4 个光源):
   for each light:
     L = normalize(lightPos - worldPos)
     H = normalize(V + L)
     F0 = lerp(0.04, albedo, metallic)
     F = F_Schlick(dotVH, F0)
     D = D_GGX(dotNH, roughness)
     G = G_SchlickSmith(dotNL, dotNV, roughness)
     specular = D*G*F / (4*dotNL*dotNV)
     kS = F
     kD = (1-kS)*(1-metallic)
     diffuse = kD*albedo/PI
     Lo += (diffuse+specular) * lightColor * intensity/dist² * dotNL

5. ambient = albedo * ambientStrength * ao
   color = ambient + Lo

6. Tone Map: color = 1 - exp(-color * exposure)
   Reinhard: color = color / (color + 1)
   Gamma: color = pow(color, 1/gamma)

7. return float4(color, 1)
```

---

## 6. 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU** | `sceneUboResources` | 每帧 `memcpy` → UBO |
| **CPU→GPU** | `gbufferInstanceBufferResources` | 每帧 `memcpy` → SSBO（49 instances） |
| **CPU→GPU** | `lightUboResources` | 每帧 `memcpy` → UBO（4 lights） |
| **CPU→GPU** | `deferredSettingsUboResources` | 每帧 `memcpy` → UBO |
| **GPU写** | `gbufferAlbedo/Normal/Material` | GBuffer frag shader MRT 输出 |
| **GPU写** | `gbufferDepth` | GBuffer frag shader 深度写入 |
| **GPU读** | `gbufferAlbedo/Normal/Material/Depth` | Lighting frag shader `Sample()` |
| **GPU读** | `sceneUboResources` | GBuffer vert shader + Lighting frag shader |
| **GPU读** | `gbufferInstanceBufferResources` | GBuffer vert/frag shader |
| **GPU读** | `lightUboResources` | Lighting frag shader |

---

## 7. 关键设计决策

- **全屏三角形**（而非四边形）：Lighting Pass 使用 `draw(3, ...)`，`vertMain` 按 `SV_VertexID` 硬编码三个顶点位置（-1,-1）、（3,-1）、（-1,3），避免额外的 VertexBuffer 开销。
- **MRT（Multi-Render Target）**：GBuffer Pass 一次 `beginRendering` 输出 3 路 color attachment + 1 路 depth attachment，避免多次 pass 的状态切换开销。
- **逆矩阵存储**：SceneUBO 中同时存储 `invProjection` 和 `invView`，Lighting Pass 在 fragment shader 中通过 `invProjection × clip` + `invView × viewPos` 精确重建世界坐标，避免前向渲染中 forward 矩阵的重复计算。
- **法线编码**：世界法线 `N` 在 GBuffer 中存储为 `(N×0.5+0.5)`（从 [-1,1] 映射到 [0,1]），Lighting 时解码为 `N×2.0-1.0`。

# 10_gi — Screen Space Ambient Occlusion（SSAO）

## 1. 目标渲染效果

在延迟渲染的基础上，叠加屏幕空间环境光遮蔽（SSAO），通过在 GBuffer 法线和深度信息中采样半球方向的深度值，计算每个像素的 AO 因子，提升场景的接触阴影和层次感。

**渲染画面**：
- 7×7 球体网格（金属度/粗糙度渐变），4 个动态点光源。
- SSAO 产生的接触阴影使球体与地面交界处、球体之间缝隙产生自然的暗区。
- 支持 6 种 Debug View：Lit（最终结果）、Albedo、Normal、Material、Depth、SSAO。
- SSAO 参数可调：Radius（采样半径）、Bias（深度偏移）、Intensity（遮蔽强度）。

---

## 2. 渲染流程总览

```
┌──────────────────────────────────────────────────────────────┐
│                      Pass 1: GBuffer Pass                      │
│  输入: 球体顶点/索引 + 49 实例数据                            │
│  输出: Albedo RT + Normal RT + Material RT + Depth RT (MRT)  │
│                                                              │
│                      Pass 2: SSAO Pass                        │
│  输入: GBuffer Normal RT + Depth RT + 4×4噪声纹理 + 16样本核  │
│  输出: ssaoColor (R8Unorm) — 原始 AO 值 [0,1]              │
│                                                              │
│                      Pass 3: SSAO Blur Pass                   │
│  输入: ssaoColor                                             │
│  输出: ssaoBlurColor (R8Unorm) — 加权双边模糊               │
│                                                              │
│                      Pass 4: Lighting + AO Pass                │
│  输入: GBuffer 4 RT + ssaoBlurColor + SceneUBO + LightUBO  │
│  输出: SwapChain                                             │
│                                                              │
│                      Pass 5: UI Pass                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 流程需要用到什么东西，用在哪里

### 3.1 Pass 1 — GBuffer Pass

| 资源 | 类型 | Binding | 用途 |
|---|---|---|---|
| `sphereMesh` | VB + IB | — | 49 实例球体 |
| `gbufferInstanceBufferResources` | StorageBuffer | b1 | 49×DeferredInstanceData |
| `sceneUboResources` | UniformBuffer | b0 | VP + 逆矩阵 + camPos |

**GBuffer 附件**：与 `deferred_gbuffer.slang` 完全相同（3×R16G16B16A16Sfloat MRT + D32Sfloat depth）。

### 3.2 Pass 2 — SSAO Pass

| 资源 | 类型 | Binding | 用途 |
|---|---|---|---|
| `ssaoSettingsUboResources` | UniformBuffer | b0 | invProj + invView + radius/bias/intensity/enabled + 16样本核 |
| `gbufferNormal` | CombinedImageSampler | b1 | 解码法线方向 |
| `gbufferDepth` | CombinedImageSampler | b2 | 重建 view-space 位置 |
| `noiseTexture` | CombinedImageSampler | b3 | 4×4 随机旋转向量（repeat 采样） |

**SSAO 参数（`SsaoSettingsUBO`）**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `invProjection` | mat4 | 投影逆矩阵（重建 view-space 位置） |
| `invView` | mat4 | 视图逆矩阵 |
| `params0.x` | float | radius=0.5（采样半球半径） |
| `params0.y` | float | bias=0.025（避免自遮挡） |
| `params0.z` | float | intensity=1.5（AO 强度幂次） |
| `params0.w` | float | ssaoEnabled=1.0 |
| `flags.x` | int | debugView（当=5 时输出 raw SSAO） |
| `kernel[0..15]` | vec4[16] | 半球采样方向（view-space） |

**噪声纹理**：4×4 R32G32B32A32Sfloat，`GL_LINEAR` + `eRepeat`，每帧固定不变。

**输出**：`ssaoColor`（R8Unorm，每像素 AO 值，0=完全遮挡，1=无遮挡）。

### 3.3 Pass 3 — SSAO Blur Pass

| 资源 | 类型 | Binding | 用途 |
|---|---|---|---|
| `ssaoBlurUboResources` | UniformBuffer | b0 | texelSize = {1/width, 1/height} |
| `ssaoColor` | CombinedImageSampler | b1 | 原始 SSAO 纹理 |

**Blur 类型**：加权双边模糊，9×9 邻域采样，距离中心越远的样本权重越小（`weight = 1/(1+length(offset))`）。

### 3.4 Pass 4 — Lighting + AO Pass

| 资源 | 类型 | Binding | 用途 |
|---|---|---|---|
| `gbufferAlbedo/Normal/Material/Depth` | CombinedImageSampler | b0~3 | GBuffer 4 RT |
| `ssaoBlurColor` | CombinedImageSampler | b4 | 模糊后 AO |
| `sceneUboResources` | UniformBuffer | b5 | VP + 逆矩阵 + camPos |
| `lightUboResources` | UniformBuffer | b6 | 4 点光源 |
| `ssaoSettingsUboResources` | UniformBuffer | b7 | params0 + flags |

---

## 4. 东西的初始化过程

### 4.1 调用链

```
prepareResource()
 ├─ generateSphere(sphereMesh, 1.0f, 100)
 ├─ createVertexBuffer / createIndexBuffer(sphereMesh)
 ├─ createDeferredBuffers()
 │   ├─ createStorageBuffers(gbufferInstanceBufferResources) // 49×DeferredInstanceData
 │   ├─ createUniformBuffers(sceneUboResources)              // SceneUBO
 │   ├─ createUniformBuffers(lightUboResources)             // LightUBO
 │   └─ createUniformBuffers(ssaoSettingsUboResources)       // SsaoSettingsUBO
 │   └─ createUniformBuffers(ssaoBlurUboResources)           // BlurUBO
 │
 ├─ createGBufferResources()    // GBuffer 4 RT + noiseTexture
 ├─ createGBufferDescriptorSetLayout() → b0:Uniform, b1:Storage
 ├─ createGBufferDescriptorPool()
 ├─ createGBufferPipeline() → deferred_gbuffer.spv
 └─ createGBufferDescriptorSets()
 │
 ├─ createSsaoDescriptorSetLayout() → b0:UBO, b1~3:ImageSampler
 ├─ createSsaoDescriptorPool()
 ├─ createSsaoResources() → ssaoColor Image + noiseTexture
 ├─ createSsaoPipeline() → ssao.spv
 └─ createSsaoDescriptorSets()
 │
 ├─ createSsaoBlurDescriptorSetLayout() → b0:UBO, b1:Image
 ├─ createSsaoBlurDescriptorPool()
 ├─ createSsaoBlurResources() → ssaoBlurColor Image
 ├─ createSsaoBlurPipeline() → ssao_blur.spv
 └─ createSsaoBlurDescriptorSets()
 │
 ├─ createLightingDescriptorSetLayout() → b0~4:Image, b5~7:UBO
 ├─ createLightingDescriptorPool()
 ├─ createLightingPipeline() → ssao_lighting.spv
 └─ createLightingDescriptorSets()
 │
 └─ initUI()
```

### 4.2 SSAO 噪声纹理与采样核生成

```cpp
// 噪声纹理：4×4 随机向量（每像素查表使用）
// uvScale = fragmentPos.xy / noiseScale → [0,4) 范围
// texture(noise, uv) → 随机旋转向量 → 构建 TBN 矩阵
std::vector<glm::vec4> ssaoNoise(16);
for (int i = 0; i < 16; ++i) {
    // 均匀分布随机单位向量，存储为 [-1,1] 范围
    ssaoNoise[i] = glm::vec4(randomDir.x, randomDir.y, 0.0f, 0.0f);
}
// 创建 Image + ImageView (R32G32B32A32Sfloat) + nearest sampler (repeat)

// 采样核：16 个 hemisphere 方向（view-space，z 朝向法线方向）
for (int i = 0; i < 16; ++i) {
    float scale = float(i)/16.0f;
    scale = 0.1f + 0.9f * scale*scale; // 1→16 逐渐变大
    kernel[i] = glm::vec4( hemisphereDir * scale, 0.0f );
}
```

---

## 5. 渲染循环

### 5.1 `render()` 主流程

```
1. waitForFences / acquireNextImage / resetFences
2. updateUIFrame()
3. updateBuffers(currentFrame)        // sceneUbo + lightUbo + ssaoSettingsUbo + blurUbo
4. recordCommandBuffer(imageIndex)
5. submit → present
```

### 5.2 `recordCommandBuffer` 完整命令序列

```
cmdBuffer.begin()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 1: GBuffer Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GBuffer → ColorAttachmentOptimal / DepthAttachmentOptimal
 beginRendering(albedo+normal+material+depth)
   bindPipeline(gbufferPipeline)
   bindVertexBuffers(sphereMesh)
   bindIndexBuffer(sphereMesh)
   bindDescriptorSets(gbufferInstanceBufferResources.descriptorSets[currentFrame])
   drawIndexed(sphereMesh × 49 instances)
 endRendering()
 GBuffer → ShaderReadOnlyOptimal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 2: SSAO Pass (全分辨率)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 transition ssaoColor → ColorAttachmentOptimal
 beginRendering(ssaoColor)
   bindPipeline(ssaoPipeline)
   bindDescriptorSets(ssaoSettingsUboResources.descriptorSets[currentFrame])
   draw(3)
 endRendering()
 transition ssaoColor → ShaderReadOnlyOptimal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 3: SSAO Blur Pass (全分辨率)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 transition ssaoBlurColor → ColorAttachmentOptimal
 beginRendering(ssaoBlurColor)
   bindPipeline(ssaoBlurPipeline)
   bindDescriptorSets(ssaoBlurColor.descriptorSet[currentFrame])
   draw(3)
 endRendering()
 transition ssaoBlurColor → ShaderReadOnlyOptimal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 4: Lighting + AO Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 swapChain → ColorAttachmentOptimal
 beginRendering(swapChain)
   bindPipeline(lightingPipeline)
   bindDescriptorSets(lightUboResources.descriptorSets[currentFrame])
   draw(3)
 endRendering()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Pass 5: UI Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 beginRendering(swapChain, loadOp=Load)
   recordUI(commandBuffer)
 endRendering()
 swapChain → PresentSrcKHR

cmdBuffer.end()
```

### 5.3 SSAO Fragment Shader 逻辑（`ssao.fragMain`）

```
当 flags.x == 5 时输出 raw SSAO，否则 return float4(1,1,1,1)

1. gDepth.Sample(uv) → depth
   if depth >= 1.0 (背景): return 1.0

2. 重建 view-space 位置:
   clip = float4(uv*2-1, depth*2-1, 1)
   viewPos = invProjection × clip
   viewPos /= viewPos.w

3. 读取法线 + 构建 TBN:
   N = gNormal.Sample(uv).rgb * 2 - 1
   randomVec = noiseSampler.Sample(uv * 0.25).xyz * 2 - 1
   tangent = normalize(randomVec - N * dot(randomVec, N))
   bitangent = cross(N, tangent)
   TBN = float3x3(tangent, bitangent, N)

4. 半球采样 (16 次):
   for i in [0, 16):
     sampleDir = kernel[i].xyz × TBN   // 转换到 view-space
     sampleViewPos = viewPos + sampleDir × radius
     offset = invProjection × float4(sampleViewPos, 1)
     offset.xyz /= offset.w
     offset.xy = offset.xy * 0.5 + 0.5 → sampleUV

     sampleDepth = gDepth.Sample(sampleUV)
     if sampleDepth >= 1.0: sampleDepth = 0 (背景忽略)

     重建 sample 的 view-space Z:
     sampleViewZ = reconstructViewPos(sampleUV, sampleDepth).z

     rangeCheck = smoothstep(0, 1, radius / |viewPos.z - sampleViewZ|)
     occlusion += (sampleViewZ >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck

5. ao = 1.0 - (occlusion / 16.0)
   ao = pow(ao, intensity)   // 增强对比度
   return float4(ao, ao, ao, 1.0)
```

### 5.4 Lighting Fragment Shader 中的 AO 合成（`ssao_lighting.fragMain`）

```
combinedAO = matAO  // GBuffer 中的 AO
if (ssaoEnabled):
    combinedAO = matAO * ao  // GBuffer AO × SSAO AO 叠加

ambient = albedo * ambientStrength * combinedAO
color = ambient + Lo
color = ACESFilmic(color * exposure)
color = pow(color, 1/gamma)
```

---

## 6. 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU** | `sceneUboResources` | 每帧 memcpy → UBO |
| **CPU→GPU** | `lightUboResources` | 每帧 memcpy → UBO（4 lights） |
| **CPU→GPU** | `ssaoSettingsUboResources` | 每帧 memcpy（invProj/invView/kernel） |
| **CPU→GPU** | `ssaoBlurUboResources` | 每帧 memcpy（texelSize） |
| **GPU写** | `gbufferAlbedo/Normal/Material` | GBuffer frag MRT |
| **GPU写** | `gbufferDepth` | 深度写入 |
| **GPU写** | `ssaoColor` | SSAO frag 输出（R8Unorm） |
| **GPU写** | `ssaoBlurColor` | SSAO blur frag 输出 |
| **GPU读** | `gbufferNormal/Depth` | SSAO frag Sample() |
| **GPU读** | `noiseTexture` | SSAO frag Sample(uv×0.25) 随机旋转 |
| **GPU读** | `ssaoColor` | Blur frag Sample() |
| **GPU读** | `gbufferAlbedo/Normal/Material/Depth` | Lighting frag Sample() |
| **GPU读** | `ssaoBlurColor` | Lighting frag Sample() → combinedAO |

# Descriptor Set 分组设计 — vkrEngine

## 1. 核心理念

Descriptor Set 分组遵循：**按更新频率分层**。

整个渲染器有三种变化节奏：

```
每帧变化     → Set 0 (Scene)
每材质变化   → Set 1 (Material)
每 Pass 变化 → Set 2 (Pass-specific)
```

把它们塞进同一个 Set 当然也能跑，但会付出不必要的 CPU/GPU 开销。

---

## 2. 当前分组总览

### Set 0 — Scene 级别（Per-Frame，8 bindings）

| Binding | 类型                   | 内容                                        | 更新频率 |
| ------- | ---------------------- | ------------------------------------------- | -------- |
| 0       | UBO                    | SceneUBO（projection, view, camPos, light） | 每帧     |
| 1       | Combined Image Sampler | Irradiance Cubemap（IBL diffuse）           | 静态     |
| 2       | Combined Image Sampler | Prefiltered Env Map（IBL specular）         | 静态     |
| 3       | Combined Image Sampler | BRDF LUT（2D）                              | 静态     |
| 4       | UBO                    | ParamsUBO（exposure, gamma, lightingMode）  | 偶尔     |
| 5       | UBO                    | CsmUBO（4×cascadeViewProj, splitDepths）    | 每帧     |
| 6       | Combined Image Sampler | Shadow Map Array（CSM depth, 双缓冲）       | 每帧     |
| 7       | UBO                    | ShadowParamsUBO（filterMode, bias, radius） | 偶尔     |

### Set 1 — Material 级别（Per-Material，5 bindings）

| Binding | 类型          | 内容                          |
| ------- | ------------- | ----------------------------- |
| 0       | Sampled Image | baseColorTexture              |
| 1       | Sampled Image | normalTexture                 |
| 2       | Sampled Image | metallicRoughnessTexture      |
| 3       | Sampler       | 共享采样器（Linear + Repeat） |
| 4       | UBO           | MaterialUBO（factor, flags）  |

### Set 2 — Cluster 级别（Per-Pass，4 bindings，Phase 5）

| Binding | 类型          | 内容                                       |
| ------- | ------------- | ------------------------------------------ |
| 0       | StorageBuffer | Light Buffer（点光源列表）                 |
| 1       | StorageBuffer | Light Grid（每个 cluster 的 offset+count） |
| 2       | StorageBuffer | Light Index List（扁平索引数组）           |
| 3       | UBO           | ClusterParamsUBO（网格参数）               |

### 各 DescriptorSetLayout 与 Pass 的归属关系

> **读法**：这张表跟踪的是 `VkDescriptorSetLayout` 对象——哪个 Pass 的 Pipeline Layout 引用了它。
> "共享"意味着同一个 `VkDescriptorSet` 对象可以被多个 Pass 的 `vkCmdBindDescriptorSets` 绑定，无需重复创建或更新。

| DescriptorSetLayout（Vulkan 对象）   | 被哪些 Pass 引用                        | 绑定内容                                                                  |
| ------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------- |
| `sceneDescriptorSetLayout`           | **前向管线**、**GBuffer**、**延迟光照** | SceneUBO + IBL(3) + ParamsUBO + CsmUBO + ShadowMapArray + ShadowParamsUBO |
| `materialDescriptorSetLayout`        | **前向管线**、**GBuffer**               | 材质纹理(3) + Sampler + MaterialUBO                                       |
| `deferredDescriptorSetLayout`        | **延迟光照**                            | GBuffer Albedo/Normal/PBR/Depth + SSR + DeferredSettingsUBO   |
| `deferredClusterDescriptorSetLayout` | **延迟光照**（Set 2）                   | Light Buffer + Light Grid + Light Index + ClusterParamsUBO                |
| `csmDepthDescriptorSetLayout`        | **CSM Depth**（独立管线）               | CsmUBO                                                                    |
| `ssrDescriptorSetLayout`             | **SSR**（独立管线）                     | SceneUBO + GBuffer Depth/Color/Normal + SSRParams UBO                     |
| `clusterComputeDescriptorSetLayout`  | **Cluster Compute**（独立管线）         | Light Buffer + Light Grid + Light Index + ClusterParamsUBO                |

**共享关系一览**：

```
sceneDescriptorSetLayout       ──→ 前向管线、GBuffer、延迟光照  (3 Pass 共享)
materialDescriptorSetLayout    ──→ 前向管线、GBuffer            (2 Pass 共享)
其他 5 个 layout              ──→ 各自仅 1 个 Pass 使用
```

**核心设计意图**：
- `sceneDescriptorSetLayout` 被 3 个 Pass 共享，因为 camera/light/shadow 数据在所有 pass 中含义一致。代码中只用一份 `sceneUboResources`，三个 Pass 绑同一个 descriptor set 对象，零冗余。
- `materialDescriptorSetLayout` 被前向管线和 GBuffer Pass 共享——两者都需要材质纹理来采样 PBR 参数。
- SSR 等后处理 Pass 各自有独立 layout，因为它们需要的资源组合完全不同。

---

## 3. 为什么要分开？四个判断标准

### 3.1 更新频率不同（最核心的原因）

Sponza 有 103 个子网格、25 种材质。如果所有 descriptor 都放 Set 0：

```
每帧 = 更新 Set 0 (8 个 binding) → bind → draw 103 次
```

每个 draw 前都要重写整个 descriptor set（即使 25 种材质的纹理从未改变）。

分开后：

```
每帧：
  vkUpdateDescriptorSets(Set 0)        // 1 次：scene UBO + CSM
  vkCmdBindDescriptorSets(Set 0)       // 1 次

  for each material (最多 25 种):
    vkCmdBindDescriptorSets(Set 1[mat]) // 仅切换材质 set
    for each submesh:
      vkCmdDrawIndexed()
```

- Set 0 写 1 次（准确说是 swapchain image 数量，即 2 次，双缓冲）
- Set 1 创建时写 25 次，之后永远不碰
- Draw 时只做 `vkCmdBindDescriptorSets` 切换，零 CPU 写入开销

### 3.2 Pipeline Layout 兼容性：不同 Pass 共享 Set 0

前向和延迟管线可以共享同一个 Set 0 layout，但有不同的 Set 1/Set 2：

```
前向渲染管线:
  Pipeline Layout = { Set0(Scene), Set1(Material) }

延迟光照管线:
  Pipeline Layout = { Set0(Scene), Set1(GBuffer inputs), Set2(Cluster) }
```

关键收益：SceneUBO 的结构在所有 Pass 中一致，只需维护一份 `sceneUboResources`，所有 Pass 都 bind 同一个 Set 0 descriptor set。避免了为每个 Pass 复制一份 camera/light/shadow 数据。

### 3.3 绑定粒度：Set 1 在 Draw Call 间切换

```cpp
// 伪代码：画 Sponza 的所有子网格
vkCmdBindDescriptorSets(cmd, 0, Set0); // Set 0 绑一次

uint32_t lastMaterial = ~0u;
for (auto& submesh : sponzaModel.submeshes) {
    if (submesh.materialIndex != lastMaterial) {
        // 只在材质变化时切换 Set 1
        vkCmdBindDescriptorSets(cmd, 1, materialSets[submesh.materialIndex]);
        lastMaterial = submesh.materialIndex;
    }
    vkCmdDrawIndexed(submesh.indexCount, 1, submesh.firstIndex, 0, 0);
}
```

如果把材质纹理放在 Set 0，每次 draw 前都要更新 Set 0 的纹理 binding —— 103 次不必要的 `vkUpdateDescriptorSets`。

### 3.4 硬件限制

单个 Descriptor Set 的 binding 数量有上限。Vulkan 规范保证最少支持 **8 个**（`maxPerStageDescriptorSampledImages` 等限制因类型而异）。你的 Set 0 已经 8 个 binding，如果再把 5 个材质纹理 + 4 个 cluster buffer 塞进去，会超过 8 个的限制，在部分硬件上直接报错。

分开后每组都安全：

| Set   | Binding 数 | 状态 |
| ----- | ---------- | ---- |
| Set 0 | 8          | 安全 |
| Set 1 | 5          | 安全 |
| Set 2 | 4          | 安全 |

---

## 4. 什么时候不分开？

以下场景全部放 Set 0 是合理的：

| 场景                       | 理由                         |
| -------------------------- | ---------------------------- |
| 只有 1~2 个材质            | 切换 Set 的收益微乎其微      |
| 所有资源每帧都变化         | 分不分开都要重写，没有收益   |
| 简单demo/原型              | 代码简洁优先，后续再重构     |
| 所有 Pass 用完全相同的资源 | Pipeline layout 不需要差异化 |

项目的早期 Phase 1/2 阶段其实就是全部放 Set 0 的——那时只有 1 个 Set。Phase 3 加入 CSM 后仍然只有一个 Set。到 Phase 4 延迟渲染才拆出 Set 1（GBuffer inputs），Phase 5 再加 Set 2（Cluster data）。**这是渐进式重构的正确节奏——先跑起来，再按需拆分。**

---

## 5. 演进历史

```
Phase 1: Set 0 × 1
  只有 SceneUBO + IBL textures + ParamsUBO（6 bindings）

Phase 2: Set 0 × 1
  同上，未变

Phase 3: Set 0 × 1（扩展到 8 bindings）
  + CsmUBO (Binding 5), ShadowMapArray (Binding 6), ShadowParamsUBO (Binding 7)
  Set 0 接近满载

Phase 4: Set 0 + Set 1
  拆分 Set 1 = Material textures + MaterialUBO
  新增独立 Pass 的 Set（SSR、Deferred 各有专属 layout）
  但前向管线仍然 {Set0, Set1}

Phase 5: Set 0 + Set 1 + Set 2
  新增 Set 2 = Cluster light data
  仅延迟光照管线使用 {Set0, Set1, Set2}，前向管线不受影响
```

---

## 6. 经验法则

问自己三个问题：

1. **这个资源跟着帧变吗？** → Set 0
2. **这个资源跟着物体/材质变吗？** → Set 1
3. **这个资源只有某个 Pass 用吗？** → Set 2（或独立 Set）

答案直接对应分组。

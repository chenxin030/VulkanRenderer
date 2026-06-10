# Phase 5: GPU 遮挡剔除 + Clustered Shading

## 1. 功能概述

Phase 5 在 Phase 4 延迟渲染基础上新增两个优化模块：

1. **GPU 遮挡剔除（Hi-Z Occlusion Culling）**：使用 Hi-Z Mipmap 加速视锥体剔除，减少被遮挡几何体的绘制
2. **Clustered Shading（聚类光照）**：将 2048 个动态点光源按屏幕空间 cluster 分配，提升多光源场景的光照效率

**渲染管线**：

```
┌──────────────────────────────────────────────────────────────────┐
│ 渲染流程                                                         │
│                                                                  │
│  1. Hi-Z Build (Compute)                                        │
│     └─ GBuffer Depth → Hi-Z Mipmap Pyramid                        │
│                                                                  │
│  2. Occlusion Cull (Compute)                                      │
│     └─ 视锥体剔除 + Hi-Z 遮挡剔除 → visibility[]                │
│                                                                  │
│  3. Cluster Compute (Compute)                                      │
│     └─ 2048 点光源分配到 screen-space clusters                   │
│                                                                  │
│  4. GBuffer Pass (Graphics)                                      │
│     └─ 只绘制 visibility[i] == 1 的 submeshes                   │
│     └─ transition depth → Hi-Z Build input                        │
│                                                                  │
│  5. SSR Pass (Graphics)                                          │
│                                                                  │
│  6. Deferred Lighting (Graphics)                                  │
│     └─ Set 2: cluster light buffer + grid + index               │
│                                                                  │
│  7. UI Pass (Graphics)                                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据结构

### 2.1 Hi-Z Occlusion Culling

**CullingInstanceUBO**（per-submesh，storage buffer）：
```cpp
struct CullingInstanceUBO {
    glm::vec4 aabbMin;     // 世界空间 AABB min
    glm::vec4 aabbMax;     // 世界空间 AABB max
    glm::ivec4 drawInfo;  // x=indexCount, y=firstIndex, z=materialIndex, w=unused
};
```

**CullingParamsUBO**（per-frame，uniform buffer）：
```cpp
struct CullingParamsUBO {
    glm::vec4 frustumPlanes[6];  // 6 个视锥体平面 (Left, Right, Bottom, Top, Near, Far)
    glm::vec4 hiZInfo;            // x=width, y=height, z=mipCount, w=unused
    uint32_t  totalInstances;     // submesh 总数
    uint32_t  enabled;            // 1=启用, 0=禁用（全可见）
    float     pad0, pad1;
};
```

**Visibility Buffer**（per-frame，storage buffer）：
- 类型：`uint32_t[]`，长度 = `cullingTotalInstances`
- 值：0 = 剔除，1 = 可见
- 回读：`copyBuffer` → CPU 用于 UI 统计

**Hi-Z Texture**（R32Sfloat，多级纹理）：
- 尺寸：与 swapchain 相同
- Mip 级别：⌊log₂(max(W, H))⌋ + 1
- Mip 0 = GBuffer 深度（只读）
- Mip 1+ = Compute shader 构建的最小深度金字塔

### 2.2 Clustered Shading

**GpuPointLight**（2048 个，storage buffer）：
```cpp
struct GpuPointLight {
    glm::vec4 position;  // xyz = 世界位置, w = unused
    glm::vec4 color;   // rgb = 颜色, w = intensity
};
```

**ClusterParamsUBO**（per-frame，uniform buffer）：
```cpp
struct ClusterParamsUBO {
    uint32_t  clusterX, clusterY, clusterZ;  // 默认 16, 9, 24
    uint32_t  totalClusters;                  // = clusterX * clusterY * clusterZ
    glm::vec3 tileSize;
    float     pad0;
    glm::vec3 cameraPos;
    float     nearZ;
    float     farZ, zMin, zMax;
    float     clusteredEnabled;   // 1.0 = 启用, 0.0 = 禁用
    float     visualizeClusters;  // 1.0 = 显示 cluster 热力图
    float     fovY;              // 相机 FOV（度）
    float     aspectRatio;        // 宽高比
    float     pad2, pad3;
};
```

**LightGridCell**（totalClusters 个，storage buffer）：
```cpp
struct LightGridCell {
    uint32_t offset;  // lightIndexBuffer 中的起始索引（固定为 0）
    uint32_t count;   // 该 cluster 包含的光源数量
};
```

**lightIndexBuffer**：
- 大小：`totalClusters × MAX_LIGHTS_PER_CLUSTER × sizeof(uint32_t)`
- 每个 cluster 最多 64 个光源索引

---

## 3. Shader 清单

| Shader | 文件 | 入口 | 用途 |
|--------|------|------|------|
| `hiz_build` | `hiz_build.slang` | `compMain` | Hi-Z Mipmap 构建 |
| `occlusion_cull` | `occlusion_cull_comp.slang` | `compMain` | 视锥体 + Hi-Z 遮挡剔除 |
| `cluster_comp` | `vkr_cluster_comp.slang` | `compMain` | Cluster 光源分配 |
| `deferred_gbuffer` | `deferred_gbuffer.slang` | `vertMain/fragMain` | GBuffer MRT（已支持 visibility 过滤） |
| `deferred_lighting` | `deferred_lighting.slang` | `vertMain/fragMain` | 延迟光照 + Cluster 光照（Set 2） |

---

## 4. Descriptor Set 布局

### Set 0 — Scene-level（所有 Pass 共享）

| Binding | 类型 | 内容 |
|---------|------|------|
| 0 | UBO | SceneUBO |
| 1 | CombinedImageSampler | Irradiance Cubemap |
| 2 | CombinedImageSampler | Prefiltered Env Map |
| 3 | CombinedImageSampler | BRDF LUT |
| 4 | UBO | ParamsUBO |
| 5 | UBO | CsmUBO |
| 6 | CombinedImageSampler | Shadow Map Array |
| 7 | UBO | ShadowParamsUBO |

### Set 1 — GBuffer/SSR（Deferred Lighting Pass）

| Binding | 类型 | 内容 |
|---------|------|------|
| 0 | SampledImage | GBuffer Albedo |
| 1 | SampledImage | GBuffer Normal+Roughness |
| 2 | SampledImage | GBuffer PBR |
| 3 | SampledImage | GBuffer Depth |
| 4 | SampledImage | SSR |
| 5 | UBO | DeferredSettingsUBO |

### Set 2 — Cluster Lights（Deferred Lighting + Forward Lighting）

| Binding | 类型 | 内容 |
|---------|------|------|
| 0 | StorageBuffer | clusterLightBuffer（2048 点光源） |
| 1 | StorageBuffer | lightGrid（LightGridCell per cluster） |
| 2 | StorageBuffer | lightIndexBuffer（光源索引列表） |
| 3 | UBO | ClusterParamsUBO |

### Hi-Z Build Descriptor Set（独立）

| Binding | 类型 | 内容 |
|---------|------|------|
| 0 | StorageImage | hizOut（当前 mip，写） |
| 1 | StorageImage | hizIn（前一级 mip，读） |
| 2 | CombinedImageSampler | gbufferDepth（仅 mip 0 采样） |

### Occlusion Cull Descriptor Set（独立）

| Binding | 类型 | 内容 |
|---------|------|------|
| 0 | StorageBuffer | instances（AABB 数据） |
| 1 | StorageBuffer | visibility（输出） |
| 2 | UBO | CullingParamsUBO |
| 3 | CombinedImageSampler | Hi-Z Texture |

---

## 5. 每帧渲染流程

### CPU 端

```
render():
  │
  ├─ waitForFences(inFlightFences[currentFrame])
  ├─ acquireNextImage()
  ├─ vkrUpdateUIFrame()
  ├─ updateSceneUBO(currentFrame)
  ├─ updateCsmBuffers(currentFrame)
  ├─ updateSsrBuffers(currentFrame)
  ├─ updateDeferredSettingsBuffer(currentFrame)
  │
  ├─ updateCullingBuffers(currentFrame)    ← frustum planes + Hi-Z info
  ├─ updateClusterBuffers(currentFrame)    ← lights + cluster params
  │
  ├─ submitHiZAndCullCommands(currentFrame)  ← 新增：提交 Hi-Z + Occlusion Cull
  │   ├─ waitForFences(cullingFences)
  │   ├─ record(hizBuildCommandBuffer)     ← Hi-Z Build
  │   ├─ record(cullingCommandBuffers)      ← Occlusion Cull
  │   ├─ submit(Hi-Z, signal=hizBuildSemaphore)
  │   ├─ submit(Cull, wait=hizBuildSemaphore, signal=cullCompleteSemaphore)
  │   └─ readbackCullingResults(currentFrame)  ← CPU 统计
  │
  ├─ submitClusterComputeCommand(currentFrame)  ← 新增：提交 Cluster Compute
  │   ├─ waitForFences(clusterFences)
  │   ├─ record(clusterCommandBuffers)
  │   ├─ submit(Cluster, signal=clusterComputeSemaphores[frame])
  │   └─ readbackClusterStats(currentFrame)
  │
  ├─ recordCommandBuffer(imageIndex)
  └─ submit(graphicsQueue, wait=[imageAcquired, cullComplete, clusterSem])
```

### GPU 端命令录制顺序

```
recordCommandBuffer(imageIndex):
  │
  ├─ Pass 0: CSM Depth Pass
  │
  ├─ Pass 1: GBuffer Pass
  │   └─ for each submesh:
  │       if visibility[i] == 1 → drawIndexed
  │
  ├─ Pass 2: SSR Pass
  │
  └─ Pass 3: Deferred Lighting
      └─ Set 0 + Set 1 + Set 2 (cluster lights)
```

### GPU 端异步执行

```
┌─────────────────────────────────────────────┐
│ Frame N-1 的 Hi-Z + Cull                   │
│   └─ hizBuildSemaphore → cullCompleteSemaphore
│                                              │
│ Frame N-1 的 Cluster Compute                 │
│   └─ clusterComputeSemaphores[N-1]          │
│                                              │
│ Frame N 的 Graphics                          │
│   └─ wait(cullCompleteSemaphore, clusterComputeSemaphores[N]) │
└─────────────────────────────────────────────┘
```

---

## 6. 同步对象

| 对象 | 类型 | 用途 |
|------|------|------|
| `hizBuildSemaphore` | Semaphore | Hi-Z Build → Occlusion Cull |
| `cullCompleteSemaphore` | Semaphore | Occlusion Cull → Graphics (GBuffer Pass) |
| `clusterComputeSemaphores[2]` | Semaphore[2] | Cluster Compute → Deferred Lighting |
| `cullingFences[2]` | Fence[2] | CPU 等待 Cull 完成（回读） |
| `clusterFences[2]` | Fence[2] | CPU 等待 Cluster 完成（回读） |

---

## 7. Cluster 光源分配算法

每个 cluster（线程组）处理一个 3D cluster：

1. **计算 cluster 的视空间 AABB**（使用 `fovY` 和 `aspectRatio`）
2. **遍历所有光源**（2048 个）
3. **Sphere-AABB 相交测试**：`lightRadius = sqrt(intensity) * 3.0`
4. **命中则写入 `lightIndexBuffer`**，最多 64 个
5. **`lightGrid[clusterIndex] = {offset=0, count=actualCount}`**

---

## 8. UI 控制

```
Phase 5: GPU Culling
├── Frustum Culling: ☑ (checkbox)
├── Reset Visibility (button)
└── Culled: X / Y sub-meshes

Phase 5: Clustered Shading
├── Clustered Enabled: ☑ (checkbox)
├── Visualize Clusters: ☐ (checkbox)
├── Grid: 16 x 9 x 24 = 3456 clusters
├── Avg Lights/Cluster: N
└── Max Lights/Cluster: 64
```

---

## 9. 性能考量

1. **Hi-Z Build**：每个 mip level 一次 dispatch，O(W×H) 总复杂度
2. **Occlusion Cull**：每个 submesh 一次线程，O(submesh_count)
3. **Cluster Compute**：每个 cluster 一次线程，O(totalClusters × lightCount)，最坏 2048 × 3456 ≈ 7M 次相交测试
4. **回读延迟**：visibility 和 cluster stats 使用 1-frame 延迟，CPU 读取上一帧结果

---

## 10. 已知问题

1. **Hi-Z 在首次渲染时无效**：因为 Hi-Z 需要上一帧的深度来构建，但第一帧没有深度
2. **Cluster 光源分配可能有误**：球-AABB 相交测试使用简化的 view-space 变换，可能不完全准确
3. **Forward 管线未使用 visibility buffer**：GBuffer Pass 过滤了 visibility，但 forward 管线暂时未集成

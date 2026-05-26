# 11_clustered — Clustered Shading（Tiled Lighting）

## 1. 目标渲染效果

将屏幕空间划分为固定大小的 Cluster（平铺 tile + 对数深度轴分割），通过 Compute Shader 在每个 Cluster 内预先计算哪些光源影响该 Cluster，Graphics Pass 在 fragment shader 中只读取影响当前像素的光源列表，实现大量光源（2048 个）的实时渲染。

**渲染画面**：
- 1024 个彩色动态点光源（HSV 全光谱分布），围绕原点运动。
- 1 个地面（50×50 平面）+ 光源代理球体（0.3r）。
- Cluster Grid 分辨率可调（默认 16×9×24），每帧统计平均每 Cluster 光源数。
- 可通过 UI 开关 Clustered Shading 开启/关闭。

---

## 2. 渲染流程总览

```
┌─────────────────────────────────────────────────────────────┐
│               CPU: updateClusterBuffers()                   │
│  ├─ memcpy(sceneUbo)                                       │
│  ├─ 光源动画计算 + memcpy(lightBuffer)                      │
│  ├─ memcpy(clusterParams)                                   │
│  └─ memcpy(groundUbo)                                       │
│                                                              │
│               Compute: Cluster Build                        │
│  dispatch(ceil(X/8), ceil(Y/8), ceil(Z/8))                │
│  → lightGrid[cluster] = {offset, count}                    │
│  → lightIndexBuffer[base+i] = lightIdx                     │
│                                                              │
│               CPU: waitForFences(computeFence)              │
│               CPU: copyBuffer → readback                    │
│               CPU: 统计 avgLightsPerCluster                 │
│                                                              │
│               Graphics: Lighting                            │
│  输入: 地面 + 光源代理球体                                  │
│  frag: getClusterIndex → grid → lightIndex → 光照          │
│  → 输出: SwapChain                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 流程需要用到什么东西，用在哪里

### 3.1 光源数据

| 资源 | 类型 | CPU 访问 | GPU 访问 |
|---|---|---|---|
| `lightBufferResources` | StorageBuffer (SSBO) | 每帧 memcpy (2048 PointLight) | Compute + Graphics |
| `sceneLights[]` | std::vector<PointLight> (CPU) | 动画更新 | — |

**`PointLight`**：

```cpp
struct PointLight {
    glm::vec4 position; // xyz: world pos, w: unused
    glm::vec4 color;    // rgb: HSV→RGB, w: intensity ∈ [5,50]
};
```

光源生成：seed=42，HSV 色彩全光谱分布，位置 X/Z∈[-20,20]，Y∈[0.5,8.0]。

### 3.2 Cluster Grid 数据

| 资源 | 类型 | 大小 | CPU 访问 | GPU 访问 |
|---|---|---|---|---|
| `lightGridBuffer` | StorageBuffer (HostVisible) | totalClusters×2×uint32 | 每帧 readback 统计 | Compute 写，Graphics 读 |
| `lightIndexBuffer` | StorageBuffer (HostVisible) | totalClusters×64×uint32 | — | Compute 写，Graphics 读 |
| `clusterParamsResources` | UniformBuffer | sizeof(ClusterParamsUBO) | 每帧 memcpy | Compute + Graphics |

**`ClusterParamsUBO`**：

```cpp
struct ClusterParamsUBO {
    uint32_t clusterX, clusterY, clusterZ; // 默认 16, 9, 24
    uint32_t totalClusters;                 // clusterX × clusterY × clusterZ
    glm::vec3 tileSize;                    // {2/clusterX, 2/clusterY, 0}
    glm::vec3 cameraPos;
    float farZ = 100.0f;
    float zCount, zMin = 0.1f, zMax = 100.0f;
};
```

**`LightGrid`**（GPU 端结构）：

```cpp
struct LightGrid {
    uint offset; // 固定为 0（本实现）
    uint count; // 该 Cluster 命中光源数
};
```

### 3.3 Descriptor Set Layouts

**Compute Pass** (`computeDescriptorSetLayout`)：

| Binding | Type | 内容 |
|---|---|---|
| b0 | StorageBuffer | `lightBuffer` — 2048 光源 |
| b1 | StorageBuffer (RW) | `lightGridBuffer` — Grid 结果 |
| b2 | StorageBuffer (RW) | `lightIndexBuffer` — 光源索引 |
| b3 | UniformBuffer | `sceneUbo` — 相机矩阵 |
| b4 | UniformBuffer | `clusterParams` — Cluster 参数 |

**Graphics Pass** (`clusteredDescriptorSetLayout`)：

| Binding | Type | 内容 |
|---|---|---|
| b0 | UniformBuffer | `sceneUbo` |
| b1 | StorageBuffer | `lightBuffer` — 光源列表 |
| b2 | StorageBuffer | `lightGridBuffer` — Grid 结果 |
| b3 | UniformBuffer | `clusterParams` |
| b4 | StorageBuffer | `lightIndexBuffer` — 光源索引 |

---

## 4. 东西的初始化过程

### 4.1 调用链

```
prepareResource()
 ├─ generateSphere(sphereMesh, 0.3f, 16)    // 光源代理球体
 ├─ generateCube(cubeMesh)
 ├─ generatePlane(planeMesh, 50×50)        // 地面
 ├─ createVertexBuffer / createIndexBuffer × 3
 │
 ├─ generateSceneLights()                   // 2048 HSV 光源，seed=42
 │
 ├─ createClusterBuffers()
 │   ├─ createUniformBuffers(sceneUboResources)
 │   ├─ createUniformBuffers(clusterParamsResources)
 │   ├─ createUniformBuffers(groundUboResources)
 │   ├─ createStorageBuffers(lightBufferResources) // SSBO 2048 光源
 │   ├─ lightGridBuffer (HostVisible)  // totalClusters×2×uint32
 │   ├─ lightIndexBuffer (HostVisible)  // totalClusters×64×uint32
 │   └─ lightGridReadbackBuffer + readbackMemory
 │
 ├─ createClusteredDescriptorSetLayout()
 ├─ createClusteredDescriptorPool()
 ├─ createClusteredDescriptorSets()
 └─ createClusteredPipeline()
     → cluster_lighting.spv → vertMain/fragMain

 ├─ createComputeDescriptorSetLayout()
 ├─ createComputeDescriptorPool()
 ├─ createComputeDescriptorSets()
 └─ createComputePipeline()
     → cluster_comp.spv → compMain
     → [numthreads(8,8,8)]

 ├─ createClusterCommandPool()  // computeQueue family
 ├─ createClusterCommandBuffers() // swapChainImageCount 个
 ├─ createComputeSyncObjects()
 │   └─ computeCompleteSemaphores[MAX_FRAMES_IN_FLIGHT]

 └─ initUI()
```

### 4.2 Cluster Grid 维度

| 维度 | 默认值 | 可调范围 | totalClusters (默认) |
|---|---|---|---|
| clusterX | 16 | 4~32 | 16×9×24 = 3456 |
| clusterY | 9 | 4~18 | |
| clusterZ | 24 | 4~64 | |

---

## 5. 渲染循环

### 5.1 `render()` 主流程

```
1. device.waitForFences(computeFence)    // CPU 等待 GPU 完成上帧 compute
2. swapChain.acquireNextImage()
3. device.resetFences()
4. updateUIFrame()
5. updateClusterBuffers(currentFrame)
   ├─ sceneUbo: projection / view / camPos / nearZ
   ├─ 光源动画 (sin/cos 运动) + memcpy(lightBuffer)
   ├─ clusterParams: clusterX/Y/Z / tileSize / cameraPos / zMin/zMax
   └─ groundUbo: model / color
6. recordComputeCommandBuffer(currentFrame)  // 录制 compute command
   └─ cmdBuffer: bindPipeline(compute) → dispatch → cmdBuffer.end()
7. computeQueue.submit(signal=computeCompleteSem)
8. recordCommandBuffer(imageIndex)           // 录制 graphics command
9. graphicsQueue.submit(wait=presentCompleteSem+computeCompleteSem)
10. presentQueue.presentKHR()
11. currentFrame = (currentFrame+1)%2
```

### 5.2 Compute Shader 调度与逻辑

**Dispatch**：`dispatch((X+7)/8, (Y+7)/8, (Z+7)/8)` — 每个线程组对应 1 个 Cluster。

**Compute Shader** (`cluster_comp.compMain`)：
```
输入: lightBuffer[i], sceneUbo, clusterParams
输出: lightGrid[clusterIndex] = {offset=0, count}, lightIndexBuffer[base+i] = localIndices[i]

每个线程处理 1 个 Cluster：
1. 由 dispatchThreadID 计算 clusterIndex
2. 调用 getClusterMin/Max() 计算该 Cluster 的 3D AABB（view-space）
   - XY 轴: 基于 NDC screen tile
   - Z 轴: 基于对数深度 [log(zMin), log(zMax)]
3. 遍历所有光源 (lightCount=2048):
   - 估算影响半径: r = sqrt(intensity) × 0.5
   - sphereIntersectsAABB(lightPos, r, clusterMin, clusterMax)
   - 命中则写入 localIndices[]（最多 MAX_LIGHTS_PER_CLUSTER=64）
4. lightGrid[clusterIndex] = {0, actualCount}
5. lightIndexBuffer[base+i] = localIndices[i]
```

### 5.3 `recordCommandBuffer` 完整命令序列

```
cmdBuffer.begin()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Graphics: 地面 + 光源球渲染
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 swapChain → ColorAttachmentOptimal
 depth → DepthAttachmentOptimal
 beginRendering(color + depth, loadOp=CLEAR)

   // 地面
   cmdBuffer.bindPipeline(clusteredPipeline)
   bindDescriptorSets(clusteredDescriptorSets[currentFrame])
   bindVertexBuffers(planeMesh)
   bindIndexBuffer(planeMesh)
   drawIndexed(planeMesh.indices × 1 instance)  // 地面

   // 光源代理球体
   bindVertexBuffers(sphereMesh)
   bindIndexBuffer(sphereMesh)
   drawIndexed(sphereMesh.indices × 2048 instances)  // 2048 光源球

 endRendering()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 UI Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 beginRendering(swapChain, loadOp=LOAD)
   recordUI(cmdBuffer)
 endRendering()

 swapChain → PresentSrcKHR

cmdBuffer.end()
```

### 5.4 Lighting Fragment Shader 逻辑（`cluster_lighting.fragMain`）

```
1. getClusterIndex(viewPos):
   - screenUV = (viewPos.xy + 1) * 0.5
   - tileX = clamp(screenUV.x * clusterX, 0, clusterX-1)
   - tileY = clamp(screenUV.y * clusterY, 0, clusterY-1)
   - logZ = log(max(viewPos.z, zMin))
   - tileZ = clamp((logZ - logMin)/(logMax-logMin) * clusterZ, 0, clusterZ-1)
   - index = tileX + tileY*clusterX + tileZ*clusterX*clusterY

2. LightGrid grid = lightGrid[index]

3. for i in [0, grid.count):
   lightIdx = lightIndexBuffer[grid.offset + i]
   light = lightBuffer[lightIdx]
   color += pointLightShading(light, worldPos, N, V)

4. color = ReinhardToneMap(color + 0.02 ambient)
```

---

## 6. CPU 回读与统计

每帧在 Compute Pass 完成后执行：

```
device.waitForFences(computeFence)  // CPU 阻塞等待

// GPU→CPU copy
beginSingleTimeCommands()
  copyBuffer(lightGridBuffer → lightGridReadbackBuffer)
endSingleTimeCommands()

// CPU 遍历所有 cluster
uint64_t totalLights = 0, nonEmptyClusters = 0
for each cluster:
  count = readbackData[i*2+1]
  totalLights += count
  if count > 0: nonEmptyClusters++

avgLightsPerCluster = totalLights / nonEmptyClusters
```

---

## 7. 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU** | `sceneUboResources` | 每帧 memcpy → UBO |
| **CPU→GPU** | `lightBufferResources` | 每帧 memcpy → SSBO（2048 光源） |
| **CPU→GPU** | `clusterParamsResources` | 每帧 memcpy → UBO |
| **CPU→GPU** | `groundUboResources` | 每帧 memcpy → UBO |
| **GPU写** | `lightGridBuffer` | Compute Shader 原子写入 |
| **GPU写** | `lightIndexBuffer` | Compute Shader 顺序写入 |
| **GPU读** | `lightBuffer` | Compute + Fragment Shader 并行读 |
| **GPU读** | `lightGridBuffer` | Fragment Shader 随机读 |
| **GPU读** | `lightIndexBuffer` | Fragment Shader 顺序读 |
| **GPU→CPU** | `lightGridReadbackBuffer` | `copyBuffer` → CPU 遍历统计 |

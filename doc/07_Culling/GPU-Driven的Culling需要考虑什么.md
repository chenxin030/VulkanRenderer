# GPU-Driven 的 Culling 需要考虑什么

[返回目录](../../README.md)

GPU-Driven 渲染将传统由 CPU 驱动的 DrawCall 提交改为 GPU 自主决定"画什么"。本文从工程实现角度梳理设计一个完整的 GPU-Driven Culling 系统需要考虑的所有问题，涵盖架构、算法、同步、性能和扩展性。

---

## 1. 为什么需要 GPU-Driven

### CPU-Driven 的瓶颈

传统管线中，CPU 负责：

1. 视锥剔除
2. 遮挡剔除（可能借助 GPU GBuffer）
3. 排序（渲染状态）
4. 提交 DrawCall

当场景规模达到数万实例时，CPU 成为瓶颈——即使 GPU 本身能轻松渲染，CPU 在逐实例剔除和 DrawCall 提交上也会卡死。

### GPU-Driven 的核心思想

```
CPU：准备实例数据 → submit（1次或少量批次）
GPU Compute：逐实例 Culling → 输出可见性结果
GPU Graphics：间接绘制 → 只渲染真正可见的实例
```

**关键转变**：CPU 不知道最终画多少个实例，由 GPU 自己决定。

---

## 2. 数据流架构设计

### 2.1 最小必要资源

实现 GPU-Driven Culling 至少需要：


| 资源             | 类型                             | 用途                                         |
| ---------------- | -------------------------------- | -------------------------------------------- |
| 实例数据缓冲区   | StorageBuffer / StructuredBuffer | CPU 写入 GPU 读取，只写一次                  |
| 可见性索引缓冲区 | StorageBuffer（RW）              | GPU Culling 写入可见实例的原始索引           |
| 间接绘制缓冲区   | IndirectBuffer                   | GPU Culling 写入 DrawCommand，Graphics 读取  |
| 可见计数器       | StorageBuffer（单 uint32）       | GPU Culling 原子递增，GPU/Graphcis/_CPU 读取 |


### 2.2 可见性索引

所有实例共享一个 `visibleIndices[]`，一个 `DrawCommand`。

```
优点：实现最简单
缺点：不支持多 mesh 混合（需扩展为 per-mesh DrawCommand）
```

### 2.3 计数器管理

```cpp
// 显式清零，否则上帧残留值会累加
fillBuffer(visibleCountBuffer, 0, sizeof(uint32), 0);

// 同步：fillBuffer 是 Transfer stage，culling 是 Compute stage
bufferMemoryBarrier(
    srcStage = Transfer,
    dstStage = ComputeShader,
    srcAccess = TransferWrite,
    dstAccess = ShaderRead | ShaderWrite
);
```

**原子操作 vs. 非原子操作**：

```slang
// 错误：如果两个线程同时写入同一个位置，会丢失数据
visibleIndices[globalCounter] = instanceIndex;
globalCounter++;

// 正确：原子加法保证安全
uint dstIndex;
InterlockedAdd(counterBuffer[0], 1, dstIndex);
visibleIndices[dstIndex] = instanceIndex;
```

---

## 3. Culling 阶段设计

### 3.1 视锥剔除（Frustum Culling）

对每个实例的包围球 / 包围盒做 6 个视锥平面测试。

```slang
bool sphereInsidePlane(float3 center, float radius, float4 plane) {
    return dot(plane.xyz, center) + plane.w + radius >= 0.0;
}

bool sphereInsideFrustum(float3 center, float radius) {
    for (int i = 0; i < 6; ++i) {
        if (!sphereInsidePlane(center, radius, frustumPlanes[i]))
            return false;
    }
    return true;
}
```

**注意事项**：

- 视锥平面从 VP 矩阵提取后必须归一化

平面的数学表达是 `ax + by + cz + d = 0`，向量 `(a, b, c)` 是法线方向。从矩阵提取出来的 `(a, b, c)` **长度不是 1**，导致 `dot(plane.xyz, center)` 计算出的投影值不是真实距离。归一化就是 `plane /= length(plane.xyz)`，让法线长度变为 1，确保 `dot(plane.xyz, center) + plane.w + radius` 才是正确的距离比较。如果不归一化：靠近相机的大物体会被误判在视锥外，远距离的小物体会被漏判。

- 包围球半径估算要包含物体旋转后的最大范围

### 3.2 遮挡剔除（Occlusion Culling）

#### Hi-Z（层次 Z-Buffer）

**核心思想**：用低分辨率的深度图近似场景遮挡关系，在粗粒度上做剔除。

```
mip0: 1920×1080 全分辨率深度
mip1: 960×540   每像素代表上一级 2×2 区域的最大深度
mip2: 480×270   ...
...
```

**保守性保证**：取 2×2 区域内的**最大值**（NDC 空间下越大 = 离相机越近），这样被剔除的物体一定是真正被遮挡的。

**动态 mip 选择**：

```slang
// 物体越大，采样粗糙 mip；物体越小，采样精细 mip
float radiusPx = radius * viewportWidth / clipW;
float mip = clamp(floor(log2(radiusPx)), 0.0, maxMipLevel);
float depthInHiZ = hiZTexture.SampleLevel(sampler, uv, mip);
```

#### Depth Prepass 的必要性

Hi-Z 的精度完全取决于 prepass 绘制的深度图质量。

- 如果 prepass 没有绘制某个物体，该物体"后面的"物体就无法被正确遮挡
- prepass 必须绘制**所有可能遮挡其他物体的几何体**
- 静态物体和动态物体需要分别处理（静态可以用 Hi-Z 回读缓存，动态每帧重建）

### 3.3 距离剔除（Distance Culling）

最简单但效果显著：

```slang
float distSq = dot(worldPos - cameraPos, worldPos - cameraPos);
if (distSq > maxCullDistSq) return false;
```

适合与 LOD 结合——远处物体本来就切换到低精度 mesh，culling 效率更高。

### 3.4 背面剔除（Backface Culling）

Vulkan/D3D12 可以依赖光栅化器的背面剔除，但如果想做更精细的控制：

```slang
// 在 vertex shader 或 compute shader 中计算
float3 faceNormal = cross(v1 - v0, v2 - v0);
if (dot(faceNormal, toCamera) < 0) return false;
```

**注意**：GPU-Driven 中，提前在 compute shader 里做背面剔除可以节省深度 prepass 的带宽，但会增加 compute 负载。需要 profile 权衡。

### 3.5 屏幕空间尺寸剔除（Small Object Culling）

当物体投影到屏幕上的像素面积过小时，跳过它的绘制：

```slang
float screenArea = computeScreenSpaceArea(instanceBounds, viewProj);
if (screenArea < minPixelsThreshold) return false;
```

常用于粒子系统或植被密度很高的场景。

### 3.6 推荐的 Culling 顺序

从最廉价到最昂贵的测试依次排列：

```
1. 距离剔除（仅一次减法比较）
2. 视锥剔除（6 次点积）
3. 背面剔除（可选，3 次叉积）
4. 遮挡剔除（Hi-Z，最贵但效果最显著）
5. 屏幕尺寸剔除（可选）
```

实际实现中常将 1+2 合并在一次 pass 中完成（Frustum + Distance），遮挡剔除单独做。

---

## 4. 同步与 Barriers

### 4.1 GPU 内部同步（Pass 间）

GPU-Driven 有多层 Pass，每层之间必须显式同步：

```
Depth Prepass → Hi-Z Build → Culling Compute → Draw
   ↓              ↓              ↓
 写入深度图     写入Hi-Z       写入可见性
```

**典型 barrier 序列**：


| 位置                 | 转换                     | Stage                   | Access                        |
| -------------------- | ------------------------ | ----------------------- | ----------------------------- |
| Depth → Hi-Z         | DepthReadOnly → General  | LateFragTests → Compute | DepthAttWrite → ShaderSampled |
| Hi-Z mip N → N+1     | General → General        | Compute → Compute       | ShaderWrite → ShaderRead      |
| Hi-Z → Culling       | General → ShaderReadOnly | Compute → Compute       | ShaderWrite → ShaderSampled   |
| fillBuffer → Culling | Transfer → Compute       | Transfer → Compute      | TransferWrite → ShaderRead    |


### 4.2 GPU → CPU 回读同步

如果 UI 需要显示统计信息（如可见实例数、剔除率），必须在 GPU 写入后做回读：

```cpp
// 方法 A：使用 Fence（推荐）
computeQueue.submit(signalFence=fence);
device.waitForFences(fence);          // CPU 等待 GPU 完成
copyBuffer(statsBuffer → readback);   // 回读
readFromHostMemory();                 // CPU 读取

// 方法 B：使用 Semaphore（只用于 GPU-GPU，不适用于 CPU 读）
computeQueue.submit(signalSemaphore=sem);
graphicsQueue.submit(waitSemaphore=sem);  // 只能让另一个 GPU queue 等
```

**重要**：回读的数据永远是"已经执行完的那一帧"，存在一帧延迟。如果需要当前帧的实时数据，必须在 Graphics 完成后再读（代价是增加一次 fence 等待）。

### 4.3 Counter Reset 的时序

```
❌ 错误做法：在 Culling Compute 之后才 fillBuffer
  Culling: InterlockedAdd(counter, 1)  // 用的是上上帧的旧值（0），没问题
            ↓
  fillBuffer: counter = 0             // 清零，但 Culling 已经结束了

✅ 正确做法：在 Culling Compute 之前 fillBuffer
  fillBuffer: counter = 0
              ↓
  barrier: Transfer → Compute
              ↓
  Culling: InterlockedAdd(counter, 1)  // 清零后的干净状态
```

---

## 5. 内存与存储布局

### 5.1 Buffer 大小估算

```cpp
struct InstanceData {
    glm::mat4 model;    // 64 bytes
    glm::vec4 color;    // 16 bytes
    // 可选扩展：
    // glm::mat4 prevModel; // TAA 用，64 bytes
    // uint32_t flags;      // culling 状态标记，4 bytes
    // uint16_t meshId;     // 多 mesh 时，2 bytes
    // uint16_t lodLevel;   // LOD 等级，2 bytes
};
// 每实例 ≈ 80 bytes（不含扩展）

// 32400 实例：
//   instanceBuffer = 32400 × 80 = 2.5 MB
//   visibleIndices = 32400 × 4 = 126 KB
//   完全可以放在 L2/LLC Cache 中
```

### 5.2 内存类型选择


| 用途                 | 内存类型      | 原因                 |
| -------------------- | ------------- | -------------------- |
| 实例数据（CPU→GPU）  | `DeviceLocal` | GPU 频繁读取，带宽高 |
| 可见性索引（GPU 写） | `DeviceLocal` | GPU 原子操作需要     |
| 间接绘制命令         | `DeviceLocal` | Graphics 直接读取    |
| 统计回读             | `HostVisible  | HostCoherent`        | CPU 读取，延迟可接受 |


### 5.3 Double / Triple Buffering

GPU-Driven 必须对所有 GPU 资源做 double buffering（帧 N 和帧 N+1 不能共用同一块 GPU 内存）：

- `instanceBuffer`：`Buffers[currentFrame]`（`MAX_FRAMES_IN_FLIGHT`）
- `visibleIndices`：`Buffers[currentFrame]`
- `drawCommands`：`Buffers[currentFrame]`

**唯一例外**：`visibleCountBuffer`（单值计数器）可以共用一块内存，因为所有帧都向同一个地址原子写入。但清零操作也需要分帧。

---

## 6. 性能考量

### 6.1 Workgroup 大小选择

```
culling_comp: numthreads(64, 1, 1)
Hi-Z Build:   numthreads(8, 8, 1)
```

- `64` 的倍数通常匹配 GPU 的 wave/warp 大小
- Hi-Z 用 2D workgroup 是因为每线程对应一个目标像素
- `64` threads × `ceil(32400/64) = 507` 个 workgroups，对于现代 GPU 来说很容易跑满

### 6.2 原子操作竞争

`InterlockedAdd` 是性能敏感操作。当大量线程同时竞争同一个 counter 时，atomic wait 会成为瓶颈：

**优化方向**：

- 使用**分桶计数**（多个 counter，减少竞争）
- 对计数器使用 `Subgroup` 级别的 reduce（如果有 API 支持）
- 对 `visibleIndices` 写入地址做分块（每个 workgroup 一段区间）

```slang
// 分桶计数示例：4 个 bucket
uint bucketId = (index / 256) % 4;
InterlockedAdd(counterBuckets[bucketId], 1, localIndex);
visibleIndices[baseOffset[bucketId] + localIndex] = instanceIndex;
```

### 6.3 带宽预算


| 操作             | 带宽消耗     | 备注                                          |
| ---------------- | ------------ | --------------------------------------------- |
| Depth Prepass    | 高           | 所有实例所有三角形的顶点着色                  |
| Hi-Z Build       | 低           | 只有深度图大小的读写                          |
| Culling Compute  | 中           | 逐实例读取 instanceBuffer（~2.5MB/帧）        |
| Draw（可见实例） | 取决于剔除率 | 100% 可见时 = 普通渲染；90% 剔除时 ≈ 10% 带宽 |


### 6.4 占用率 vs. 效率

Compute Shader 的 occupancy（占用率）不等于效率：

- 高占用率 = 更多线程并行执行
- 但如果每线程工作量很小（只有一个 float4 比较），atomic 操作会成为瓶颈

profile 工具（如 RenderDoc、NVIDIA Nsight、AMD Radeon GPU Profiler）可以显示实际的 occupancy 和 stall 原因。

---

## 7. 多 Mesh 支持

### 7.1 基本思路

当前项目只渲染一种 cube。如果要支持多 mesh：

```
每帧 CPU 准备：
  - meshA 的 instanceBuffer_A（model + color）
  - meshB 的 instanceBuffer_B
  - meshC 的 instanceBuffer_C

GPU Culling 后：
  - drawCmd_A.instanceCount → meshA 可见数
  - drawCmd_B.instanceCount → meshB 可见数
  - drawCmd_C.instanceCount → meshC 可见数
```

### 7.2 实现方式

#### 方式 1：独立的 Per-Mesh Culling（简单但昂贵）

每个 mesh 有独立的 instance buffer + culling compute dispatch。
当有 N 种 mesh 时，N 次 dispatch 开销不可忽视。

#### 方式 2：合并实例批次（推荐）

把所有 mesh 的 instance data 拼在一个大 buffer 里：

```
[MeshA_instances][MeshB_instances][MeshC_instances]
```

Culling 时一次 dispatch 遍历所有实例，跳过不属于当前 mesh 的索引。但 `drawIndexedIndirect` 需要按 mesh 分开调用。

#### 方式 3：Clustered（高级）

按空间 cluster 分组，每个 cluster 包含来自不同 mesh 的实例。
适合开放世界场景（UE5 Nanite 思路）。

---

## 8. 调试与验证

### 8.1 常见 Bug 模式


| 症状                         | 可能原因                             |
| ---------------------------- | ------------------------------------ |
| 每帧实例数递增               | counter 没有在 culling 前清零        |
| 第一次运行正常，后续帧异常   | 缺少 double buffering                |
| 统计数量总是上一帧的值       | 回读时机不对（等待的是错误的 fence） |
| GPU crash / validation error | barrier 缺失或 stage/access 不匹配   |
| 可见性跳变（闪烁）           | atomic 竞争导致 counter 不稳定       |


### 8.2 验证手段

**1. GPU Capture（RenderDoc / Nsight）**

- 检查每帧的 DrawCall 数量和参数
- 验证 barrier 前后的 Image/Buffer 状态
- 查看 Compute 的 dispatch count

**2. Counter 一致性验证**

```cpp
// 在 Draw 之前加一个 compute pass 输出 debug 信息
// 比较 visibleIndices 中实际写入的数量 vs. visibleCountBuffer 的值
if (atomicAdd(debugCounter, 0) != visibleCountBuffer[0]) {
    // 两者不一致，说明 atomic 有问题
}
```

**3. 强制全可见模式**

```slang
if (params.forceAllVisible == 1) {
    visibleIndices[index] = index;
    InterlockedAdd(counter[0], 1);
    return;
}
```

与关闭 culling 对比，验证剔除逻辑的正确性。

---

## 9. 扩展方向

### 9.1 Temporal 扩展

- **Temporal Reprojection**：利用上一帧的深度/可见性做预测，减少每帧需要处理的新实例
- **历史遮挡信息**：上一帧被遮挡的物体，可以复用其 Hi-Z 信息（适用于缓慢移动的相机）

### 9.2 LOD 集成

GPU-Driven 天然适合 LOD：

- 每帧 CPU 根据相机距离决定各实例的 LOD 等级，写入 instanceBuffer
- Culling 时按 LOD 调整包围球半径
- Draw 时 `drawIndexedIndirect` + instance buffer 切 mesh 本身需要额外机制

### 9.3 动态 vs. 静态分离

- **静态物体**（建筑、树木）：预计算可见性，每帧只需读取
- **动态物体**（角色、载具）：每帧 Culling
- **静止但相机移动**：可以用 Hi-Z 做遮挡，不需要每帧 prepass

### 9.4 Clustered Shading 联动

本项目的 Clustered Shading（`RENDERING_LEVEL==12`）在光源很多时有巨大优势。可以将 Clustered 的 tile-grid 信息传给 Culling Compute：

- 对不在任何 cluster 光源范围内的实例提前剔除
- 属于两个系统的"联合裁剪"

### 9.5 可见性索引的组织方式

#### 方式 A：全局扁平索引（本项目当前方式）

所有实例共享一个 `visibleIndices[]`，一个 `DrawCommand`。

实现最简单，适合单 mesh 场景。但不支持多 mesh 混合（需扩展为 per-mesh DrawCommand）。
```
visibleIndices[0..N-1]: 所有可见实例的索引
drawCommands[0]: {instanceCount = N}
```

#### 方式 B：Per-Mesh DrawCommand

每个 mesh 类型有自己的 DrawCommand 和可见性缓冲区。支持多 mesh 类型，每种 mesh 独立可见性。mesh 数量增加时，需要 N 次 `drawIndexedIndirect` 调用。

```
MeshA: drawCmdA {instanceCount}, visibleIndicesA[0..Na-1]
MeshB: drawCmdB {instanceCount}, visibleIndicesB[0..Nb-1]
MeshC: drawCmdC {instanceCount}, visibleIndicesC[0..Nc-1]
...
```

#### 方式 C：Clustered

按空间 cluster 分组，每个 cluster 包含来自不同 mesh 的实例。适合开放世界场景。

### 9.6 视锥剔除的精确化

#### 包围球 vs. OBB（方向包围盒）

当前实现使用**立方体 AABB** 近似包围球：

```cpp
// 立方体 [-0.5, 0.5]^3 的半径
float3 extents = 0.5 * (aabbMax.xyz - aabbMin.xyz);
float radius = length(extents);  // ≈ 0.866
```

立方体的对角线 = `√3 × 0.5 ≈ 0.866`，作为球体半径能包住任意旋转。但对于细长物体（飞船、建筑），球体远大于实际物体，culling 不够精确。

**OBB（Oriented Bounding Box）** 的优势：

| 类型   | 轴对齐            | 贴合度               | 测试成本           |
| ------ | ----------------- | -------------------- | ------------------ |
| 包围球 | 球形，旋转对称    | 差（细长物体浪费大） | 低（1 次半径比较） |
| AABB   | 固定与 XYZ 轴对齐 | 中                   | 中（6 次点积）     |
| OBB    | 跟随物体旋转      | 好                   | 高（需 3 轴投影）  |

OBB 的视锥测试公式（比球体更精确但更慢）：

```slang
// 对每个视锥平面，需要考虑 OBB 的 3 个轴各自贡献的投影量
float r = extents[0] * abs(dot(frustumPlane.xyz, axes[0]))
        + extents[1] * abs(dot(frustumPlane.xyz, axes[1]))
        + extents[2] * abs(dot(frustumPlane.xyz, axes[2]));
if (dot(frustumPlane.xyz, center) + frustumPlane.w + r < 0.0)
    return false;  // 在视锥外
```

实践中的常见策略：**两阶段剔除**
1. 用包围球做快速预剔除（大多数物体在这一步被丢弃）
2. 对包围球命中的物体再用 OBB 做精确剔除（极少数需要更精确判断的物体）

---


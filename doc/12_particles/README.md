# 12_particles — GPU 粒子系统

## 1. 目标渲染效果

粒子系统在 CPU 端初始化后在 GPU 上通过 Compute Shader 完成全部更新（位置+速度+生命周期），Graphics Pass 使用顶点着色器中的 GPU 读取来渲染粒子，无需 CPU 回读，实现 100,000 粒子的实时模拟。

**渲染画面**：
- 发射器位于原点，向上发射橙红色粒子流（火焰/喷泉效果）。
- 粒子受重力和湍流扰动影响，产生自然的抛物线轨迹。
- Alpha 混合使粒子具有半透明发光感。
- UI 可调节：发射率、粒子生命周期、大小、重力、湍流强度、扩散角、初始速度。

---

## 2. 渲染流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU: updateParticleBuffers()             │
│  ├─ memcpy(sceneUbo)   // 相机矩阵                          │
│  └─ memcpy(particleParamsUbo)  // emitter/gravity/rate...  │
│                                                              │
│                    Compute: Particle Update                  │
│  dispatch(ceil(MAX_PARTICLES/64), 1, 1)                    │
│  → 更新每个粒子的 position / velocity / lifetime / color     │
│                                                              │
│                    Graphics: Particle Render                 │
│  输入: 无 VertexBuffer（粒子数据全部从 StorageBuffer 读取）  │
│  → draw(MAX_PARTICLES=100000)                              │
│  → SwapChain                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 流程需要用到什么东西，用在哪里

### 3.1 粒子数据结构

**CPU 端**（`Particle`）：

```cpp
struct Particle {
    glm::vec3 position;   // 世界坐标
    float    lifetime;    // 当前剩余生命
    glm::vec3 velocity;  // 速度向量
    float    maxLifetime; // 最大生命周期
    glm::vec4 color;    // RGBA 颜色
    float    size;      // 粒子尺寸
    glm::vec3 padding;  // 对齐填充
};  // 48 bytes

static_assert(sizeof(Particle) == 48, "Particle must be 48 bytes");
```

### 3.2 Compute Shader 端结构（`ParticleComp`）

```cpp
struct ParticleComp {
    float4 position_lifetime;  // xyz=position, w=lifetime
    float4 velocity_maxLifetime;
    float4 color;
    float  size;
};  // 48 bytes，与 CPU 端布局一致
```

### 3.3 参数 UBO

```cpp
struct ParticleParamsUBO {
    glm::vec3 emitterPosition;  // 发射器位置 (0,0,0)
    float    emissionRate;      // 发射率 (5000/s)
    glm::vec3 gravity;         // 重力 (0, -9.8, 0)
    float    particleLifetime; // 生命周期 (3.0s)
    float    particleSize;     // 粒子尺寸 (0.15)
    float    turbulenceStrength; // 湍流强度 (0.5)
    float    spread;          // 扩散角 (1.5)
    float    velocity;       // 初始速度 (5.0)
};
```

### 3.4 Descriptor Sets

**Compute Pass** (`computeDescriptorSetLayout`)：

| Binding | Type | 内容 |
|---|---|---|
| b0 | StorageBuffer | `particleBuffer` — 粒子数据（RW） |
| b1 | UniformBuffer | `particleParamsBuffer` — 参数 |

**Graphics Pass** (`particleDescriptorSetLayout`)：

| Binding | Type | 内容 |
|---|---|---|
| b0 | UniformBuffer | `sceneUboResources` — VP 矩阵 |
| b1 | StorageBuffer | `particleBuffer` — 粒子数据（RO） |

---

## 4. 东西的初始化过程

### 4.1 调用链

```
prepareResource()
 ├─ createParticleBuffers()
 │   ├─ createStorageBuffers(particleBufferResources)
 │   │   → 100,000 × sizeof(Particle) = ~4.8MB
 │   ├─ createUniformBuffers(particleParamsBufferResources)
 │   └─ createUniformBuffers(sceneUboResources)
 │
 ├─ createParticleDescriptorSetLayouts()
 │   ├─ particleDescriptorSetLayout (b0:UBO, b1:Storage)
 │   └─ computeDescriptorSetLayout (b0:Storage, b1:UBO)
 │
 ├─ createParticleDescriptorPools()
 ├─ createParticleDescriptorSets()
 │
 ├─ createParticlePipeline()
 │   → particle_render.spv → vertMain/fragMain
 │   → 输入: 无 VertexBuffer
 │   → 输出: SwapChain (depth test + alpha blend)
 │
 ├─ createComputePipeline()
 │   → particle_update_comp.spv → compMain
 │   → [numthreads(64, 1, 1)]
 │
 ├─ createComputeCommandPool()  // computeQueue family
 ├─ createComputeCommandBuffers()
 ├─ createComputeSyncObjects()
 │   └─ computeCompleteSemaphores[MAX_FRAMES_IN_FLIGHT]
 │
 ├─ initUI()
 │
 └─ generateInitialParticles()
     → 所有 100,000 粒子初始化，随机 lifetime/velocity/color
```

### 4.2 初始粒子生成

```cpp
generateInitialParticles():
  for i in [0, MAX_PARTICLES):
    particle.position = (0, 0, 0)
    particle.velocity = (dist11*rng * spread,
                         velocity + dist01*rng * velocity,
                         dist11*rng * spread)
    particle.lifetime = dist01 * particleLifetime  // 随机初始生命
    particle.maxLifetime = particleLifetime
    particle.color = (1, 0.5, 0.1, 1)  // 橙红色
    particle.size = particleSize

  // 同步到所有帧的 buffer
  for frame in [0, MAX_FRAMES_IN_FLIGHT):
    memcpy(buffer[frame], buffer[0], MAX_PARTICLES × sizeof(Particle))
```

---

## 5. 渲染循环

### 5.1 `render()` 主流程

```
1. device.waitForFences(inFlightFences[currentFrame])
2. swapChain.acquireNextImage()
3. device.resetFences()
4. updateParticleBuffers(currentFrame)
   ├─ sceneUbo: projection / view / camPos
   └─ particleParamsUbo: emitterPos / gravity / emissionRate / ...
5. recordComputeCommandBuffer(currentFrame)
   └─ cmdBuffer: bindPipeline(compute) → dispatch → end()
6. computeQueue.submit(signal=computeCompleteSem)  // async，不等

7. commandBuffers[currentFrame].reset()
8. recordCommandBuffer(imageIndex)
9. graphicsQueue.submit(wait={presentCompleteSem, computeCompleteSem})
10. presentQueue.presentKHR()
11. currentFrame = (currentFrame+1)%2
```

### 5.2 Compute Shader 调度与逻辑

**Dispatch**：`dispatch(ceil(100000/64), 1, 1)` — 1563 个线程组，每组 64 个线程。

**Compute Shader** (`particle_update_comp.compMain`)：
```
1. 读取当前粒子数据: particleBuffer[globalInvocationID]
2. 更新生命周期:
   particle.lifetime -= deltaTime
   if (particle.lifetime <= 0):
     // 重置粒子
     particle.position = emitterPosition + small random offset
     particle.velocity = normalize(random3D) × velocity + emitterUpBias
     particle.lifetime = maxLifetime
     // color 基于初始生命重新设置
3. 应用物理:
   particle.velocity += gravity × deltaTime
   // 添加湍流: 随机扰动
   particle.velocity += noise × turbulenceStrength × deltaTime
   particle.position += particle.velocity × deltaTime
4. 写回: particleBuffer[globalInvocationID]
```

### 5.3 `recordCommandBuffer` 完整命令序列

```
cmdBuffer.begin()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Graphics: 粒子渲染
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 swapChain → ColorAttachmentOptimal
 depth → DepthAttachmentOptimal
 beginRendering(color + depth, loadOp=CLEAR)

   cmdBuffer.setViewport(0, fullExtent)
   cmdBuffer.setScissor(0, fullExtent)

   cmdBuffer.bindPipeline(particlePipeline)
   cmdBuffer.bindDescriptorSets(particleDescriptorSets[currentFrame])
   cmdBuffer.draw(MAX_PARTICLES=100000, 1)

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

### 5.4 粒子渲染 Vertex Shader 逻辑（`particle_render.vertMain`）

```
// 无顶点缓冲区，完全由 globalVertexID 索引 StorageBuffer
uint idx = globalVertexID / 3  // 每粒子 3 个顶点（point sprite）
ParticleComp p = particleBuffer[idx]

// point sprite：将其扩展为三角形对（光栅器自动处理）
// 或直接输出 gl_PointSize（opengl legacy）

// 输出 position → NDC
output.Position = projection × view × float4(p.position.xyz, 1.0)
// 输出 color / size 供 fragment shader 使用
```

### 5.5 粒子渲染 Fragment Shader 逻辑（`particle_render.fragMain`）

```
// 读取顶点着色器传递的 color / size
// Point sprite 形状: gl_PointCoord ∈ [0,1]²
// 圆形: length(gl_PointCoord - 0.5) < 0.5
// 越接近中心 alpha 越大，产生发光感

float2 uv = gl_PointCoord - 0.5
if (length(uv) > 0.5) discard
float alpha = 1.0 - smoothstep(0.0, 0.5, length(uv)) * p.color.a
return float4(p.color.rgb, alpha)
```

---

## 6. 粒子渲染 Pipeline 关键配置

```
particlePipeline:
  topology: ePointList                    // 点列表
  vertexInput: stride=0 (无顶点缓冲)
  depthTest: Yes, depthWrite: Yes
  colorBlend: Alpha混合
    srcBlend: eSrcAlpha
    dstBlend: eOne (Additive blending 效果)
    alphaSrc: eOne
    alphaDst: eOneMinusSrcAlpha
```

---

## 7. 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU** | `sceneUboResources` | 每帧 memcpy → UBO（相机矩阵） |
| **CPU→GPU** | `particleParamsBufferResources` | 每帧 memcpy → UBO（参数） |
| **GPU读** | `particleParamsBuffer` | Compute shader uniform |
| **GPU读写** | `particleBuffer` | Compute: 读旧值→计算→写回 |
| **GPU读** | `particleBuffer` | Graphics: vertex shader 按 vertexID 读取 |
| **GPU写** | SwapChain | Particle frag shader 输出 |

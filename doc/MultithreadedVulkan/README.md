# Vulkan 多线程渲染项目设计文档（`15_multithreaded`）

[返回目录](../../README.md)

本文参考 `doc-flow-12-clustered.mdc` 的文档组织方式，给出一个可直接落地的新项目方案：
- 语言：`C++20`
- 图形 API：`Vulkan 1.3`
- 目标：利用 Vulkan 的多线程命令录制与异步队列能力，提高复杂场景的 CPU/GPU 并行度

---

## 1. 项目目标与预期效果

## 1.1 核心目标

1. **CPU 多线程录制命令**
   - 每帧将场景按渲染批次拆分到多个 Worker 线程。
   - 每个线程录制独立 Secondary Command Buffer，主线程只负责拼装与提交。

2. **GPU 异步并行执行**
   - Graphics Queue：主光栅渲染。
   - Compute Queue：粒子更新、可见性裁剪（如 frustum/cluster culling）。
   - Transfer Queue（可选）：异步资源上传。

3. **可量化的性能对比**
   - 支持 UI 一键切换单线程/多线程模式。
   - 输出 CPU 录制时间、GPU 各 Pass 时间、帧时间和 FPS。

## 1.2 预期画面

- 一个中型到大型压力场景（见第 6 节测试场景）：
  - 大量静态网格（建筑/岩石/地形块）
  - 大量动态实例（角色或小物件）
  - 粒子系统（烟雾/火花）
  - 阴影 + PBR + 后处理（Bloom/TAA 可选）

---

## 2. 模块划分（代码结构建议）

建议新增模块目录：

- `src/multithreaded/`
  - `MultithreadedRenderer.h/.cpp`：总控渲染器
  - `FrameGraph.h/.cpp`：Pass 调度与依赖
  - `ThreadPool.h/.cpp`：CPU 线程池
  - `RenderBatcher.h/.cpp`：可见对象分桶与任务切分
  - `GpuProfiler.h/.cpp`：时间戳查询与统计
  - `StandaloneMain.cpp`：独立入口

- `shaders/15_multithreaded/`
  - `gbuffer.slang`
  - `lighting.slang`
  - `particle_update_comp.slang`
  - `particle_render.slang`
  - `postfx.slang`

---

## 3. 初始化阶段（一次性资源）

## 3.1 Vulkan 设备与队列

- `initVulkan()` 创建：
  - `device`
  - `swapChain`
  - `graphicsQueue`
  - `computeQueue`
  - `presentQueue`
  - `transferQueue`（若硬件支持独立队列族）

## 3.2 多线程命令池

每帧、每线程独立命令池，避免锁竞争：

- `graphicsCmdPools[frame][thread]`
- `computeCmdPools[frame][thread]`（可选）

每个 Worker 使用自己的 `VkCommandPool` 分配 Secondary CB：
- `secondaryOpaqueCB[frame][thread]`
- `secondaryTransparentCB[frame][thread]`
- `secondaryShadowCB[frame][thread]`

## 3.3 关键缓冲区

| 资源 | 类型 | 用途 | 更新方式 |
|---|---|---|---|
| `sceneUbo` | UniformBuffer | 相机、全局参数 | 每帧 CPU 写 |
| `objectBuffer` | StorageBuffer | 实例变换、材质索引 | 每帧/按需更新 |
| `drawIndirectBuffer` | Storage/Indirect | MultiDrawIndirect 参数 | Compute 写，Graphics 读 |
| `particleBuffer` | StorageBuffer | 粒子状态 | Compute 读写 |
| `visibleListBuffer` | StorageBuffer | 可见实例列表 | Compute 写，Graphics 读 |

## 3.4 Pipeline

- Graphics:
  - `shadowPipeline`
  - `gbufferPipeline` 或 `forwardPipeline`
  - `lightingPipeline`
  - `particlePipeline`
  - `postfxPipeline`
- Compute:
  - `cullingPipeline`
  - `particleUpdatePipeline`

## 3.5 同步对象

- 每帧资源：
  - `imageAvailableSemaphore[frame]`
  - `computeFinishedSemaphore[frame]`
  - `renderFinishedSemaphore[frame]`
  - `inFlightFence[frame]`
- 线程级同步：
  - `std::barrier` 或 `latch`（等待所有 Worker 完成命令录制）

---

## 4. 每帧执行顺序与数据流

## 4.1 CPU 侧（主线程）

1. 等待 `inFlightFence[currentFrame]`
2. 更新帧常量（相机、时间、UI 参数）
3. 提交 Compute（粒子更新 + 可见性裁剪）
4. 将渲染任务切分为 N 份，分发给线程池
5. 等待 Worker 线程录制完成
6. 主线程录制 Primary CB，`vkCmdExecuteCommands` 拼接 Secondary CB
7. 提交 Graphics Queue（等待 Compute 完成信号量）
8. Present

## 4.2 CPU 侧（Worker 线程）

每个线程处理一个任务分片：

1. Reset 本线程本帧命令池
2. 录制 Secondary CB（仅本分片可见对象）
3. 按材质/管线排序后发出 `vkCmdBindPipeline + vkCmdDrawIndexed`（或 indirect）
4. 上报统计（draw call 数、录制耗时）

## 4.3 GPU 侧

1. Compute Queue：
   - `particleUpdateComp`
   - `cullingComp` 写 `visibleListBuffer` / `drawIndirectBuffer`
2. Graphics Queue：
   - Shadow Pass
   - GBuffer/Forward Pass
   - Lighting Pass
   - Particle Pass
   - PostFX + UI

---

## 5. 线程与同步设计要点

## 5.1 CPU 多线程原则

1. **无共享写热点**：每线程独立命令池、线性分配临时内存。
2. **任务颗粒度可控**：按可见对象数动态切分（避免线程负载不均）。
3. **主线程轻量化**：主线程不参与重度 draw 录制，专注编排和提交。

## 5.2 GPU 同步原则

1. Compute 写入 `drawIndirectBuffer` 后，需插入 Buffer Memory Barrier。
2. Graphics 读取前等待 `computeFinishedSemaphore`。
3. 使用 Timeline Semaphore（推荐）可减少二进制信号量管理复杂度。

## 5.3 帧资源隔离

- `MAX_FRAMES_IN_FLIGHT = 2~3`
- 每帧独立：UBO/描述符/命令池/查询池
- 禁止跨帧复用未完成资源

---

## 6. 测试场景设计（性能压力场景）

## 6.1 场景名称

`MT_StressCity`

## 6.2 场景构成

1. **静态建筑实例**：20,000 ~ 50,000
2. **动态单位实例**：5,000（带简单骨骼或变换动画）
3. **粒子系统**：200,000 粒子（Compute 更新）
4. **光源**：
   - 点光 512
   - 聚光 64
   - 1 个方向光（阴影）

## 6.3 摄像机轨迹

- 固定巡航路径（60 秒循环），覆盖：
  - 俯视大范围
  - 近景穿行（高 overdraw）
  - 光源密集区

## 6.4 测试维度

1. **线程数缩放**：1 / 2 / 4 / 8 / 12
2. **对象规模缩放**：10k / 25k / 50k
3. **特效开关**：粒子开关、阴影级联数量、后处理等级
4. **模式对比**：
   - 单线程录制（基线）
   - 多线程录制（目标）

## 6.5 关键指标

- CPU：
  - `recordCmd(ms)`
  - `update(ms)`
  - `main thread idle(ms)`
- GPU：
  - `compute(ms)`
  - `shadow(ms)`
  - `geometry(ms)`
  - `lighting(ms)`
  - `postfx(ms)`
- Overall：
  - `frame time(ms)`
  - `FPS`
  - `draw calls`
  - `triangles`

---

## 7. UI 调试面板设计

窗口：`Multithreaded Vulkan Profiler`

- `Enable Multi-thread Recording`（开关）
- `Worker Threads`（1~16）
- `Enable Async Compute`（开关）
- `Scene Scale`（Small/Medium/Large）
- `Particle Count`（滑条）
- `CPU Record Time`（曲线）
- `GPU Pass Timing`（分段柱状）
- `FPS` 与 `Frame Time`

---

## 8. 验收标准

1. 在 `MT_StressCity-Large` 下：
   - 多线程模式相较单线程模式，`CPU record time` 降低 **30%+**（目标值，可按硬件调整）。
2. 开启异步 Compute 后：
   - 粒子更新与主渲染有可观重叠，`GPU frame` 总时长下降或持平。
3. 视觉一致性：
   - 单线程与多线程输出在可接受误差内一致（不出现闪烁、丢物体、同步错误）。
4. 稳定性：
   - 连续运行 30 分钟无崩溃、无 Validation Error（开启 Vulkan Validation Layer）。

---

## 9. 开发阶段里程碑

1. **M1（基础并行）**
   - 线程池 + Secondary CB 并行录制跑通。
2. **M2（异步计算）**
   - 粒子更新与可见性裁剪迁移到 Compute Queue。
3. **M3（指标体系）**
   - CPU/GPU Profiling + UI 面板。
4. **M4（压力场景）**
   - `MT_StressCity` 完整搭建并形成对比报告。

---

## 10. 一句话总结

该项目通过“**CPU 多线程命令录制 + GPU 异步队列并行 + 标准化压力场景测试**”三者协同，验证并量化 C++/Vulkan 在复杂实时渲染中的多线程扩展能力。
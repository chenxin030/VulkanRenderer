# Clustered Forward Shading（`12_clustered`）说明文档

[返回目录](../../README.md)

本文对应代码：
- `src/clustered/ClusteredRenderer.cpp`
- `src/clustered/ClusteredRenderer.h`
- `shaders/12_clustered/cluster_comp.slang`
- `shaders/12_clustered/cluster_lighting.slang`

---

## 1. 渲染出来的画面应当是什么

`12_clustered` 展示的是一个 **Clustered Forward Shading（分簇前向光照）** 的示例场景，画面特征如下：

1. **场景构成**
   - 一块大地面（plane）作为承载。
   - 大量小球（实例化绘制）分布在场景中。这里每个实例本质上对应一个点光源的位置，视觉上会看到很多发光影响叠加的球体区域。

2. **光照表现**
   - 使用了大量动态点光源（默认 2048 个上限，当前场景会生成并动画更新）。
   - 小球和地面会出现明显的多光源漫反射叠加效果，颜色较丰富（随机色调 + 不同强度）。
   - 光源会随时间运动（正弦/余弦轨迹），因此画面中的亮斑和明暗分布会持续变化。

3. **后处理风格**
   - 在片元着色中有简单 tone mapping（`color / (color + 1)`）和 gamma 校正（`pow(color, 0.4545)`），所以最终画面是可视化后的 LDR 效果，不会过曝成纯白。

4. **关闭/开启 Clustered Shading 的差异（UI 开关）**
   - 开启时：每个 cluster 只遍历属于该 cluster 的灯光索引，性能和统计信息更合理。
   - 关闭时：跳过 compute 阶段，画面仍然会按已有 `lightGrid/lightIndex` 数据渲染，但不会更新分簇结果（主要用于对比流程和性能，而不是严格“全光逐像素”路径）。

---

## 2. 渲染流程：经过哪些着色器、各自作用

## 2.1 初始化阶段（一次性）

1. 创建网格：
   - `sphereMesh`（小球）
   - `planeMesh`（地面）

2. 生成灯光数据：
   - `generateSceneLights()` 生成最多 `MAX_LIGHTS`（2048）个点光。

3. 创建两套核心 pipeline：
   - **Compute Pipeline**：`cluster_comp.spv`（入口 `compMain`）
   - **Graphics Pipeline**：`cluster_lighting.spv`（入口 `vertMain` + `fragMain`）

4. 创建 UI Pipeline：
   - `imgui.spv`（用于参数和统计显示）

---

## 2.2 每帧主流程（render）

在 `ClusteredRenderer::render()` 里，关键顺序是：

1. 更新 CPU->GPU 数据（`updateClusterBuffers`）
   - 更新相机矩阵、cluster 参数、灯光动画位置、地面参数。

2. 更新 UI 帧（`updateUIFrame`）

3. 若开启 clustered：
   - 录制并提交 compute 命令（`recordComputeCommandBuffer`）
   - 计算并回读 cluster 统计（`updateClusterStats`）
   - 再录制 graphics 命令并提交（`recordCommandBuffer`）
   - graphics 提交时等待 compute 完成信号量，确保片元阶段读到最新光列表。

4. 若关闭 clustered：
   - 直接走 graphics 提交（不执行 compute 更新）。

5. present 到交换链。

---

## 2.3 Compute 着色器：`cluster_comp.slang`（`compMain`）

职责：**构建每个 cluster 对应的灯光列表**。

- 线程组织：`[numthreads(8,8,8)]`，dispatch 覆盖 
  \($clusterX \times clusterY \times clusterZ$\) 三维簇网格。
- 每个线程处理一个 cluster：
  1. 根据 `dispatchThreadID` 得到 cluster 坐标 `(x,y,z)`。
  2. 通过 `getClusterMin/getClusterMax` 计算该 cluster 在视锥空间近似 AABB。
  3. 遍历点光源，做 `sphereIntersectsAABB`（球-AABB 相交）测试。
  4. 命中的灯索引写入 `lightIndexBuffer`。
  5. 在 `lightGrid[clusterIndex]` 写入：
     - `offset`
     - `count`

输出结果：
- `lightGrid`：每个 cluster 的灯光范围信息。
- `lightIndexBuffer`：压缩后的灯光索引列表（每 cluster 最多 `MAX_LIGHTS_PER_CLUSTER = 64`）。

---

## 2.4 图形着色器：`cluster_lighting.slang`

同一个 slang 文件中包含顶点和片元入口。

### a) 顶点着色器 `vertMain`

职责：
- 使用 `SV_InstanceID` 从 `lightBuffer` 取当前实例对应点光位置。
- 将球体局部顶点偏移到该点光世界坐标附近（形成“每灯一个小球”的实例化渲染）。
- 输出：
  - `WorldPos`
  - `Normal`
  - `ViewPos`
  - `SV_POSITION`

### b) 片元着色器 `fragMain`

职责：
1. 根据当前片元 `ViewPos` 计算所属 cluster（`getClusterIndex`，Z 方向是对数切分）。
2. 从 `lightGrid[clusterIdx]` 取得该 cluster 的灯光索引区间。
3. 只遍历该区间中的灯：
   - 计算 Lambert 漫反射
   - 距离衰减
   - 颜色与强度叠加
4. 加环境光（`ambient=0.02`）
5. tone mapping + gamma 校正

核心价值：
- 不再“每像素遍历全部灯”，而是“每像素只遍历所在 cluster 的灯”。

---

## 2.5 UI 着色器：`imgui.spv`

- 用于叠加 ImGui 面板。
- 不参与场景光照计算，只负责参数调节与统计展示。

---

## 3. UI 中各个参数分别代表什么

UI 窗口标题：`Clustered Forward Shading`

## 3.1 `Enable Clustered Shading`

- 类型：勾选框（bool）
- 含义：是否启用 compute 分簇更新流程。
- 影响：
  - 开启：每帧执行 cluster 构建并用最新 light grid 渲染。
  - 关闭：跳过 compute 更新，主要用于流程/性能对照。

## 3.2 `Cluster X`

- 类型：滑条（4 ~ 32）
- 含义：屏幕 X 方向切分簇数量。
- 影响：
  - 值越大，横向簇更细，每簇候选灯通常更少；
  - 但 compute dispatch 数和管理开销会增加。

## 3.3 `Cluster Y`

- 类型：滑条（4 ~ 18）
- 含义：屏幕 Y 方向切分簇数量。
- 影响与 `Cluster X` 类似，决定纵向分辨率。

## 3.4 `Cluster Z`

- 类型：滑条（4 ~ 64）
- 含义：视深度方向簇数量（按对数深度分层）。
- 影响：
  - 值越大，深度方向划分越细，远近灯光分离更准确；
  - 同时 cluster 总量增加，compute 和内存负担提高。

## 3.5 `Lights`

- 类型：只读统计（uint）
- 含义：场景灯数量（当前代码里通常是 2048）。

## 3.6 `Total Clusters`

- 类型：只读统计（uint）
- 含义：总簇数 = `Cluster X * Cluster Y * Cluster Z`。
- 用途：衡量当前分簇粒度和 compute 工作量规模。

## 3.7 `Lights/Cluster (avg)`

- 类型：只读统计（uint）
- 含义：平均每个非空 cluster 包含的灯数量（CPU 从 `lightGrid` 回读计算）。
- 用途：观察分簇效果是否合理：
  - 太大说明分簇太粗；
  - 太小说明可能分簇过细（管理开销大）。

## 3.8 `Max Lights/Cluster`

- 类型：只读常量（uint）
- 含义：单个 cluster 允许存储的灯索引上限（当前是 64）。
- 作用：限制显存和循环上界，避免极端场景无限增长。

## 3.9 `Frame` / `FPS`

- 类型：只读性能指标
- 含义：
  - `Frame`：单帧耗时（ms）
  - `FPS`：每秒帧数
- 用途：调整 cluster 参数时观察实时性能变化。

---

## 4. 一句话总结

`12_clustered` 的核心是：
**先用 compute 给每个空间簇建立“灯光索引表”，再在前向片元着色阶段按簇取灯，从而在大量动态点光下保持正确光照并降低逐像素遍历成本。**

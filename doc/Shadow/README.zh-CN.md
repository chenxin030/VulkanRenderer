# Shadow（Level 5）

返回目录：[README.zh-CN.md](../../README.zh-CN.md)

运行时通过 UI 切换： Hard / PCF / PCSS

## 阴影原理
把光源当做摄像机渲染一次，得到深度图。

渲染场景时，把顶点位置换到光源视角下的位置，然后把光源视角下的深度和上面得到的深度图做对比，如果深度较大，说明相对位置是光源——遮挡物——顶点。即该顶点位置被遮挡了，位于阴影下。

## 测试场景

场景用于对比“贴地 vs 离地”对阴影质量的影响：

- 高柱：立柱上方有高悬浮方块
- 低柱：立柱高度极低，方块几乎贴地
- 静态方块：直接放置在地面上

## shadow_lit.slang 渲染流程

`shadow_lit.slang` 负责场景主渲染与阴影采样，主要流程如下。

### 顶点着色器

**阴影坐标**：`shadowUbo.pointLightPosIntensity.xyz * shadowUbo.lightViewProj` 得到光源视角下的位置 `shadowPos`，“正交投影”得到 UV 采样阴影贴图。

### 片段着色器

- 计算视线方向 `viewDir` 和基础颜色 `baseColor`。
- 计算三种光源的漫反射 + 高光贡献。
- 只有 **方向光** 参与阴影（`computeShadow`）。
- 最终颜色：`ambient + dirShadow * (dirDiffuse + dirSpecular) + pointDiffuse + pointSpecular + areaLight`。

### 阴影采样（`computeShadow`）

1. **光源 NDC 转 UV**：`shadowPos.xyz / shadowPos.w`，映射到 `uv = proj.xy * 0.5 + 0.5`。
2. **深度合法性检查**：超出 `[0,1]` 直接返回可见度 1（如果是 UV超了，说明不在这盏光的 shadow frustum 覆盖区域内；如果是 `currentDepth` 超了，说明深度超出光源裁剪范围（在 near/far 外）；返回 1 表示不施加阴影（完全可见））
3. **自适应偏移**：`bias = max(0.0018 * (1 - ndotl), shadowBiasMin)`。
4. **滤波模式**：
   - `Hard`：单采样深度比较。 
   - `PCF`：16 点  Poisson 采样，求平均可见度。
   - `PCSS`：Blocker Search → Penumbra → PCF 过滤。

## 光照模型（3 种光源）

场景包含三种光源，可调节强度。

### Directional light（投射阴影）

- 方向与强度：`dirL = normalize(-dirLightDirIntensity.xyz)`，`dirIntensity = dirLightDirIntensity.w`
- 漫反射：`dirDiffuse = baseColor * max(dot(N, L), 0) * dirIntensity * dirColor`
- 高光：`dirSpecular = pow(max(dot(N, H), 0), 64) * 0.25 * dirIntensity * dirColor`
- 阴影： `dirShadow = computeShadow(shadowPos, N, dirL)`，最终贡献乘以 `dirShadow`。

### Point light（不投射阴影，含距离衰减）

- 位置与强度：`pointPos = pointLightPosIntensity.xyz`，`pointIntensity = pointLightPosIntensity.w`
- 距离衰减：`pointAtt = 1 / (1 + dist^2)`
- 漫反射：`pointDiffuse = baseColor * max(dot(N, L), 0) * pointIntensity * pointAtt * pointColor`
- 高光：`pointSpecular = pow(max(dot(N, H), 0), 64) * 0.20 * pointIntensity * pointAtt * pointColor`

### Area light（矩形面积光，4 点采样近似）

- 光源由中心 `areaLightPosIntensity.xyz` 和两条边向量 `areaLightU/V` 定义。
- 以 4 个角点采样（`[-1,-1]`, `[-1,1]`, `[1,-1]`, `[1,1]`）：
  1. 对每个采样点计算 `L`、衰减、漫反射与高光。
  2. 将 4 次采样求和并取平均（乘 `0.25`）。
- 最终：`areaLight = areaSum * (areaIntensity * 0.25) * areaColor`。

## Shadow Map Pass
先 model * position 得到世界空间坐标，再乘以光源的 VP 矩阵，就是光源视角下的场景，着色器输出场景深度到深度附件，转换格式给到场景渲染使用。

## PCF
将单次比较扩展为多点采样求平均：

- 采样多个 offset（例如 Poisson disk）
- 对每个 sample 比较 `currentDepth - bias` 与 `closestDepth`
- 取平均得到可见度

### 公式
- 采样半径（纹理坐标空间）： $radius = pcfRadiusTexels \times invShadowMapSize$ 也就是“多少个 texel 的半径”，换算成 UV 偏移量。
- 每个点的判断： $\text{lit}_i = ((currentDepth - bias) \le closestDepth_i)\ ?\ 1:0$ 被挡住记 0，没挡住记 1。
  - currentDepth - bias 为什么是减: “receiver 侧深度回退”，用来抑制 shadow acne。
  - 比较条件本质是：若 receiverDepth > shadowMapDepth ? 被挡住（在阴影里）, 但数值误差会让本该相等的深度也被判“>”，产生 acne, 所以用 receiverDepth - bias 去比较，相当于“放宽判定”，减少误判阴影。
  - 如果改成 currentDepth + bias，会更容易判成阴影（更黑），acne 会更严重；
  - 而减太多又会导致 peter-panning（阴影与物体脱离）。
  - 所以 bias 是在 acne 和 peter-panning 之间折中调参。
- 平均可见度： $visibility = \frac{1}{16}\sum_i lit_i$
- 最终返回： $shadow = lerp(0.25,1.0,visibility)$


### 参数
- pcfRadiusTexels
  - PCF 的核半径，单位是“阴影贴图 texel”
  - 它影响采样的范围，采样阴影的半径越大，平均范围越大，16 个采样点分布得越“散”，覆盖的阴影图邻域越大，平均后边缘就越软、越模糊。
  - 值越大：阴影边缘越软，但更糊、漏光风险更高
  - 值越小：边缘更硬，容易锯齿
- texelSize = invShadowMapSize（float2）
  - 阴影贴图尺寸倒数 $(1/width,\ 1/height)$
  - 用于把“texel 单位”转换成“UV 单位”
  - 比如 2048 图：invShadowMapSize ≈ (0.000488, 0.000488)
- shadowBiasMin
  - 最小深度偏移（bias）的下限，防止自阴影粉刺（shadow acne）
  - 代码里实际 bias 是： $bias = \max(0.0018 \cdot (1-ndotl),\ shadowBiasMin)$ 即“斜率相关 bias + 最小 bias”组合
- poisson[16]
  - 16 个固定采样方向（分布比较均匀）
  - 比规则 4x4 核更不容易出现条纹感
## PCSS
- PCF 半径固定 `pcfRadiusTexels`，所以软硬基本固定。
- PCSS 会先估计“接收点和遮挡体之间的相对距离”，再动态放大滤波半径 `pcssLightSizeTexels`：
  - 接收点离遮挡体越远 → 半影更宽 → 阴影更软
  - 接收点靠近遮挡体 → 半影更窄 → 阴影更硬
  
两阶段：

1. **pcssLightSizeTexels**
   - 光源尺寸尺度（也用于搜索半径和半影放大）
2. **Blocker Search**
   - 和PCF一样,在范围内多次采样，把周围的深度求和取平均,得到平均遮挡深度 `avgBlockerDepth`
3. **Penumbra Size**
   - “接收点比遮挡体远多少”的归一化度量。越大表示半影应更宽。
   - $penumbra = \frac{currentDepth - avgBlockerDepth}{max(avgBlockerDepth, 1e-4)};$
4. **filterRadiusTexels**
   - 用来做 PCF

### 一句话
做了2次深度采样，第一次的深度用来计算第二次PCF的采样半径要缩放多少

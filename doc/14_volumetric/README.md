# 14_volumetric — Volumetric Lighting（体积光）

## 1. 项目目标

实现一套完整的屏幕空间体积光方案，用于展示现代游戏引擎核心技术能力，核心目标：

### 技术目标
- **方向光源体积光**：实现经典的上帝光线/体积雾效果
- **Ray Marching 算法**：掌握 GPU 渲染中的光线步进技术
- **Temporal Reprojection**：实现时域降噪，接近生产级别质量
- **Clustered Volumetric**（可选）：复用 Cluster Grid 架构优化剔除
- **阴影集成**：体积光与不透明物体遮挡的正确交互

### 求职定位
本项目定位为**高质量渲染技术展示**，完成后可作为简历核心项目：

> "实现了一套完整的屏幕空间体积光方案，包含 Ray Marching 散射计算、Temporal Reprojection 时域降噪、Clustered 架构优化，以及与 Shadow Map 的深度集成。能够在半分辨率下达到 60fps。"

---

## 2. 为什么分 4 个阶段

### 分阶段原因

| 原因 | 说明 |
|------|------|
| **复杂度递增** | 每个阶段在前一阶段基础上增加一层技术，每层都可以独立验证正确性 |
| **调试友好** | 出问题时可以快速定位是基础算法问题还是优化/集成问题 |
| **风险可控** | 如果某阶段遇到困难不会影响前一阶段的成果 |
| **面试友好** | 面试时可以逐步展开："首先是基础版，然后我们加了 Temporal 降噪..." |

### 技术依赖关系

```
Phase 1: 基础 Ray Marching
    ↓ 复用
Phase 2: Temporal Reprojection
    ↓ 可选复用
Phase 3: Clustered Volumetric
    ↓ 复用
Phase 4: Shadow Integration
```

### 时间预算

| 阶段 | 建议时间 | 最低时间 |
|------|----------|----------|
| Phase 1 | 2-3 天 | 1-2 天 |
| Phase 2 | 2-3 天 | 1-2 天 |
| Phase 3 | 2-3 天 | 可跳过 |
| Phase 4 | 1-2 天 | 0.5-1 天 |

---

## 3. 四阶段详细说明

### Phase 1: 最小可行产品 — 基础 Ray Marching

#### 目标
实现一个能跑的基础体积光 Demo，包含 Ray Marching 散射计算。

#### 要做什么
- [ ] 单方向光源（模拟太阳/聚光灯）
- [ ] 固定步长等距采样
- [ ] Beer-Lambert 指数衰减吸收
- [ ] Henyey-Greenstein 各向异性散射（可调 g 参数）
- [ ] 半分辨率渲染（可选，先全分辨率验证正确性）
- [ ] UI 参数控制：散射系数、步长、最大距离

#### 技术要点
```glsl
// Beer-Lambert 衰减
float absorption = exp(-density * marchDistance);

// Henyey-Greenstein 相函数
float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1 - g2) / (4 * PI * pow(1 + g2 - 2*g*cosTheta, 1.5));
}
```

#### 产出
- `volumetric_lighting.spv`：基础体积光 shader
- 可调节的体积光效果（浓淡可调）
- 验证 Ray Marching 算法正确性

#### 验收标准
- [ ] 体积光效果可见，近处浓、远处淡
- [ ] 调整参数能明显改变效果
- [ ] 帧率可接受（优化前 30fps 可接受）

---

### Phase 2: 性能优化 — Temporal Reprojection 降噪

#### 目标
通过时域 Reprojection 大幅降低采样噪声，达到接近生产级别的质量。

#### 要做什么
- [ ] 历史缓冲（2 帧 ping-pong）
- [ ] 速度缓冲（Velocity Buffer）计算
- [ ] History Clamp / Variance Clamp（防止鬼影）
- [ ] Blue Noise 抖动采样偏移
- [ ] 半分辨率渲染 + 上采样到全分辨率
- [ ] 收敛速度调优（tonemporalFactor 参数）

#### 技术要点
```glsl
// Velocity 计算（基于相机矩阵变化）
float4 currentClipPos = mul(projMatrix, currentViewPos);
float4 prevClipPos = mul(prevProjMatrix, prevViewPos);
float2 velocity = (currentClipPos.xy/currentClipPos.w - prevClipPos.xy/prevClipPos.w) * 0.5;

// History Clamp（防止鬼影）
float historyMin = history.r * (1 - variance);
float historyMax = history.r * (1 + variance);
float clampedHistory = clamp(history.r, historyMin, historyMax);

// Temporal 混合
float result = lerp(clampedHistory, currentSample, temporalFactor);
```

#### 产出
- `volumetric_reproject.spv`：Temporal Accumulation shader
- `volumetric_blur.spv`：可选的双边模糊 shader
- 历史缓冲管理代码
- 速度缓冲计算逻辑

#### 验收标准
- [ ] 噪声明显减少或消失
- [ ] 相机移动时无明显鬼影
- [ ] 半分辨率下帧率提升 50%+

---

### Phase 3: 架构增强 — Clustered Volumetric（可选）

#### 目标
复用 12_clustered 的 Cluster Grid 架构，减少无效 Ray Marching 计算。

#### 要做什么
- [ ] 复用 Cluster Grid 的 Tile/Cluster 划分逻辑
- [ ] Compute Shader 预计算每个 Cluster 的体积散射贡献
- [ ] Fragment 只对可见 Cluster 进行积分
- [ ] Early Exit 优化（Cluster 内无遮挡时快速跳过）
- [ ] 与 Phase 1/2 的集成

#### 技术要点
```glsl
// Clustered Volumetric Compute Pass
[numthreads(8, 8, 8)]
void compMain(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint3 clusterId = dispatchThreadId;
    uint clusterIndex = flatIndex(clusterId);

    // 预计算该 Cluster 的体积散射
    float3 scattering = computeClusterScattering(clusterId, clusterParams);

    // 写入 Cluster 散射缓冲
    clusterScatteringBuffer[clusterIndex] = scattering;
}

// Fragment 只读取命中 Cluster
float3 getClusterScattering(uint3 clusterId) {
    return clusterScatteringBuffer[flatIndex(clusterId)];
}
```

#### 产出
- `volumetric_cluster_comp.spv`：Compute 预计算 shader
- 扩展的 Cluster Grid 数据结构
- Early Exit 优化逻辑

#### 验收标准
- [ ] 性能提升 20%+（相比非 Clustered）
- [ ] 与原有 Clustered 代码无冲突
- [ ] Cluster 分辨率可调

---

### Phase 4: 视觉增强 — Shadow Integration

#### 目标
让体积光与场景中的不透明物体正确交互，产生被遮挡的阴影效果。

#### 要做什么
- [ ] 复用 4_shadow 的 Shadow Map 资源
- [ ] Ray Marching 过程中查询 Shadow Map
- [ ] 被遮挡区域体积光衰减
- [ ] 可选：多级 Shadow Map（级联阴影）
- [ ] UI 控制：阴影强度、阴影偏差

#### 技术要点
```glsl
// Ray Marching 中查询阴影
float shadowAttenuation = 1.0;
for (int i = 0; i < STEPS; i++) {
    float3 samplePos = rayOrigin + rayDir * currentDist;

    // 查询阴影贴图
    float shadowDepth = sampleShadowMap(samplePos);
    float currentDepth = getDepthFromLight(samplePos);

    // 遮挡检测
    if (currentDepth < shadowDepth - bias) {
        shadowAttenuation *= shadowFalloff; // 阴影衰减
    }

    // 体积光累积（乘阴影衰减）
    transmittance *= exp(-density * stepSize);
    scatteredLight += inScattering * shadowAttenuation * stepSize * transmittance;
}
```

#### 产出
- Shadow Map 集成逻辑
- 体积光阴影效果
- 参数可调的 UI 控制面板

#### 验收标准
- [ ] 体积光经过物体时有明显阴影衰减
- [ ] 无明显的阴影瑕疵（Peter Panning、Shadow Acne）
- [ ] 与 Phase 2 的 Temporal 无冲突

---

## 4. 完整项目结构

```
src/14_volumetric/
├── VolumetricRenderer.h           # 头文件
├── VolumetricRenderer.cpp         # 主类、init/render/cleanup
├── VolumetricRenderer_resources.cpp  # 资源创建
├── VolumetricRenderer_core.cpp    # 核心初始化
├── VolumetricRenderer_rendering.cpp  # 命令录制
├── VolumetricRenderer_utils.cpp   # 辅助函数
├── StandaloneMain.cpp             # 程序入口
└── CMakeLists.txt

shaders/14_volumetric/
├── volumetric_lighting.slang      # Ray Marching 核心
├── volumetric_reproject.slang    # Temporal Reprojection
├── volumetric_blur.slang         # 可选模糊
└── volumetric_cluster_comp.slang  # Compute 预计算（可选）

doc/14_volumetric/
└── README.md                      # 本文档
```

---

## 5. 技术栈对齐（面试要点）

| 现代引擎技术 | 在项目中体现 |
|--------------|--------------|
| UE Lumen | Temporal Reprojection + Multi-bounce 近似 |
| Naughty Dog TAA | Velocity Buffer + History Clamp |
| GPU Driven Rendering | Clustered Volumetric + Compute Shader |
| Screen-space Effect | 体积光作为屏幕空间后处理 |

---

## 6. 实施建议

### 如果时间充裕（3-4 周）
完成全部 4 个 Phase，代码量和技术深度都足够。

### 如果时间紧张（1-2 周）
**推荐组合**：Phase 1 + 2 + 4，跳过 Phase 3。

- Phase 1：基础 Ray Marching（核心能力）
- Phase 2：Temporal Reprojection（高频面试点）
- Phase 4：Shadow Integration（视觉效果增强）

### 最小可行简历项目
**Phase 1 + 2 即可打动面试官**，核心是展示：
1. 手动实现 Ray Marching 算法
2. 理解 Beer-Lambert + Henyey-Greenstein 散射模型
3. 掌握 Temporal Reprojection 降噪技术

---

## 7. 参考资料

| 类型 | 资源 |
|------|------|
| Siggraph | "Volumetric Fog: Building a Procedural Volumetric Fog Shader in Unity" |
| GDC | "Next Generation Post Processing in Call of Duty: Advanced Warfare" |
| 论文 | "Real-time Volumetric Cloudscapes" - Andrew Schneider |
| 引擎 | Unreal Engine 5 Lumen 文档 |
| 开源 | https://github.com/SaschaWillems/Vulkan （volumetric lights 示例）|

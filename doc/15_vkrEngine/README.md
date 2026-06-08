# vkrEngine — 统一渲染器（Project 15）

> 尽可能把前面实现的小功能塞进去，场景采用 [Sponza](https://github.com/jimmiebergmann/Sponza) ，使用 GPU Timestamp 分析性能。

---

## 总体架构

```
vkrEngine.exe
├── VkrRenderer (继承 VulkanBase)
│   ├── Scene (场景管理)
│   │   ├── Renderable (Model* + Transform)
│   │   └── PointLight (位置 + 颜色 + 强度)
│   ├── VkrModel (glTF 加载，sub-mesh + 材质)
│   ├── TextureCache (贴图去重)
│   └── GpuProfiler (Timestamp Query 自动性能分析)
├── Shaders (Slang → SPIR-V)
│   ├── scene_vert.slang
│   └── scene_frag.slang
└── ImGui (运行时控制面板)
    ├── 渲染特性开关 (SSR/CSM/Clustered)
    ├── 各 Pass 耗时分解
    └── 场景统计 (DrawCall / 三角形数 / 显存)
```

---

## 分阶段开发计划

### Phase 1: 基础设施 + 场景加载 ✅
- [x] 重构 Model 系统（sub-mesh + 材质 + tangent）
- [x] Scene 场景管理器
- [x] GpuProfiler（Timestamp Query）
- [x] 加载 Sponza + 基础渲染
- [x] 12 Bug 修复并记录

![Phase 1](assets/01_phase1_final.png)

**场景数据**:

| 指标               | 数值    |
| ------------------ | ------- |
| 场景三角形数       | 262,267 |
| Sub-meshes         | 103     |
| Draw Calls         | 103     |
| 材质（已加载纹理） | 25 / 25 |
| FPS (Debug)        | ~900    |

详见: [Phase 1 开发日志](dev-log-phase1-baseline.md)

### Phase 2: PBR + IBL ✅
- [x] Cook-Torrance GGX BRDF (微表面模型)
- [x] HDR 环境贴图 IBL（Irradiance + Prefiltered EnvMap + BRDF LUT）
- [x] HDR 环境: newport_loft.hdr
- [x] 法线贴图 + 切线空间 TBN
- [x] 逐材质 PBR 参数 UBO
- [x] UI Exposure / Gamma / LightingMode / DirLight 开关
- [x] 纹理格式: baseColor→sRGB, normal/metallicRoughness→UNORM
- [x] Descriptor: Set0=Scene+IBL, Set1=Material (5 bindings)
- [x] A/B 对比: Phase1 Simple ↔ Phase2 PBR+IBL 实时切换

![Phase 2](assets/02_phase2_pbr_ibl.png)

详见: [Phase 2 开发日志](dev-log-phase2-pbr-ibl.md)

### Phase 3: CSM 级联阴影 ✅
- [x] 4 级 CSM 阴影映射（2048×2048 纹理数组，双缓冲）
- [x] Hard / PCF / PCSS 滤波切换
- [x] Practical Split Scheme（λ=0.45, far=2000m）
- [x] 级联可视化（红/绿/蓝/黄）
- [x] 场景 AABB 自适应光源距离（避免光源在模型内部）
- [x] 7 种调试可视化模式（ShadowMap/ShadowFac/NDC-Z/w 热力图等）
- [x] 6 个 Bug 修复并记录

详见: [Phase 3 开发日志](dev-log-phase3-csm.md)

### Phase 4: 延迟渲染 + SSR ✅
- [x] GBuffer 管线（Albedo + Normal + Material + Depth）
- [x] SSR 屏幕空间反射（视空间 raymarching + binary refinement）
- [x] Forward vs Deferred 管线共存
- [x] Debug Views（Final / Albedo / Normal / PBR / Depth）

详见: [Phase 4 开发日志](dev-log-phase4-deferred-ssao-ssr.md)

### Phase 5: GPU-Driven Culling + Clustered Shading ✅
- [x] Frustum + Hi-Z Culling
- [x] Clustered Shading（2048 光源）
- [x] DrawCall / 帧时间对比

详见: [Phase 5 开发日志](dev-log-phase5-culling-clustered.md)

### Phase 6: 后处理 + TAAU + 性能优化
- [ ] Bloom + ToneMapping
- [ ] TAAU 时序抗锯齿
- [ ] RenderDoc/NSight 全面 Profile
- [ ] 2-3 项深度优化，记录优化报告

---

## 主要因素

- **场景格式**: glTF 2.0（tinygltf 加载）
- **主测试场景**: Sponza（~66K tris）
- **混合场景**: Bistro（FBX→glTF 转换后引入，Phase 4+）
- **不使用完整 ECS**: `vector<Renderable>` 足够
- **RenderGraph 简化**: 手动编排 Pass，参考 FrameGraph 概念但不引入完整依赖图
- **多线程暂不整合**: 先跑通单线程管线

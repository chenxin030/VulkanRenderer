# vkrEngine — 统一渲染器（Project 15）

> 将 14 个独立模块整合为**单一可配置的延迟渲染引擎 Demo**，以 Sponza 大场景为载体，集成 GPU Timestamp 自动性能分析，逐步实现并量化每个阶段的性能提升。

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
    ├── 渲染特性开关 (SSAO/SSR/CSM/Clustered)
    ├── 各 Pass 耗时分解
    └── 场景统计 (DrawCall / 三角形数 / 显存)
```

---

## 分阶段开发计划

### Phase 1: 基础设施 + 场景加载
- [x] 重构 Model 系统（sub-mesh + 材质 + tangent）
- [x] Scene 场景管理器
- [x] GpuProfiler（Timestamp Query）
- [x] 加载 Sponza + 基础渲染
- [ ] 记录基线性能指标

**基线数据**:

| 指标          | 数值 |
| ------------- | ---- |
| 场景三角形数  |      |
| Draw Call 数  |      |
| 帧时间 (ms)   |      |
| 显存占用 (MB) |      |

### Phase 2: PBR + IBL
- [ ] 集成 PBR 管线
- [ ] HDR 环境贴图 IBL（Irradiance + Prefiltered EnvMap + BRDF LUT）
- [ ] UI 切换不同 HDR

### Phase 3: CSM 级联阴影
- [ ] 4 级 CSM 阴影映射
- [ ] Hard / PCF / PCSS 切换
- [ ] 各模式性能对比

### Phase 4: 延迟渲染 + SSAO + SSR
- [ ] GBuffer 管线（Albedo + Normal + Material + Depth）
- [ ] SSAO 屏幕空间环境光遮蔽
- [ ] SSR 屏幕空间反射
- [ ] Forward vs Deferred 性能对比

### Phase 5: GPU-Driven Culling + Clustered Shading
- [ ] Frustum + Hi-Z Culling
- [ ] Clustered Shading（2048 光源）
- [ ] DrawCall / 帧时间对比

### Phase 6: 后处理 + TAAU + 性能优化
- [ ] Bloom + ToneMapping
- [ ] TAAU 时序抗锯齿
- [ ] RenderDoc/NSight 全面 Profile
- [ ] 2-3 项深度优化，记录优化报告

---

## 关键决策

- **场景格式**: glTF 2.0（tinygltf 加载）
- **主测试场景**: Sponza（~66K tris）
- **混合场景**: Bistro（FBX→glTF 转换后引入，Phase 4+）
- **不使用完整 ECS**: `vector<Renderable>` 足够
- **RenderGraph 简化**: 手动编排 Pass，参考 FrameGraph 概念但不引入完整依赖图
- **多线程暂不整合**: 先跑通单线程管线

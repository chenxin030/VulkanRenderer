# CSM (Cascaded Shadow Maps) 实现文档 — vkrEngine

## 1. 算法概述

CSM 将摄像机视锥体沿深度方向分割为多个级联（cascade），每个级联渲染一张独立的阴影贴图。近处级联分辨率高、远处级联分辨率低，从而在全场景范围内获得均匀的阴影质量。

```
摄像机视锥体 (0.1 ~ 2000):
├── Cascade 0: near~split1  → ShadowMap Layer 0 (高分辨率, 近距离)
├── Cascade 1: split1~split2 → ShadowMap Layer 1
├── Cascade 2: split2~split3 → ShadowMap Layer 2
└── Cascade 3: split3~far    → ShadowMap Layer 3 (低分辨率, 远距离)
```

## 2. 数据结构

### 2.1 CPU 端

```
VkrRenderer.h:

// 级联分割深度 (view-space)
float cascadeSplitDepths[CASCADE_COUNT + 1];  // [near, split1, split2, split3, far]

// 每个级联的 lightView * lightProj 矩阵
glm::mat4 cascadeViewProj[CASCADE_COUNT];
```

### 2.2 GPU 端 (CsmUBO, Set 0 Binding 5)

```hlsl
struct CsmUBO {
    float4x4 cascadeViewProj[4];   // 每个级联的 VP 矩阵
    float4 cascadeSplitDepths;     // x=split1, y=split2, z=split3, w=far
};
```

### 2.3 阴影贴图

- 格式: `VK_FORMAT_D32_SFLOAT`, 2D Array
- 尺寸: `2048 × 2048 × 4 layers`
- 双缓冲: 2 份 (per frame-in-flight)，避免读写竞争
- 采样器: Linear + ClampToBorder (白色边框 = 无阴影)

## 3. 渲染流程

```
每帧:
  1. updateSceneUBO()        → 写入 lightDir 到 sceneUbo
  2. updateCsmBuffers()      → 从 sceneUbo 读取 lightDir
     ├─ calculateCascadeSplits()   → 计算级联分割深度
     └─ computeCascadeViewProj(x4) → 计算每个级联的 lightView*lightProj
  3. recordCommandBuffer()
     ├─ Pass 0: CSM Depth Pass
     │   ├─ Transition shadow map → DEPTH_ATTACHMENT
     │   ├─ for each cascade (0..3):
     │   │   ├─ beginRendering (depth-only, clear=1.0)
     │   │   ├─ bind csmDepthPipeline + descriptor set (CsmUBO)
     │   │   ├─ pushConstants: modelMatrix + cascadeIndex
     │   │   └─ draw Sponza geometry
     │   └─ Transition shadow map → SHADER_READ_ONLY
     └─ Pass 1: Main Color Pass (采样 shadow map)
```

### 3.1 CSM Depth Pipeline

独立的管线，仅输出深度：
- Vertex Shader: `csm_depth.slang` — 用 CsmUBO.cascadeViewProj[cascadeIndex] 变换顶点
- Fragment Shader: 空（深度自动写入）
- Rasterizer: Back-face culling (CCW), depth bias enabled
- Dynamic Rendering: depth-only attachment

### 3.2 级联分割 (calculateCascadeSplits)

使用实用分割方案 (Practical Split Scheme)：

```
split_i = λ × (near × (far/near)^(i/N)) + (1-λ) × (near + (far-near) × i/N)
```

- `λ = 0.45` (UI 可调)
- `near = CAMERA_NEAR = 0.1f`
- `far = CAMERA_FAR = 2000.0f`
- `N = CASCADE_COUNT = 4`

### 3.3 灯光 View-Proj 矩阵 (computeCascadeViewProj)

**当前实现 (相机无关，使用模型实际 AABB)**：

1. **场景包围盒** (从模型顶点自动计算):
   ```
   sceneMin = sponzaModel.aabbMin - padding
   sceneMax = sponzaModel.aabbMax + padding
   ```

2. **灯光 View 矩阵** (固定):
   ```
   lightPos = sceneCenter - lightDir × 50
   lightView = lookAt(lightPos, sceneCenter, up)
   ```
   灯光看向场景中心；view space 中 Z 轴沿视线方向为负。

3. **包围盒变换到灯光空间 → AABB**:
   ```
   将 8 个 world-space 角点通过 lightView 变换，取 min/max。
   view-space Z 为负值。
   ```

4. **正交投影**:
   ```
   // glm::ortho 的 zNear/zFar 是正距离值，实际裁剪面在 Z = -zNear / -zFar
   glm::ortho(snappedMinX, snappedMaxX, snappedMinY, snappedMaxY, -maxAABB.z, -minAABB.z)
   ```
   - `zNear = -maxAABB.z` (最近物体到灯光的距离，正值；maxAABB.z 通过 clamp 保持为负)
   - `zFar = -minAABB.z` (最远物体到灯光的距离，正值)
   - Texel snapping: 将 AABB 量化到阴影贴图像素网格
   - `lightProj[1][1] *= -1` 修正 Vulkan Y 轴

5. **所有级联共享同一投影** — 级联选择由 shader 根据 viewDepth 处理。

## 4. Shader 端 (csm_depth.slang)

```hlsl
// Vertex Shader
VSOutput vertMain(VSInput input) {
    float4 worldPos = mul(push.modelMatrix, float4(input.inPosition, 1.0));
    output.pos = mul(csmUbo.cascadeViewProj[push.cascadeIndex], worldPos);
    return output;
}

// Fragment Shader — 无需输出颜色
void fragMain() {}
```

## 5. 关键参数

| 参数                      | 默认值  | 说明                  |
| ------------------------- | ------- | --------------------- |
| `CASCADE_COUNT`           | 4       | 级联数量              |
| `SHADOW_MAP_SIZE`         | 2048    | 阴影贴图分辨率        |
| `CAMERA_NEAR`             | 0.1f    | 摄像机近平面          |
| `CAMERA_FAR`              | 2000.0f | 摄像机远平面          |
| `uiSplitLambda`           | 0.45    | 级联分割 λ            |
| `uiShadowFilterMode`      | 2       | 0=Hard, 1=PCF, 2=PCSS |
| `depthBiasConstantFactor` | 1.25    | 深度偏移常量          |
| `depthBiasSlopeFactor`    | 1.75    | 深度偏移斜率          |
| `zPadding`                | 50.0f   | 正交投影 Z 轴扩展     |

## 6. 调试

### 6.1 RenderDoc 检查清单

1. **CSM Depth Pass**: 检查 4 个 cascade layer 是否包含场景几何体
   - 若全白 (clear=1.0) → 几何体被裁剪或未覆盖
   - 若全黑 → 几何体太近，被 near plane 裁剪

2. **CsmUBO 内容**: 检查 cascadeViewProj 矩阵是否合理（非零/非单位阵）

3. **顶点变换**: 验证 `worldPos → cascadeViewProj → clipPos` 中顶点是否在 NDC [-1,1] 范围内

### 6.2 常见问题

| 现象                    | 可能原因                                           |
| ----------------------- | -------------------------------------------------- |
| 阴影贴图全白 (无几何体) | zPadding 过大导致 zNear 变负；场景包围盒不覆盖模型 |
| 阴影随摄像机移动        | 投影矩阵依赖摄像机视锥体                           |
| 阴影闪烁/锯齿           | 投影未做 texel snapping                            |
| 级联间阴影不连续        | split depths 与 shader 中 cascade selection 不一致 |

## 7. 已知问题 & 待修复

- [ ] zPadding 可能导致靠近灯光的级联 zNear 变负 (zPadding > 最近物体距离)
- [ ] 所有级联共享同一正交投影，未按深度范围独立裁剪
- [ ] 灯光方向接近垂直时，lookAt 需要 fallback up 向量 (已处理)

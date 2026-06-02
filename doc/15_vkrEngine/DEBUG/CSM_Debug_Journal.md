# CSM 阴影调试日志 — vkrEngine

> 记录了从"阴影随摄像机移动"到"阴影正确静止"的完整调试过程。

---

## Bug #1: 阴影随摄像机移动

**报告日期**: 2026-05-30

**现象**: 移动摄像机位置或视角时，阴影也会跟着变化。场景中只有固定方向的方向光，阴影应该是静止不动的。

**我的初步判断**: `computeCascadeViewProj()` 使用 `camera.GetViewMatrix()` 和 `camera.Zoom` 来计算级联投影矩阵。摄像机移动 → 视锥体角点世界坐标改变 → 灯光空间 AABB 改变 → 正交投影改变 → 阴影跟随移动。这是标准 CSM 的行为，但用户期望的是相机无关的静态阴影。

**修复 v1**: 将投影改为相机无关，使用固定世界空间包围盒 `sceneMin(-40,-5,-35)` ~ `sceneMax(40,40,35)`（硬编码），并加入 texel snapping 防抖动。

**结果**: 编译通过，但引入新 Bug。

---

## Bug #2: 阴影贴图为空（只有黑→白渐变，无几何体）

**报告日期**: 2026-05-31

**现象**: RenderDoc 中 4 个 shadow pass 深度贴图都是从下往上的由黑渐变成白色，没有看到任何场景物体的轮廓。

**我的假设 #1 — zPadding 导致 zNear 变负**: 
- `glm::ortho(left, right, bottom, top, zNear, zFar)` 的 `zNear`/`zFar` 参数是**正距离值**，实际裁剪面在 view space Z = -zNear / -zFar。
- 场景 AABB 在灯光 view space 中 Z 为负值（约 -87 ~ -3）。
- `maxAABB.z += zPadding = -3.3 + 50 = +46.7` → `zNear = -maxAABB.z = -46.7`（负数！）
- 负 zNear → 近裁剪面在摄像机后方 → 所有几何体被裁剪 → 阴影贴图全白。

**调试手段**: 在 `computeCascadeViewProj` 中添加诊断输出，打印 lightDir、lightViewSpace AABB、zNear/zFar。

**修复 v2**: `maxAABB.z = std::min(maxAABB.z + zPadding, -0.01f)` — clamp 保持负值。

**结果**: zNear/zFar 变为正数，但阴影贴图仍然为空。

---

## Bug #3: 阴影贴图仍然为空（修复 v2 后）

**报告日期**: 2026-05-31

**我的假设 #2 — 硬编码包围盒不覆盖 Sponza 模型**:
硬编码 `(-40,-5,-35)` ~ `(40,40,35)` 可能和实际 Sponza 位置/尺寸不匹配。

**修复 v3**: 
1. 在 `VkrModel` 中添加 `aabbMin`/`aabbMax`，加载时从顶点自动计算
2. `computeCascadeViewProj` 使用 `sponzaModel.aabbMin/Max` 替代硬编码
3. 灯光距离也根据 AABB 大小动态计算

**结果**: 阴影 pass 开始显示场景几何体！但包围盒异常大:
```
[VkrModel] Loaded 'Sponza': AABB:(-1920.95,-126.442,-1182.81)-(1799.91,1429.43,1105.43)
[CSM diag] lightViewSpace AABB: zNear=2619.67 zFar=6719.56
```

Sponza 包围盒约 3720×1556×2288 单位（可能是厘米单位，约 37m 宽），导致阴影贴图每个像素覆盖约 1.8 世界单位——分辨率很差，大物体阴影应该仍可见但细节丢失。

---

## Bug #4: 阴影贴图有内容但主渲染看不到阴影

**报告日期**: 2026-05-31

**现象**: RenderDoc 中 shadow pass 有完整场景几何体，但主渲染画面中找不到阴影。

**我的假设 #3 — 全场景 AABB 太大导致分辨率过低**:
每像素覆盖约 1.8 世界单位，阴影细节完全丢失。用户反馈说"阴影应该覆盖视野范围即可"。

**用户建议**: 阴影贴图大小应该根据摄像机位置和方向进行调整（即标准 CSM 方案）。

**修复 v4**: 恢复相机视锥体 CSM + texel snapping，替代全场景方案。

**结果**: 
```
[CSM diag] zNear=0.01 zFar=278.796 texelSize=(0.16, 0.21)
```
分辨率极好（每像素 0.16 单位），阴影可见！但用户反馈阴影又随视角移动了——回归了 Bug #1。

---

## 根本矛盾

| 方案                    | 阴影静止     | 分辨率             |
| ----------------------- | ------------ | ------------------ |
| 相机无关（全场景 AABB） | ✅ 静止       | ❌ 低（1.8/pixel）  |
| 相机视锥体 CSM          | ❌ 随视角移动 | ✅ 高（0.16/pixel） |

用户核心需求：**阴影必须静止（方向光固定）**。

---

## 最终方案: 相机无关投影 + 模型实际 AABB

**最终修复 (v5)**:

回到相机无关方案，但使用模型实际 AABB（自动计算，不再硬编码）：

```cpp
// 使用加载时计算的模型 AABB + padding
const glm::vec3 sceneMin = sponzaModel.aabbMin - kAabbPad;
const glm::vec3 sceneMax = sponzaModel.aabbMax + kAabbPad;

// 灯光看向场景中心（固定）
const glm::vec3 sceneCenter = (sceneMin + sceneMax) / 2.0f;
lightView = lookAt(lightPos, sceneCenter, lightUp);

// 所有级联共享同一投影（级联选择由 shader viewDepth 处理）
lightProj = glm::ortho(snappedMinX, snappedMaxX, snappedMinY, snappedMaxY,
                        -maxAABB.z, -minAABB.z);
```

**关键组件**:
- `VkrModel::aabbMin/Max`: 加载时从顶点数据自动计算
- `zPadding + clamp`: 防止 zNear 变负
- `Texel snapping`: 防止灯光方向微小变化引起亚像素抖动
- `lightUp fallback`: 处理灯光方向接近垂直时的 `lookAt` 退化

**验证结果**:
- 不同视角下 shadow pass 的 4 个深度图完全一样 ✅
- 阴影不随摄像机移动 ✅
- 分辨率可接受（约 1.8 世界单位/像素，对于 2048 阴影贴图覆盖 ~3700 单位场景）

---

## 附录: 诊断输出格式

```
[CSM diag] frame N lightDir=(x,y,z)
[CSM diag] modelAABB:(minX,minY,minZ)-(maxX,maxY,maxZ)
[CSM diag] lightView AABB:(minX,minY,minZ)-(maxX,maxY,maxZ) zNear=N zFar=F texelSize=(tx,ty)
```

## 附录: zNear 变负问题详解

`glm::ortho(left, right, bottom, top, zNear, zFar)`:
- zNear/zFar 是**正距离值**（从摄像机到裁剪面的距离）
- 实际裁剪面: near = Z = **-zNear**, far = Z = **-zFar**
- 灯光 view space 中，摄像机看向 -Z，场景物体 Z 为负值

当 `maxAABB.z += zPadding` 中 maxAABB.z 原本是负值（如 -3.3），加上 50 的 padding 变成 +46.7（正值），则 `zNear = -46.7`（负数），裁剪面跑到摄像机后方。

**修复**: `maxAABB.z = std::min(maxAABB.z + zPadding, -0.01f)` — 确保始终为负，`zNear` 始终为正。

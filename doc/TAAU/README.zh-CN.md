# TAAU（Level 6）

返回文档目录：[README.zh-CN.md](../../README.zh-CN.md)

该 Level 提供 **TAAU 测试路径**，复用现有的 shadow 渲染管线，用于暴露时域累积常见弱点，便于在不改动整体架构的情况下迭代 TAAU 策略。

## 目标（弱点覆盖）

- **拖影**：历史混合导致运动物体残影。
- **细线闪烁**：亚像素几何重投影时易抖动。
- **快速运动**：大位移造成历史拒绝与稳定性问题。
- **屏幕边缘 / 高频纹理**：边缘反遮挡与密集细节带来不稳定。

## 入口位置

- `src/Core/Renderer_TAAU.cpp`（TAAU 逻辑）
- `src/Core/Renderer.h`（`RENDERING_LEVEL == 6` 切换）
- `src/Core/ResourceManager.h`（测试场景 transform）

## 运行流程

1. **资源初始化**
   - 复用 Level 5 的 shadow descriptor set 与 pipeline。
   - `createShadowMapResources()` 先执行，再调用 `createShadowDescriptorSets()`，确保 shadow map image view 有效。

2. **逐帧更新**
   - Level 6 的 `updateShadowBuffers()` 更新：
     - 相机 UBO
     - 光源 UBO
     - shadow 参数 UBO
     - 实例变换与颜色
   - `updateTAAUScene()` 负责动态 stress-test 物体。

3. **UI**
   - `updateUIFrame()` 中先 `ImGui::NewFrame()`，再调用 `updateTAAUUI()`。
   - `updateTAAUUI()` 独立窗口展示 TAAU 调试参数。

## 测试场景布局

在 `ResourceManager.h` 的 `RENDERING_LEVEL == 6` 中定义：

- 地面（边缘闪烁）
- 细柱阵列（细线抖动）
- 快速移动 probe
- 屏幕边缘 probe
- 高频条纹阵列（高频压力）

## UI 控件

`TAAU Debug` 窗口提供：

- **BlendFactor**：历史混合权重（拖影敏感度）
- **ReactiveClamp**：高频与反遮挡的夹制
- **AntiFlicker**：细节稳定因子
- **VelocityScale**：快速运动压力
- **FreezeHistory**：冻结历史以观察残影

## 说明

当前 Level 仍 **复用 shadow 管线**，尚未加入真正的 TAAU resolve shader。该路径用于搭建骨架，后续可逐步加入 velocity buffer、jitter、history texture 以及 resolve pass。

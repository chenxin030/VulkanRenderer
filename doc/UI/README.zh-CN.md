# UI 渲染流程说明（以 `RENDERING_LEVEL == 5` 为例）

UI 渲染采用的是：

- 使用 ImGui 生成绘制数据（`ImDrawData`）
- 在 Vulkan 中使用独立 UI Pipeline 绘制
- 将 UI 叠加到主场景颜色附件（swapchain image）上

本质上是“接管 ImGui DrawData，然后在当前帧的命令缓冲中手动绘制”。

---

## 1. 初始化阶段：`initUI()`

在 `src/Core/Renderer_UI.cpp` 中，`initUI()` 主要完成以下工作：

1. 创建 ImGui Context，并设置样式（Dark）
2. 获取字体图集（`GetTexDataAsRGBA32`）
3. 使用 staging buffer 将字体纹理上传到 GPU（`uiFontTexture`）
4. 创建 UI Descriptor：
   - `binding 0` 为 `CombinedImageSampler`（字体纹理）
5. 创建 UI Pipeline Layout：
   - DescriptorSetLayout（字体采样）
   - Push Constant（`float2 scale + float2 translate`，共 16 字节）
6. 创建 UI Graphics Pipeline（基于 Dynamic Rendering）：
   - 顶点输入格式对应 `ImDrawVert`（`pos/uv/col`）
   - 关闭深度测试与深度写入
   - 开启 Alpha Blending
   - 使用动态状态 `viewport + scissor`

因此，UI 是一套完全独立于主场景材质/管线的渲染路径。

---

## 2. 每帧 UI 构建：`updateUIFrame()`

在 `render()`（`src/Core/Renderer.h`）中，当 `RENDERING_LEVEL == 5` 时，每帧会先调用：

- `updateUIFrame()`
- `updateShadowBuffers(currentFrame)`

`updateUIFrame()` 的职责：

1. 更新 ImGui IO：
   - 显示尺寸
   - `deltaTime`
   - 鼠标输入状态
2. 调用 `ImGui::NewFrame()` 开始新帧
3. 构建 level 5 对应 UI（例如 `Shadows & Lights` 面板）
4. 调用 `ImGui::Render()` 生成 `ImDrawData`

---

## 3. 命令录制位置：`recordCommandBuffer()` → `recordUI()`

在 `src/Core/Renderer_rendering.cpp` 的 `RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6` 分支中，顺序是：

1. Shadow depth pass（渲染到 shadow map）
2. 主场景光照 pass（`shadowLitPipeline`）
3. 调用 `recordUI(commandBuffer)` 绘制 UI
4. `endRendering()` 结束本次 dynamic rendering

也就是说，UI 是叠加在主场景之后的 overlay。

---

## 4. `recordUI()` 的绘制细节

`recordUI(vk::raii::CommandBuffer& commandBuffer)` 的核心流程：

1. 获取 `ImGui::GetDrawData()`
2. 使用按帧管理的 `uiFrameBuffers[currentFrame]`：
   - 若顶点/索引缓冲区容量不足，则重建
   - 使用 `mapMemory` 后将 ImGui 顶点与索引数据 `memcpy` 到 GPU 可见内存
3. 绑定绘制状态：
   - `uiPipeline`
   - `uiDescriptorSets[currentFrame]`（字体纹理）
   - 顶点/索引缓冲
4. 设置 Push Constant：
   - `scale = (2 / DisplaySize.x, 2 / DisplaySize.y)`
   - `translate = (-1, -1)`
5. 遍历每个 `ImDrawList/ImDrawCmd`：
   - 将 `ClipRect` 转换为 Vulkan `scissor`
   - 调用 `drawIndexed()` 绘制

---

## 5. `shaders/imgui.slang` 的职责

### 顶点着色器 `vertMain`

- 输入：`Pos / UV / Color`
- 通过 Push Constant 将屏幕坐标映射到裁剪空间：

`output.Pos = float4(input.Pos * pc.scale + pc.translate, 0.0, 1.0)`

### 片段着色器 `fragMain`

- 采样字体纹理：`fontTexture.Sample(input.UV)`
- 最终输出：

`finalColor = input.Color * tex`

即：顶点色与字体纹理颜色（通常主要是 alpha）相乘。

---

## 6. 结论（Level 5）

- UI 与主场景渲染解耦，使用独立 Pipeline。
- UI 绘制依赖 ImGui 生成的 `ImDrawData`。
- UI 在主场景之后绘制，作为最终覆盖层。
- 通过每条 `ImDrawCmd` 的 `scissor` 实现精确裁剪。
- UI 关闭深度测试，无需测试，直接覆盖在上面。

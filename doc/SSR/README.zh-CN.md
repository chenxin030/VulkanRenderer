# SSR（屏幕空间反射）

返回目录：[README.zh-CN.md](../../README.zh-CN.md)

英文版本：[README.md](README.md)

本渲染器在 `RENDERING_LEVEL == 7` 下实现了轻量级 SSR。目标是用当前帧的深度与颜色实现视角相关反射，不依赖反射探针或光追。

## 渲染流程

1. **光照/不透明阶段** 输出到 swapchain 颜色。
2. **SSR 输入捕获** 将 swapchain 颜色复制到 `ssrColor` 纹理。
3. **SSR 合成** 使用深度进行视空间光线步进，并将反射结果混合回 swapchain。
4. **UI 渲染** 在 SSR 之后绘制，避免被反射。

## 输入数据

- **深度**：当前帧深度缓冲（`DEPTH_READ_ONLY_OPTIMAL` 采样）。
- **颜色**：swapchain 颜色捕获（`SHADER_READ_ONLY_OPTIMAL` 采样）。
- **Scene UBO**：投影、视图、逆投影、相机位置、近远平面。
- **SSR 参数**：最大距离、厚度、步长、最大步数、强度。

## 算法要点

- 使用逆投影矩阵进行 **视空间重建**。
- **视空间光线步进**，并用短二分进行命中细化。
- **屏幕边缘衰减**，减轻边缘伪影。

## 相关文件

- `src/Core/Renderer_SSR.cpp`
- `shaders/ssr.slang`

## 说明

SSR 受屏幕可见性限制，离屏或被遮挡的物体不会出现在反射中。

# Instanced（Level 2）

English version: [README.md](README.md)

本文介绍 Level 2 的实例化渲染路径：用一次 draw call 渲染多个实例，并通过 SSBO/UBO 提供每实例的变换数据。

## 目标

- 使用 `vkCmdDrawIndexed(..., instanceCount=MAX_OBJECTS, ...)` 一次性绘制 N 个实例
- 相机矩阵放在全局 UBO；每实例模型矩阵放在 SSBO
- 使用 Dynamic Rendering（不依赖传统 RenderPass）

## 运行流程

1. 创建缓冲
   - `GlobalUBO`：view/proj
   - `InstanceData[]`：model 矩阵数组
2. 创建描述符
   - binding 0：全局 UBO
   - binding 1：实例 SSBO
   - binding 2：纹理采样器（Level 2 demo 使用一张纹理）
3. 每帧更新
   - 更新相机 UBO
   - 更新实例 SSBO（每个实例的 model）
4. 录制命令
   - 绑定 pipeline / descriptor set
   - `drawIndexed(indexCount, instanceCount, ...)`

## 数据与绑定

### C++ 侧

相关实现集中在：

- `src/Core/Renderer_instanced.cpp`
- `src/Core/Renderer_rendering.cpp`
- `src/Core/Renderer.h`

核心数据结构：

- `GlobalUBO`：view/proj
- `InstanceData`：model

### Shader 侧（Slang）

- `shaders/instanced.slang`

约定：

- 顶点着色器使用 `SV_InstanceID` 索引 SSBO
- `globalUbo` 提供 view/proj
- `instanceBuffer[instanceID].model` 提供 model

## 常见问题

### Validation：Vertex attribute not consumed

当 pipeline 的 vertex attribute 声明（location 0/1/2）与 shader 输入不匹配时，验证层会给出 “not consumed” 警告。

建议：

- 为每类 shader 输入提供对应的 `Vertex::getXXXAttributeDescriptions()`，避免在某条 pipeline 中声明未使用的 vertex attributes。

实现位于 `src/Core/Mesh.h` 的 `Vertex`。

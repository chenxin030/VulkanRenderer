# 实例化渲染（Level 2）

返回目录：[README.zh-CN.md](../../README.zh-CN.md)

英文版本：[README.md](README.md)

本章描述 Level 2 的实例化渲染路径：使用一次 draw 调用渲染多个实例，并通过 UBO/SSBO 提供每个实例的变换矩阵。

## 目标

- 使用 `vkCmdDrawIndexed(..., instanceCount=MAX_OBJECTS, ...)` 一次绘制 N 个实例
- 相机矩阵放在全局 UBO；每实例的 model 矩阵放在 SSBO
- 使用 Dynamic Rendering（不依赖传统 RenderPass）

## 运行流程

1. 创建缓冲
   - `GlobalUBO`：view / proj
   - `InstanceData[]`：每个实例的 model 矩阵
2. 创建描述符
   - binding 0：全局 UBO
   - binding 1：实例 SSBO
   - binding 2：纹理采样器（Level 2 演示使用纹理）
3. 每帧更新
   - 更新相机 UBO
   - 更新实例 SSBO（每个实例的 model）
4. 记录命令
   - 绑定 pipeline / descriptor set
   - `drawIndexed(indexCount, instanceCount, ...)`

## 数据与绑定

### C++

主要实现文件：

- `src/Core/Renderer_instanced.cpp`
- `src/Core/Renderer_rendering.cpp`
- `src/Core/Renderer.h`

核心数据：

- `GlobalUBO`：view / proj
- `InstanceData`：model

### 着色器（Slang）

- `shaders/instanced.slang`

约定：

- 顶点着色器使用 `SV_InstanceID` 索引 SSBO
- `globalUbo` 提供 view / proj
- `instanceBuffer[instanceID].model` 提供 model

## 常见问题

### 验证层提示：Vertex attribute not consumed

如果 pipeline 顶点属性声明（location 0/1/2）与 shader 输入不匹配，会出现 “not consumed” 警告。

建议：

- 使用 `Vertex::getXXXAttributeDescriptions()` 提供专用布局
- 避免为某个 pipeline 声明未使用的 attribute

参考：`src/Core/Mesh.h` 中的 `Vertex`。

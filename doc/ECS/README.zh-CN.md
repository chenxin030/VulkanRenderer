# ECS（实体组件系统）

返回目录：[README.zh-CN.md](../../README.zh-CN.md)

英文版本：[README.md](README.md)

本渲染器采用轻量级 ECS 风格的场景组织方式，用于管理变换、渲染实例与场景查询。它提供了结构化的组件管理方式，同时保持渲染数据连续、易于迭代。

## 目标

- **数据导向布局**，提高缓存友好性。
- **简单 API**，便于创建变换与渲染实例。
- **场景查询**，按 MeshTag 收集渲染实例。

## 典型流程

1. 创建实体并附加组件（如 Transform、MeshTag）。
2. 每帧更新 Transform。
3. 收集渲染实例，写入 GPU 缓冲用于绘制。

## 相关代码

- `src/Core/Scene.h`
- `src/Core/Scene.cpp`

## 说明

该 ECS 层定位为渲染服务的轻量实现，不是完整的游戏逻辑框架。

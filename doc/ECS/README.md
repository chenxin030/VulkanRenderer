# ECS (Entity Component System)

Back to root: [README.md](../../README.md)

Chinese version: [README.zh-CN.md](README.zh-CN.md)

This renderer uses a lightweight ECS-style scene representation to organize transforms, render instances, and scene queries. It provides a structured way to manage objects and their components while keeping rendering data contiguous and efficient.

## Goals

- **Data-oriented layout** for cache-friendly iteration.
- **Simple APIs** for creating transforms and render instances.
- **Scene queries** to collect instances for rendering passes.

## Typical Flow

1. Create entities and attach components (e.g., transform, mesh tag).
2. Update transforms each frame.
3. Collect render instances for a specific mesh tag and feed them into GPU buffers.

## Related Code

- `src/Core/Scene.h`
- `src/Core/Scene.cpp`

## Notes

The ECS layer is intentionally minimal—focused on rendering needs and iteration efficiency rather than a full gameplay framework.

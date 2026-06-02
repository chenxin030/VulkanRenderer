# createBuffer 函数参考手册

> 记录了 vkrEngine 及 Base 库中所有 `createBuffer` 调用，按 usage+properties 组合分类，说明各自用途。

---

## 0. 底层函数签名

```cpp
// VulkanBase::createBuffer — VulkanBase_Resource.cpp:133
void createBuffer(
    vk::DeviceSize          size,       // 缓冲区字节大小
    vk::BufferUsageFlags    usage,      // 用途标志（决定 GPU 如何使用该 buffer）
    vk::MemoryPropertyFlags properties, // 内存属性（决定 buffer 存在哪、谁能访问）
    vk::raii::Buffer&       buffer,     // [out] 缓冲区句柄
    vk::raii::DeviceMemory& bufferMemory // [out] 显存分配句柄
);
```

**内部实现**:
1. `vk::BufferCreateInfo{ .size, .usage, .sharingMode=eExclusive }` 创建 buffer
2. `getMemoryRequirements()` 查询所需内存大小和类型约束
3. `findMemoryType(typeFilter, properties)` 找到满足属性要求的 GPU 内存堆索引
4. `vk::MemoryAllocateInfo{ .allocationSize, .memoryTypeIndex }` 分配内存
5. `bindMemory(buffer, memory, 0)` 绑定

---

## 1. Usage + Properties 组合速查

### 组合 A: Staging Buffer（中转缓冲）

```
usage:      eTransferSrc
properties: eHostVisible | eHostCoherent
用途:       CPU 写入数据 → GPU 侧通过 copyBuffer/copyBufferToImage 读取
特点:       CPU 可 map/memcpy/unmap，GPU 只读
生命周期:   用完即弃（一次性 submit + waitIdle 后销毁）
```

**出现位置**:

| 文件                      | 行      | 用途                                       | 数据大小               |
| ------------------------- | ------- | ------------------------------------------ | ---------------------- |
| `VulkanBase_Resource.cpp` | 13      | `createVertexBuffer`: 顶点数据中转         | `sizeof(vertex)*count` |
| `VulkanBase_Resource.cpp` | 33      | `createIndexBuffer`: 索引数据中转          | `sizeof(index)*count`  |
| `VulkanBase_Resource.cpp` | 192     | `LoadHDRTextureFromFile`: HDR 纹理数据中转 | `W*H*4*sizeof(float)`  |
| `VulkanBase_Resource.cpp` | 222     | `LoadTextureFromFile`: LDR 纹理数据中转    | `W*H*4`                |
| `VulkanBase_UI.cpp`       | 71      | ImGui 字体纹理上传                         | `fontW*fontH*4`        |
| `VkrRenderer.cpp`         | 198-201 | Sponza 顶点数据中转                        | `sizeof(VkrVertex)*N`  |
| `VkrRenderer.cpp`         | 221-224 | Sponza 索引数据中转                        | `sizeof(uint32_t)*N`   |
| `VkrRenderer.cpp`         | 249     | `createDummyWhiteTexture`: 1×1 白色纹理    | `4 bytes`              |
| `VkrRenderer.cpp`         | 285     | `createDummyNormalTexture`: 1×1 法线纹理   | `4 bytes`              |
| `VkrRenderer.cpp`         | 336     | `loadMaterialTexture`: 材质纹理上传        | `W*H*4`                |

**典型用法**:
```cpp
createBuffer(size,
    vk::BufferUsageFlagBits::eTransferSrc,                          // GPU 可从中拷贝
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, // CPU 可写
    stagingBuf, stagingMem);
void* m = stagingMem.mapMemory(0, size);
memcpy(m, cpuData, size);
stagingMem.unmapMemory();
copyBuffer(stagingBuf, deviceLocalBuf, size);  // GPU 拷贝到设备本地内存
```

---

### 组合 B: Device Vertex Buffer（设备端顶点缓冲）

```
usage:      eTransferDst | eVertexBuffer
properties: eDeviceLocal
用途:       GPU 端顶点数据只读，渲染时绑定为 VK_VERTEX_INPUT_RATE_VERTEX
特点:       CPU 不可直接访问，需通过 staging buffer + copyBuffer 上传
```

**出现位置**:

| 文件                      | 行      | 用途                                    |
| ------------------------- | ------- | --------------------------------------- |
| `VulkanBase_Resource.cpp` | 19      | `createVertexBuffer`: Mesh 顶点缓冲     |
| `VkrRenderer.cpp`         | 207-210 | Sponza 顶点缓冲（VkrVertex，stride=48） |

**典型用法**:
```cpp
// 1. 创建 staging buffer（组合 A）
// 2. 创建 device local buffer
createBuffer(size,
    vk::BufferUsageFlagBits::eTransferDst |                           // GPU 拷贝目标
    vk::BufferUsageFlagBits::eVertexBuffer,                           // 可绑定为顶点缓冲
    vk::MemoryPropertyFlagBits::eDeviceLocal,                         // 仅 GPU 可访问
    model.vertexBuffer, model.vertexBufferMemory);
// 3. GPU 端拷贝
copyBuffer(stagingBuf, model.vertexBuffer, size);
```

---

### 组合 C: Device Index Buffer（设备端索引缓冲）

```
usage:      eTransferDst | eIndexBuffer
properties: eDeviceLocal
用途:       GPU 端索引数据只读，渲染时绑定为 VK_INDEX_TYPE_UINT32
```

**出现位置**:

| 文件                      | 行      | 用途                               |
| ------------------------- | ------- | ---------------------------------- |
| `VulkanBase_Resource.cpp` | 39      | `createIndexBuffer`: Mesh 索引缓冲 |
| `VkrRenderer.cpp`         | 230-233 | Sponza 索引缓冲（uint32_t）        |

---

### 组合 D: Uniform Buffer（统一变量缓冲/UBO）

```
usage:      eUniformBuffer
properties: eHostVisible | eHostCoherent
用途:       每帧 CPU 写入变换矩阵、材质参数等 → GPU shader 中读取
特点:       CPU 持久映射 (persistent map)，无需 staging buffer
数量:       按 MAX_FRAMES_IN_FLIGHT 批量创建（每帧独立，避免竞争）
```

**出现位置**:

| 文件                      | 行         | 用途                                        | 每帧大小           |
| ------------------------- | ---------- | ------------------------------------------- | ------------------ |
| `VulkanBase_Resource.cpp` | 55         | `createUniformBuffers`: 通用 UBO            | 可变               |
| `VulkanBase_Resource.cpp` | 73         | `createUniformBuffers(count)`: 指定数量     | 可变               |
| `VkrRenderer.cpp`         | 69         | `sceneUboResources`: SceneUBO               | 176 bytes          |
| `VkrRenderer.cpp`         | 70         | `paramsUboResources`: ParamsUBO             | 16 bytes           |
| `VkrRenderer.cpp`         | 71         | `csmUboResources`: CsmUBO                   | 272 bytes          |
| `VkrRenderer.cpp`         | 72         | `shadowParamsUboResources`: ShadowParamsUBO | 32 bytes           |
| `VkrRenderer.cpp`         | 按材质数量 | `materialUboResources[i]`: MaterialUBO      | 80 bytes × 25 材质 |

**典型用法**:
```cpp
createBuffer(size,
    vk::BufferUsageFlagBits::eUniformBuffer,                          // UBO
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
    uboBuffer, uboMemory);
void* mapped = uboMemory.mapMemory(0, size);  // 持久映射
// 每帧更新:
memcpy(mapped, &uboData, size);
```

---

### 组合 E: Storage Buffer（存储缓冲/SSBO）

```
usage:      eStorageBuffer
properties: eHostVisible | eHostCoherent
用途:       CPU 写入可变大小的实例数据 → GPU shader 中以 StructuredBuffer 读取
区别 UBO:   UBO 有 64KB 上限且只读，SSBO 可更大且可读写
```

**出现位置**:

| 文件                      | 行  | 用途                                               |
| ------------------------- | --- | -------------------------------------------------- |
| `VulkanBase_Resource.cpp` | 91  | `createStorageBuffers`: 通用 SSBO                  |
| `VulkanBase_Resource.cpp` | 109 | `createStorageBuffers(usage)`: 带自定义 usage 标志 |
| `VulkanBase_Resource.cpp` | 127 | `createStorageBuffers(usage, count)`: 指定数量     |

> 注意: vkrEngine 当前未使用 SSBO（Sponza 模型为单一实例，用 push constant 传 model matrix），但在 Multithreaded/Particles/Clustered 等模块中使用。

---

### 组合 F: Host-Visible Vertex Buffer（CPU 可写顶点缓冲）

```
usage:      eVertexBuffer
properties: eHostVisible | eHostCoherent
用途:       ImGui 动态顶点数据（每帧变化，CPU 直接写入，无需 staging + copy）
特点:       相比 DeviceLocal，性能略低但支持 CPU 直接更新
```

**出现位置**:

| 文件                | 行    | 用途                       |
| ------------------- | ----- | -------------------------- |
| `VulkanBase_UI.cpp` | 19-22 | ImGui 顶点缓冲（动态大小） |

---

### 组合 G: Host-Visible Index Buffer（CPU 可写索引缓冲）

```
usage:      eIndexBuffer
properties: eHostVisible | eHostCoherent
用途:       ImGui 动态索引数据
```

**出现位置**:

| 文件                | 行    | 用途                       |
| ------------------- | ----- | -------------------------- |
| `VulkanBase_UI.cpp` | 35-38 | ImGui 索引缓冲（动态大小） |

---

## 2. 关键参数详解

### 2.1 `vk::BufferUsageFlags usage` — 用途标志

| 标志             | 值     | 含义                                         |
| ---------------- | ------ | -------------------------------------------- |
| `eTransferSrc`   | 0x0001 | 可作为 `vkCmdCopyBuffer` 的源                |
| `eTransferDst`   | 0x0002 | 可作为 `vkCmdCopyBuffer` 的目标              |
| `eVertexBuffer`  | 0x0004 | 可绑定为 `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` |
| `eIndexBuffer`   | 0x0008 | 可绑定为 `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`  |
| `eUniformBuffer` | 0x0010 | 可作为 descriptor type = `eUniformBuffer`    |
| `eStorageBuffer` | 0x0020 | 可作为 descriptor type = `eStorageBuffer`    |

> 标志可通过 `|` 组合，如 `eTransferDst | eVertexBuffer`。

### 2.2 `vk::MemoryPropertyFlags properties` — 内存属性

| 标志            | 值     | 含义                                 |
| --------------- | ------ | ------------------------------------ |
| `eDeviceLocal`  | 0x0001 | GPU 本地显存，带宽最高，CPU 不可访问 |
| `eHostVisible`  | 0x0002 | CPU 可通过 map/unmap 访问            |
| `eHostCoherent` | 0x0004 | CPU 写入自动对 GPU 可见，无需 flush  |

**常见组合**:
- `eDeviceLocal` — 纯 GPU 数据（顶点/索引/纹理），性能最高
- `eHostVisible | eHostCoherent` — CPU 可写数据（UBO/SSBO/ImGui/Staging），免 flush
- `eHostVisible`（单独） — CPU 可写但需手动 `vkFlushMappedMemoryRanges`

### 2.3 `vk::SharingMode::eExclusive` — 独占模式

所有 buffer 均使用 `eExclusive`（单队列族独占），因为 vkrEngine 只使用一个 graphics queue。

---

## 3. 数据流向图

```
┌─────────────────────────────────────────────────────────────────┐
│ CPU                                                             │
│  ┌──────────┐  memcpy  ┌──────────────┐                         │
│  │ 原始数据  │ ──────→ │ Staging 内存  │ (eHostVisible|Coherent) │
│  │ (RAM)    │         │ (CPU 写入)    │                         │
│  └──────────┘         └──────┬───────┘                         │
│                              │ copyBuffer (GPU 操作)            │
│                              ▼                                  │
│  ┌──────────┐  memcpy  ┌──────────────┐                         │
│  │ 每帧数据  │ ──────→ │ UBO 持久映射  │ (eHostVisible|Coherent) │
│  │ (CPU计算) │         │               │                         │
│  └──────────┘         └───────────────┘                        │
├─────────────────────────────────────────────────────────────────┤
│ GPU                                                             │
│  ┌──────────────────┐   ┌─────────────────┐                    │
│  │ Device Vertex Buf │   │ Device Index Buf │ (eDeviceLocal)   │
│  │ (vkbinding 0)    │   │ (drawIndexed)    │                   │
│  └──────────────────┘   └─────────────────┘                    │
│                                                                 │
│  ┌──────────────────┐   ┌─────────────────┐                    │
│  │ UBO (set=0,b=0)  │   │ SSBO (set=0,b=1) │ (eHostVisible)   │
│  └──────────────────┘   └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. `findMemoryType` — 内存类型查找

```cpp
// VulkanBase_utils.cpp:6
uint32_t VulkanBase::findMemoryType(
    uint32_t typeFilter,              // getMemoryRequirements().memoryTypeBits
    vk::MemoryPropertyFlags properties // 需要的属性（如 eDeviceLocal）
) const {
    auto memProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        // 内存类型必须在 typeFilter 中允许，且包含所有要求的属性
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}
```

GPU 通常有这些内存堆:
- `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` — VRAM，快但 CPU 不可访问
- `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT` — CPU 可访问的系统 RAM 映射区
- `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | HOST_VISIBLE_BIT` — 部分 GPU 有（如集成显卡），又快又 CPU 可见

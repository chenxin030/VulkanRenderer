# createImage / transitionImageLayout 函数参考手册

> 记录了 vkrEngine 及 Base 库中所有 `createImage` 和 `transitionImageLayout`（含变体）调用，按用途分类说明。

---

# Part A: createImage

## 0. 底层函数签名

```cpp
// VulkanBase::createImage (简化版) — VulkanBase_Resource.cpp:239
void createImage(
    uint32_t width, uint32_t height, uint32_t mipLevels,
    vk::Format format, vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    TextureData& texData);

// VulkanBase::createImage (完整版) — VulkanBase_Resource.cpp:242
void createImage(
    uint32_t width, uint32_t height, uint32_t mipLevels,
    uint32_t arrayLayers,           // ★ >1 表示纹理数组或 Cubemap
    vk::ImageCreateFlags flags,      // ★ eCubeCompatible 等标志
    vk::Format format, vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    TextureData& texData);
```

**内部实现（完整版）**:
1. `vk::ImageCreateInfo{ .flags, .imageType=e2D, .format, .extent={w,h,1}, .mipLevels, .arrayLayers, .samples=e1, .tiling, .usage }`
2. `device.createImage(info)` → 创建 image 句柄
3. `getMemoryRequirements()` + `findMemoryType()` + `vk::MemoryAllocateInfo` → 分配显存
4. `image.bindMemory(memory, 0)` → 绑定

简化版是完整版的 wrapper，传 `arrayLayers=1, flags={}`。

---

## 1. 参数详解

### 1.1 `vk::Format format` — 像素格式

| 格式                  | 字节/像素 | 用途                                 |
| --------------------- | --------- | ------------------------------------ |
| `eR8G8B8A8Unorm`      | 4 bytes   | 普通 RGBA 纹理（dummy white/normal） |
| `eR8G8B8A8Srgb`       | 4 bytes   | sRGB 色彩空间纹理（baseColor）       |
| `eR32G32B32A32Sfloat` | 16 bytes  | HDR 浮点纹理（环境贴图 BRDF LUT）    |
| `eR16G16B16A16Sfloat` | 8 bytes   | 半精度 HDR（环境 Cubemap）           |
| `eD32Sfloat`          | 4 bytes   | 32-bit 深度（CSM 阴影贴图）          |
| `eD16Unorm`           | 2 bytes   | 16-bit 深度（fallback）              |

### 1.2 `vk::ImageUsageFlags usage` — 用途标志

| 标志                      | 含义                                         |
| ------------------------- | -------------------------------------------- |
| `eTransferSrc`            | 可作为拷贝源（如 generateMipmaps）           |
| `eTransferDst`            | 可作为拷贝目标（上传纹理数据）               |
| `eSampled`                | 可在 shader 中采样（combined image sampler） |
| `eColorAttachment`        | 可作为颜色附件（渲染目标）                   |
| `eDepthStencilAttachment` | 可作为深度附件                               |
| `eStorage`                | 可作为 storage image（compute shader 读写）  |

### 1.3 `vk::MemoryPropertyFlags properties`

| 标志           | 含义                           |
| -------------- | ------------------------------ |
| `eDeviceLocal` | GPU 本地显存，CPU 不可直接访问 |

> 所有 createImage 调用均使用 `eDeviceLocal`，数据通过 staging buffer + copy 上传。

### 1.4 `vk::ImageCreateFlags flags`

| 标志                 | 含义                                          |
| -------------------- | --------------------------------------------- |
| `eCubeCompatible`    | 6 层数组 + 立方体贴图视图兼容（用于环境贴图） |
| `e2DArrayCompatible` | 3D 图像可按 2D 数组切片访问（项目中未使用）   |

---

## 2. 所有 createImage 调用分类

### 类别 1: 普通 2D 纹理（1 layer, 1 mip）

```
简化版签名: createImage(w, h, mips, format, tiling, usage, properties, texData)
实质调用: createImage(w, h, mips, 1, {}, format, tiling, usage, properties, texData)
```

| 位置                          | 尺寸 | 格式               | usage                               | 用途                                   |
| ----------------------------- | ---- | ------------------ | ----------------------------------- | -------------------------------------- |
| `VkrRenderer.cpp:258`         | 1×1  | R8G8B8A8Unorm      | `TransferDst\|Sampled`              | Dummy white 纹理（fallback）           |
| `VkrRenderer.cpp:294`         | 1×1  | R8G8B8A8Unorm      | `TransferDst\|Sampled`              | Dummy normal 纹理（flat normal）       |
| `VkrRenderer.cpp:348`         | W×H  | format（参数）     | `TransferSrc\|TransferDst\|Sampled` | Sponza 材质纹理（baseColor/normal/MR） |
| `VulkanBase_Resource.cpp:200` | W×H  | R32G32B32A32Sfloat | `TransferDst\|Sampled`              | HDR 环境贴图（equirect）               |
| `VulkanBase_Resource.cpp:230` | W×H  | R8G8B8A8Srgb       | `TransferSrc\|TransferDst\|Sampled` | LDR 纹理加载                           |
| `VulkanBase_UI.cpp:80`        | W×H  | R8G8B8A8Unorm      | `TransferDst\|Sampled`              | ImGui 字体纹理                         |
| `VulkanBase_Resource.cpp:359` | W×H  | depthFormat        | `DepthStencilAttachment`            | 场景深度缓冲                           |

### 类别 2: 纹理数组（>1 layer）

```
完整版签名: createImage(w, h, mips, layers, flags, format, tiling, usage, properties, texData)
```

| 位置                   | 尺寸      | 层数 | flags             | 格式               | usage                             | 用途                               |
| ---------------------- | --------- | ---- | ----------------- | ------------------ | --------------------------------- | ---------------------------------- |
| `VkrRenderer.cpp:1031` | 512×512   | 6    | `eCubeCompatible` | R16G16B16A16Sfloat | `ColorAttachment\|Sampled`        | 环境 Cubemap（equirect→cube）      |
| `VkrRenderer.cpp:1034` | 64×64     | 6    | `eCubeCompatible` | R16G16B16A16Sfloat | `ColorAttachment\|Sampled`        | Irradiance Cubemap                 |
| `VkrRenderer.cpp:1037` | 512×512   | 6    | `eCubeCompatible` | R16G16B16A16Sfloat | `ColorAttachment\|Sampled`        | Prefiltered EnvMap（mipmap chain） |
| `VkrRenderer.cpp:1040` | 512×512   | 1    | `{}`              | R16G16B16A16Sfloat | `ColorAttachment\|Sampled`        | BRDF LUT（2D）                     |
| `VkrRenderer.cpp:1461` | 4096×4096 | 4    | `{}`              | D32Sfloat          | `DepthStencilAttachment\|Sampled` | **CSM 阴影贴图数组**（×2 双缓冲）  |

### 类别 3: 纹理数组的 Image View 创建

> 注意：Image View 不属于 `createImage`，但与之紧密相关。此处列出关键的 view 类型：

| 图像                  | ViewType   | subresourceRange             | 用途                            |
| --------------------- | ---------- | ---------------------------- | ------------------------------- |
| Cubemap（6 layers）   | `eCube`    | `{Color, 0, mips, 0, 6}`     | shader 中 `SamplerCube` 采样    |
| Cubemap single face   | `e2D`      | `{Color, 0, 1, faceIdx, 1}`  | `beginRendering` 渲染到单面     |
| CSM array（4 layers） | `e2DArray` | `{Depth, 0, 1, 0, 4}`        | shader 中 `Sampler2DArray` 采样 |
| CSM single layer      | `e2DArray` | `{Depth, 0, 1, layerIdx, 1}` | CSM depth pass 渲染到单层       |

---

# Part B: transitionImageLayout（含变体）

## 0. 三种函数变体对比

| 函数                                                                                                          | 定义位置                      | 参数    | 同步模型                           | 使用场景                     |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------- | ---------------------------------- | ---------------------------- |
| `VulkanBase::transitionImageLayout(image, old, new, mips)`                                                    | `VulkanBase_Resource.cpp:297` | 4 参数  | Vulkan 1.0 `vkCmdPipelineBarrier`  | 初始化/上传阶段（单次提交）  |
| `VkrRenderer::transitionImageLayoutCmd(cmd, image, aspect, old, new, ...)`                                    | `VkrRenderer.cpp:987`         | 13 参数 | Vulkan 1.0 `vkCmdPipelineBarrier`  | IBL 生成（多 mip/layer）     |
| `VulkanBase::transition_image_layout(cmd, image, old, new, srcAccess, dstAccess, srcStage, dstStage, aspect)` | `VulkanBase_commands.cpp:110` | 9 参数  | Vulkan 1.3 `vkCmdPipelineBarrier2` | 渲染循环（synchronization2） |

---

## 变体 1: VulkanBase::transitionImageLayout（简化版）

```cpp
void transitionImageLayout(
    const vk::raii::Image& image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    uint32_t mipLevels);
```

**特点**: 仅支持两种 layout 转换，使用 single-time command buffer 立即执行，`waitIdle` 同步。

**支持的两条路径**:

| 路径     | oldLayout             | newLayout                | srcAccess        | dstAccess        | srcStage    | dstStage         |
| -------- | --------------------- | ------------------------ | ---------------- | ---------------- | ----------- | ---------------- |
| 上传纹理 | `eUndefined`          | `eTransferDstOptimal`    | `{}`             | `eTransferWrite` | `TopOfPipe` | `Transfer`       |
| 完成上传 | `eTransferDstOptimal` | `eShaderReadOnlyOptimal` | `eTransferWrite` | `eShaderRead`    | `Transfer`  | `FragmentShader` |

> 其他组合会 `throw invalid_argument`。

**调用位置**（全部在初始化阶段，single-time submit）:

| 文件 + 行                     | 图像        | 路径                       |
| ----------------------------- | ----------- | -------------------------- |
| `VkrRenderer.cpp:262`         | dummyWhite  | Undefined→TransferDst      |
| `VkrRenderer.cpp:265`         | dummyWhite  | TransferDst→ShaderReadOnly |
| `VkrRenderer.cpp:298`         | dummyNormal | Undefined→TransferDst      |
| `VkrRenderer.cpp:301`         | dummyNormal | TransferDst→ShaderReadOnly |
| `VkrRenderer.cpp:354`         | materialTex | Undefined→TransferDst      |
| `VulkanBase_Resource.cpp:203` | HDR tex     | Undefined→TransferDst      |
| `VulkanBase_Resource.cpp:205` | HDR tex     | TransferDst→ShaderReadOnly |
| `VulkanBase_Resource.cpp:232` | LDR tex     | Undefined→TransferDst      |
| `VulkanBase_UI.cpp:85`        | fontTex     | Undefined→TransferDst      |
| `VulkanBase_UI.cpp:89`        | fontTex     | TransferDst→ShaderReadOnly |

---

## 变体 2: VkrRenderer::transitionImageLayoutCmd（自定义版）

```cpp
void transitionImageLayoutCmd(
    vk::raii::CommandBuffer& cmd,  // 外部传入的命令缓冲
    vk::Image image,
    vk::ImageAspectFlags aspectMask,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
    uint32_t baseMipLevel, uint32_t levelCount,   // ★ mip 范围
    uint32_t baseArrayLayer, uint32_t layerCount,   // ★ layer 范围
    vk::PipelineStageFlags srcStage, dstStage,
    vk::AccessFlags srcAccessMask, dstAccessMask);
```

**特点**: 
- 支持任意 layout 转换（无硬编码路径限制）
- 支持指定 mip 层范围和 array layer 范围
- 调用者自行传入 pipeline stage 和 access mask
- 用在 IBL 生成中（需要 transition 多个 mip/layer）

**调用位置** (全部在 `generateIBLResources` 中):

| 行   | 图像              | 方向                    | layers | mip range | 用途                   |
| ---- | ----------------- | ----------------------- | ------ | --------- | ---------------------- |
| 1102 | envCubemap        | Undef→ColorAttachment   | 0..6   | mip 0     | 准备渲染 equirect→cube |
| 1107 | irradianceCubemap | Undef→ColorAttachment   | 0..6   | mip 0     | 准备渲染 irradiance    |
| 1112 | prefilteredEnvMap | Undef→ColorAttachment   | 0..6   | 全部 mips | 准备渲染 prefiltered   |
| 1117 | brdfLut           | Undef→ColorAttachment   | 0..1   | mip 0     | 准备渲染 BRDF LUT      |
| 1317 | envCubemap        | ColorAtt→ShaderReadOnly | 0..6   | mip 0     | 完成 equirect→cube     |
| 1350 | irradianceCubemap | ColorAtt→ShaderReadOnly | 0..6   | mip 0     | 完成 irradiance        |
| 1390 | prefilteredEnvMap | ColorAtt→ShaderReadOnly | 0..6   | 全部 mips | 完成 prefiltered       |
| 1412 | brdfLut           | ColorAtt→ShaderReadOnly | 0..1   | mip 0     | 完成 BRDF LUT          |

---

## 变体 3: VulkanBase::transition_image_layout（渲染循环版）

```cpp
// 使用当前帧命令缓冲的版本
void transition_image_layout(
    vk::Image image,
    vk::ImageLayout old_layout, vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask, dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask, dst_stage_mask,
    vk::ImageAspectFlags image_aspect_flags);

// 使用外部命令缓冲的版本
void transition_image_layout(
    vk::raii::CommandBuffer& cmd,
    vk::Image image, ...);
```

**特点**:
- 使用 `vkCmdPipelineBarrier2`（Vulkan 1.3 synchronization2）
- 外部传入 stage/access masks（调用者完全掌控同步）
- 自动处理 `eUndefined` 源布局（srcAccess→None, srcStage→TopOfPipe）
- 使用当前帧命令缓冲（第一个版本）或外部命令缓冲（第二个版本）

### 调用位置

**在 recordCommandBuffer 中（Pass 1 主渲染）**:

| 行   | 图像               | old→new                 | srcAccess     | dstAccess     | srcStage       | dstStage       | aspect |
| ---- | ------------------ | ----------------------- | ------------- | ------------- | -------------- | -------------- | ------ |
| 1935 | swapChainImages[i] | Undef→ColorAttachment   | None          | ColorAttWrite | TopOfPipe      | ColorAttOutput | Color  |
| 1946 | depthData          | Undef→DepthAttachment   | None          | DepthAttWrite | TopOfPipe      | EarlyFragment  | Depth  |
| 2030 | swapChainImages[i] | ColorAttachment→Present | ColorAttWrite | None          | ColorAttOutput | BottomOfPipe   | Color  |

### 在 CSM 深度通道中（Pass 0）

CSM 深度通道**不使用** `transition_image_layout`，而是直接用 `vk::ImageMemoryBarrier2` 写入 `recordCommandBuffer`：

```
阴影贴图首帧:
  Undefined → DepthAttachmentOptimal  (src=TopOfPipe/None)
阴影贴图后续帧:
  ShaderReadOnly → DepthAttachmentOptimal  (src=FragmentShader/ShaderRead)
CSM pass 完成后:
  DepthAttachmentOptimal → ShaderReadOnly  (src=LateFragment/DepthWrite)
```

---

## 3. 关键 layout 状态机

```
所有纹理遵循的标准生命周期:

  Undefined
    │
    ├─[transitionImageLayout]──→ TransferDstOptimal
    │                             │ copyBufferToImage
    │                             │
    │                             ├─[generateMipmaps]
    │                             │  (内部自行 transition)
    │                             │
    │                             └─[transitionImageLayout]──→ ShaderReadOnlyOptimal
    │                                                           │
    │                                                           └─ shader 采样可用
    │
    ├─[transitionImageLayoutCmd (IBL)]──→ ColorAttachmentOptimal
    │                                       │ beginRendering 渲染
    │                                       │
    │                                       └─[transitionImageLayoutCmd]──→ ShaderReadOnlyOptimal
    │
    └─[CSM ImageMemoryBarrier2]──→ DepthAttachmentOptimal
                                    │ beginRendering 深度通道
                                    │
                                    └─[ImageMemoryBarrier2]──→ ShaderReadOnlyOptimal
```

---

## 4. 关键参数详解

### 4.1 `vk::PipelineStageFlags` — 管线阶段（Vulkan 1.0 变体）

| 阶段                     | 含义                   |
| ------------------------ | ---------------------- |
| `eTopOfPipe`             | 最早的阶段，无前置依赖 |
| `eTransfer`              | 拷贝/清除操作          |
| `eFragmentShader`        | 片段着色器执行         |
| `eColorAttachmentOutput` | 颜色附件写入           |
| `eEarlyFragmentTests`    | 早期深度/模板测试      |
| `eLateFragmentTests`     | 晚期深度/模板测试      |
| `eBottomOfPipe`          | 最晚的阶段，后续无依赖 |

### 4.2 `vk::AccessFlags` — 内存访问类型（Vulkan 1.0 变体）

| 访问类型                                                       | 含义                              |
| -------------------------------------------------------------- | --------------------------------- |
| `eTransferWrite` / `eTransferRead`                             | 拷贝操作读写                      |
| `eShaderRead`                                                  | 着色器采样读取                    |
| `eColorAttachmentWrite` / `eColorAttachmentRead`               | 颜色附件读写                      |
| `eDepthStencilAttachmentWrite` / `eDepthStencilAttachmentRead` | 深度附件读写                      |
| `eNone`                                                        | 无访问（仅用于 Undefined→X 转换） |

### 4.3 `vk::ImageLayout` — 图像布局

| 布局                                                         | 用途                          |
| ------------------------------------------------------------ | ----------------------------- |
| `eUndefined`                                                 | 初始/丢弃内容，GPU 可重新解释 |
| `eTransferDstOptimal`                                        | 作为拷贝/清除目标             |
| `eShaderReadOnlyOptimal`                                     | 着色器只读采样                |
| `eColorAttachmentOptimal`                                    | 颜色渲染目标                  |
| `eDepthAttachmentOptimal` / `eDepthStencilAttachmentOptimal` | 深度渲染目标                  |
| `ePresentSrcKHR`                                             | 交换链呈现                    |

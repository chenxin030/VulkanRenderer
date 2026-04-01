# 10_atmospheric 渲染流程与数据流

## 1) 写作目标

严格依据 `src/atmospheric/` + `shaders/10_atmospheric/` 的真实代码路径，以执行顺序为主线，说明谁写、谁读、何时同步。

---

## 2) 文件清单

| 文件 | 角色 |
|---|---|
| `src/atmospheric/StandaloneMain.cpp` | 程序入口：Platform + Renderer 初始化、主循环 |
| `src/atmospheric/AtmosphericRenderer.h` | 类定义：PushConstants、DescriptorSet 布局、Pipeline 对象 |
| `src/atmospheric/AtmosphericRenderer.cpp` | 实现：资源创建、UI、命令录制 |
| `shaders/10_atmospheric/atmospheric.slang` | 天空穹顶着色器，Rayleigh + Mie 大气散射 |
| `shaders/10_atmospheric/imgui.slang` | ImGui 覆盖层着色器 |

---

## 3) 初始化阶段（一次性资源）

### 3.1 调用链

```
main()
 └─ AtmosphericRenderer::initialize(platform)
    └─ VulkanBase::initVulkan("VulkanRenderer - 10_atmospheric")
       ├─ createInstance / debugMessenger
       ├─ createSurface / device / queues
       ├─ createSwapChain / commandPool
       ├─ createDepthResources
       └─ createCommandBuffers
    └─ prepareResource()
```

### 3.2 `prepareResource()` 详细步骤

```
createSkyDescriptorSetLayout()    // 空布局，0 个 binding
createSkyDescriptorPool()         // MAX_FRAMES_IN_FLIGHT=2 个 descriptor sets
createSkyPipeline()                // Graphics pipeline: atmospheric.spv
createSkyDescriptorSets()          // 分配 2 个 descriptor sets（无绑定资源）
initUI()                           // ImGui context + 字体纹理 + UI pipeline
```

### 3.3 Sky Pipeline 创建参数

| 参数 | 值 |
|---|---|
| Shader | `atmospheric.spv`（编译自 `atmospheric.slang`） |
| 顶点入口 | `vertMain` |
| 片段入口 | `fragMain` |
| 图元拓扑 | `eTriangleList` |
| 深度测试 | **开启**，`eLessOrEqual`（仅在深度=1.0处绘制天空） |
| 深度写入 | **开启** |
| 背面剔除 | `eBack` |
| 混合 | 关闭（不透明） |
| Push Constant | 184 bytes，`VK_SHADER_STAGE_VERTEX_BIT \| VK_SHADER_STAGE_FRAGMENT_BIT` |

Pipeline Layout：1 个空 descriptor set + 1 个 push constant range。

### 3.4 UI Pipeline 创建参数

| 参数 | 值 |
|---|---|
| Shader | `imgui.spv` |
| 顶点入口 | `vertMain` |
| 片段入口 | `fragMain` |
| 深度测试 | **关闭** |
| 深度写入 | **关闭** |
| 背面剔除 | `eNone` |
| 混合 | **开启**：`srcAlpha / OneMinusSrcAlpha` |

**ImGui 字体纹理创建流程**：

```
1. ImGui::GetIO().Fonts->GetTexDataAsRGBA32() → 提取 RGBA 像素
2. 创建 staging buffer，memcpy 像素数据
3. 创建 device local image（eTransferDstOptimal）
4. copyBufferToImage(staging → uiFontTexture)
5. transition → eShaderReadOnlyOptimal
6. 创建 ImageView + Sampler
7. DescriptorSet: binding=0, CombinedImageSampler(fontView, linearSampler)
```

UI Pipeline Layout：1 个 descriptor set（font sampler）+ 16-byte push constant（scale + translate）。

---

## 4) 每帧 CPU 数据准备

**无 UBO/SSBO 上传**。本 Renderer 全部依赖 Push Constants 传递数据。

### 4.1 太阳动画（可选）

```cpp
if (animateSun) {
    sunElevation += platform->frameTimer * 0.05f;  // 太阳仰角
    sunAzimuth   += platform->frameTimer * 0.02f;  // 太阳方位角
    timeOfDay     = sunElevation * 6.0f + 12.0f;  // 映射到 0-24h
}
```

### 4.2 UI 参数准备

`updateUIFrame()` — 更新 ImGui 上下文，响应鼠标/键盘输入。

### 4.3 PushConstants 填充

| 字段 | 值 |
|---|---|
| `invView` | `inverse(camera.GetViewMatrix())` |
| `invProjection` | `inverse(projection)`，`[1][1] *= -1`（Vulkan NDC 校正） |
| `cameraPos` | `camera.Position`（世界坐标） |
| `viewportSize` | `swapChainExtent.width/height` |
| `sunElevation` | 运行时调整（Slider，范围 0~π/2） |
| `sunAzimuth` | 运行时调整（Slider，范围 0~2π） |
| `timeOfDay` | 运行时调整（Slider，范围 0~24） |
| `turbidity` | 运行时调整（Slider，范围 1~10） |
| `mieCoefficient` | 固定 `0.000021f` |
| `rayleighCoefficient` | 固定 `0.0000058f` |

---

## 5) 每帧命令录制主流程

### 5.1 `render()` 主循环

```
CPU: device.waitForFences(inFlightFences[currentFrame])
CPU: swapChain.acquireNextImage(presentCompleteSemaphore)
CPU: device.resetFences() + commandBuffer.reset()
CPU: updateUIFrame()
CPU: recordCommandBuffer(imageIndex)
CPU: graphicsQueue.submit(signal=renderFinishedSemaphore)
CPU: presentQueue.presentKHR(wait=renderFinishedSemaphore)
CPU: currentFrame = (currentFrame + 1) % 2
```

### 5.2 `recordCommandBuffer()` 详细流程

#### Pass 1：天空渲染

```
cmdBuffer.begin()

// Image layout 转换
cmdBuffer.pipelineBarrier(
  srcStageMask = TOP_OF_PIPE,
  dstStageMask = EARLY_FRAGMENT_TESTS,
  imageMemoryBarriers = [
    depthImage     → eDepthAttachmentOptimal,
    swapChainImage → eColorAttachmentOptimal
  ]
)

// Begin sky rendering
cmdBuffer.beginRendering(
  colorAttachments = [{
    imageView = swapChainImages[imageIndex],
    loadOp = CLEAR,  clearValue = (0.53, 0.81, 0.92, 1.0),  // 天蓝色
    storeOp = STORE
  }],
  depthAttachment = [{
    imageView = depthImage,
    loadOp = CLEAR, clearValue = (1.0, 0),  // 深度=1.0
    storeOp = STORE
  }]
)

cmdBuffer.setViewport(0, {0,0, W,H, 0,1})
cmdBuffer.setScissor(0, {{0,0}, {W,H}})

// Bind sky pipeline + descriptor set
cmdBuffer.bindPipeline(skyPipeline)
cmdBuffer.bindDescriptorSets(
  pipelineBindPoint = GRAPHICS,
  layout = skyPipelineLayout,
  firstSet = 0,
  descriptorSets = [skyDescriptorSets[currentFrame]]  // 空 set
)

// Push all atmospheric parameters
cmdBuffer.pushConstants(
  layout = skyPipelineLayout,
  stageFlags = VERTEX | FRAGMENT,
  offset = 0,
  size = sizeof(PushConstants),
  pValues = &pushConstants
)

// Draw full-screen triangle (3 vertices)
cmdBuffer.draw(3, 1, 0, 0)

cmdBuffer.endRendering()
```

#### Pass 2：UI 覆盖层

```
cmdBuffer.beginRendering(
  colorAttachments = [{
    imageView = swapChainImages[imageIndex],
    loadOp = LOAD,    // 保留天空内容
    storeOp = STORE
  }],
  depthAttachment = null  // 无深度
)

cmdBuffer.setViewport(0, {0,0, W,H, 0,1})
cmdBuffer.setScissor(0, {{0,0}, {W,H}})

// === recordUI() ===
// 1. 获取 ImDrawData
// 2. 如果 uiFrameBuffers[currentFrame] 空间不足则 resize
// 3. memcpy ImDrawVert[] → vertexBufferMapped
// 4. memcpy ImDrawIdx[]  → indexBufferMapped

cmdBuffer.bindPipeline(uiPipeline)
cmdBuffer.bindDescriptorSets(
  pipelineBindPoint = GRAPHICS,
  layout = uiPipelineLayout,
  firstSet = 0,
  descriptorSets = [uiDescriptorSets[0]]  // font sampler
)
cmdBuffer.bindVertexBuffers(0, uiVertexBuffers[currentFrame])
cmdBuffer.bindIndexBuffer(uiIndexBuffers[currentFrame], 0, UINT16)

cmdBuffer.pushConstants(
  layout = uiPipelineLayout,
  stageFlags = VERTEX,
  offset = 0, size = 16,
  pValues = &{scale = (2/W, -2/H), translate = (-1, 1)}
)

for each cmdList in ImDrawData.CmdLists:
  for each cmd in cmdList.CmdBuffer:
    cmdBuffer.setScissor(
      offset = {cmd.ClipRect.x, cmd.ClipRect.y},
      extent = {cmd.ClipRect.z - cmd.ClipRect.x,
                cmd.ClipRect.w - cmd.ClipRect.y}
    )
    cmdBuffer.drawIndexed(cmd.ElemCount, 1, cmd.IdxOffset, cmd.VtxOffset, 0)

cmdBuffer.endRendering()

// Present layout
cmdBuffer.pipelineBarrier(
  srcStageMask = BOTTOM_OF_PIPE,
  dstStageMask = PRESENT_SRC,
  imageMemoryBarriers = [
    swapChainImage → ePresentSrcKHR
  ]
)

cmdBuffer.end()
```

---

## 6) Shader 逻辑（atmospheric.slang）

### 6.1 顶点着色器 `vertMain`

```
输入: 无顶点属性（全屏三角）
输出:
  fragCoord = (position.xy, 0, 1)  // 屏幕坐标
  outViewRay = invProjection * vec4(position.xy, -1, 1)  // 未校正的视线方向
```

### 6.2 片段着色器 `fragMain`

```
1. viewRay = vec4(outViewRay.xy, outViewRay.z, 0) * invView
   // 反投影 + 反视图 = 世界空间视线方向

2. 光线与大气球体求交: sphereIntersection(cameraPos, viewRay, ATMOSPHERE_RADIUS)
   // ATMOSPHERE_RADIUS = 6420km（大气层上界）
   // EARTH_RADIUS = 6360km（地球半径）

3. 如果有有效交点，调用 computeIncidentLight(P, viewRay, V_SUN, ...)
   否则: color = vec3(0.53, 0.81, 0.92); return  // 天蓝色

4. computeIncidentLight():
   - 沿视线方向采样 N 个点
   - 每个采样点:
     a) 计算太阳光衰减（视线方向到太阳方向的大气厚度）
     b) 计算 Rayleigh 散射: βR * (1 + cos²θ) / (3(4+2P))
     c) 计算 Mie 散射: βM * phase(a, b, θ) * exp(σ * t)
     d) 累加并指数衰减
   - 叠加太阳圆盘颜色（沿太阳方向最近点采样）
   - 使用 `exp(-opticalDepth)` 衰减

5. toneMapping: ACES filmic
   return max(tonemap(color), 0) + 0.02 ambient
```

### 6.3 天空球渲染的关键

- 深度写入开启，但天空三角形的深度值经过投影后**总是落在 1.0**（NDC far plane）
- `invProjection[1][1] *= -1` 将 Vulkan 的 [-1,1] depth range 反转为 [1,-1]，配合 `LessOrEqual` 测试使天空覆盖所有深度 ≥ 1.0 的像素
- 后续 UI Pass 使用 `loadOp=LOAD` 保留天空内容

---

## 7) UI 控制面板

`AtmosphericRenderer::updateUI()` — ImGui 窗口：

| UI 控件 | 变量 | 范围 |
|---|---|---|
| Checkbox | `animateSun` | 开启/关闭自动太阳运动 |
| Slider | `sunElevation` | 0 ~ π/2（弧度） |
| Slider | `sunAzimuth` | 0 ~ 2π（弧度） |
| Slider | `timeOfDay` | 0 ~ 24（小时） |
| Slider | `turbidity` | 1 ~ 10 |
| Label | FPS / Frame Time | — |
| Label | Atmospheric renderer | — |

---

## 8) 总时序（文字版）

```
┌─────────────────────────────────────────────────────────────┐
│ T=0  CPU  waitForFences() + acquireNextImage()              │
│       CPU  resetFences() + commandBuffer.reset()            │
│       CPU  updateUIFrame() + animateSun()                   │
│                                                             │
│ T=1  CPU  recordCommandBuffer():                            │
│        ├─ pipelineBarrier (depth/color → attachment)        │
│        ├─ beginRendering (sky pass, clear depth=1.0)        │
│        ├─ bindPipeline(skyPipeline)                        │
│        ├─ pushConstants(184 bytes)                          │
│        ├─ draw(3) — full-screen triangle                   │
│        ├─ endRendering()                                   │
│        ├─ beginRendering (ui pass, loadOp=LOAD sky)         │
│        ├─ recordUI() — ImDrawData triangles                │
│        ├─ endRendering()                                   │
│        └─ pipelineBarrier → present layout + cmdBuffer.end()│
│                                                             │
│ T=2  CPU  submit(graphicsQueue, signal=renderFinishedSem)   │
│       CPU  presentQueue.presentKHR()                        │
│       CPU  currentFrame = (currentFrame + 1) % 2           │
└─────────────────────────────────────────────────────────────┘
```

---

## 9) 数据读写关系速查

| 方向 | 资源 | 操作 |
|---|---|---|
| **CPU→GPU** | `PushConstants`（via `cmd.pushConstants`） | 每帧 CPU 填充结构体，GPU shader 直接读取 |
| **CPU→GPU** | `uiVertexBuffer` / `uiIndexBuffer`（mapped） | 每帧 `memcpy` ImDrawData，顶点着色器读取 |
| **CPU→GPU** | `uiFontTexture` | 一次性初始化，GPU 片段着色器读取 |
| **GPU→GPU** | `swapChainImage` | `acquireNextImage` → sky pass 写入 → present |
| **GPU→GPU** | `depthImage` | sky pass 写入（深度=1.0），UI pass 不使用 |

---

## 10) 同步对象

| 对象 | 角色 |
|---|---|
| `presentCompleteSemaphores[currentFrame]` | `acquireNextImage` → sky pass 等待 |
| `renderFinishedSemaphores[currentFrame]` | graphics `submit` → present 等待 |
| `inFlightFences[currentFrame]` | CPU 等待 GPU 完成本帧，CPU 端栅栏 |
| **无 Compute Pass** | 本 Renderer 为纯 Graphics 渲染，无 compute 队列同步 |

---

## 11) 与其他 Example 的差异

| 对比项 | 其他 Example（Shadow/PBR/SSR 等） | 10_atmospheric |
|---|---|---|
| 数据传递 | UBO/SSBO + DescriptorSets | **PushConstants**（无描述符绑定资源） |
| Pass 数量 | Compute → G-buffer → Lighting → Post | **Sky → UI**（极简两 Pass） |
| 深度测试 | 通常 Render to G-buffer / Depth map | **天空球在 far plane 绘制** |
| 纹理资源 | Mesh/Texture/IBL cubemap | **仅 ImGui 字体纹理** |
| CPU-GPU 同步 | 多队列同步（compute/graphics） | **单队列 graphics + fence** |
| 回调/回读 | 可能有的 stats 回读 | **无回读** |

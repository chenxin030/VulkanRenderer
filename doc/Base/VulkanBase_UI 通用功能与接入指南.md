# VulkanBase UI 通用功能与接入指南

## 1. 通用功能总览

`VulkanBase_UI.h/cpp` 提供了一套完整的 ImGui UI 渲染基础设施。实现上采用**自由函数 + inline 成员包装器**的模式：

- 自由函数（`vkrShutdownUI`、`vkrUpdateUIFrame` 等）在 `VulkanBase_UI.cpp` 中实现，接收 `VulkanBase* self` 参数访问成员。
- `VulkanBase` 中通过 inline 成员函数包装调用这些自由函数，子类无需了解内部实现。

所有渲染器通过继承 `VulkanBase` 复用这些能力。

### 1.1 基础设施组件

| 组件                    | 类型                     | 说明                                                                       |
| ----------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `ImGui` Context         | 单例                     | ImGui 上下文，由 `initVulkanUI()` 创建，`shutdownVulkanUI()` 销毁          |
| 字体纹理                | `TextureData`            | ImGui 默认字体的 GPU 纹理（staging buffer → GPU upload）                   |
| `uiDescriptorSetLayout` | DescriptorSetLayout      | UI 渲染的 descriptor set layout，binding=0 绑定字体采样器                  |
| `uiDescriptorPool`      | DescriptorPool           | 包含 1 个 CombinedImageSampler descriptor                                  |
| `uiDescriptorSets`      | DescriptorSets           | 字体纹理的 descriptor set                                                  |
| `uiPipelineLayout`      | PipelineLayout           | PushConstant(Vertex stage, 16 bytes) + 1 descriptor set                    |
| `uiPipeline`            | GraphicsPipeline         | 使用 `imgui.spv`（`vertMain`/`fragMain`）渲染 ImDrawData                   |
| `uiFrameBuffers`        | `vector<UiFrameBuffers>` | 每帧缓冲（按 `swapChainImages.size()` 分配），每帧动态重分配顶点和索引缓冲 |
| `uiEnabled`             | bool                     | 全局 UI 开关，默认 `true`，可置为 `false` 关闭 UI                          |

### 1.2 文件结构

```
src/Base/
├── VulkanBase_UI.h        # UiFrameBuffers / UiPushConsts 数据结构
│                         # 自由函数声明
│                         # VulkanBase 的 inline 成员包装器
│                         # 包含 vulkan/imgui/glm 头文件
└── VulkanBase_UI.cpp      # 自由函数实现
```

`VulkanBase.h` include `VulkanBase_UI.h`，所以只需 `#include <Base/VulkanBase.h>` 即可使用全部 UI 功能。

### 1.3 资源生命周期

```
initVulkanUI()                              GPU资源创建（一次性）
  ├── ImGui::CreateContext()
  ├── 字体纹理上传（staging → device local）
  ├── DescriptorSet（字体采样器绑定）
  ├── UI Pipeline（imgui.spv）
  └── uiFrameBuffers[swapChainImages.size()] 预分配

shutdownVulkanUI()                          GPU资源销毁（程序退出时）
  ├── device.waitIdle()
  └── ImGui::DestroyContext()
```

### 1.4 核心成员函数

| 函数                                       | 职责                                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `initVulkanUI()`                           | 一次性初始化：ImGui context、字体纹理、Descriptor Set、Pipeline、FrameBuffers                                                   |
| `shutdownVulkanUI()`                       | 程序退出时销毁所有 UI 资源（非纯虚，有默认实现）                                                                                |
| `updateUIFrame()`                          | 每帧调用：设置 `ImGuiIO`（DisplaySize、DeltaTime、鼠标）、调用 `ImGui::NewFrame()` → `updateUIPanel()` → `ImGui::Render()`      |
| `recordUICmdBuffer(cmdBuffer, frameIndex)` | 将 `ImDrawData` 拷贝到 GPU buffer，绑定 Pipeline、Descriptor、Vertex/Index Buffer，按 DrawCmd 逐个设置 Scissor 并 `drawIndexed` |
| `recordUICmdBuffer(cmdBuffer)`             | 重载，默认使用 `currentFrame`                                                                                                   |

### 1.5 `UiFrameBuffers` 结构（定义于 `VulkanBase_UI.h`）

```cpp
struct UiFrameBuffers {
    vk::raii::Buffer      vertexBuffer;       // ImDrawVert 缓冲
    vk::raii::DeviceMemory vertexBufferMemory; // HostVisible + HostCoherent
    void*                 vertexMapped;       // CPU 映射指针
    size_t                vertexSize;          // 当前使用量（bytes）
    size_t                vertexBufferSize;    // 已分配大小（用于判断是否需重分配）
    vk::raii::Buffer      indexBuffer;         // ImDrawIdx 缓冲
    vk::raii::DeviceMemory indexBufferMemory;
    void*                 indexMapped;
    size_t                indexSize;
    size_t                indexBufferSize;
};
```

顶点/索引缓冲使用 **HostVisible + HostCoherent** 内存，无需显式 flush/invalidate。每帧根据 ImGui DrawData 大小动态重分配（`vkrReallocateUIMemory`），避免固定大小导致的溢出。

---

## 2. 各渲染器接入清单

每个渲染器需要在以下位置完成适配：

### 2.1 Header 文件（`.h`）

**1. include `VulkanBase.h`（`VulkanBase_UI.h` 由其 transitively 引入）：**

```cpp
#include <Base/VulkanBase.h>
```

**2. 声明 `initUI()` 和 `updateUIPanel()`：**

```cpp
struct XxxRenderer final : VulkanBase
{
public:
    bool initVulkan();
    bool initUI();                         
    bool prepareResource();
    void render();
    void cleanup();

private:
    void updateUIPanel() override;         // 新增 — 纯虚函数，必须 override
};
```
### 2.2 实现文件（`.cpp`）

**1. include：**

```cpp
#include "XxxRenderer.h"
#include <Base/VulkanBase_UI.h>           // 可选：访问 UiFrameBuffers 等类型
#include <glm/gtc/matrix_transform.hpp>
```

**2. 实现 `initUI()` — 委托给自由函数：**

```cpp
bool XxxRenderer::initUI()
{
    return initVulkanUI();
}
```

**3. 实现 `updateUIPanel()` — 具体的 ImGui 面板（重点）：**

```cpp
void XxxRenderer::updateUIPanel()
{
    ImGui::SetNextWindowSize(ImVec2(300.0f, 200.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Xxx Renderer", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::SeparatorText("Settings");
    ImGui::SliderFloat("Value", &someValue, 0.0f, 1.0f);

    // ... 其他控件

    ImGui::End();
}
```

**4. 在 `prepareResource()` 中调用 `initUI()`：**

```cpp
bool XxxRenderer::prepareResource()
{
    // ... 现有资源创建代码 ...

    createXxxDescriptorSets();             // 最后一行列出描述符集

    if (!initUI()) {                       // 新增
        std::cerr << "Failed to initialize UI" << std::endl;
        return false;
    }

    return true;
}
```

**5. 在 `render()` 中调用 `updateUIFrame()`：**

```cpp
void XxxRenderer::render()
{
    // ... existing acquire + reset code ...

    updateUIFrame(currentFrame);            // ① CPU — 构建 ImDrawData（必须在前）
    updateXxxBuffers(currentFrame);
    recordCommandBuffer(imageIndex);         // ② CPU 录制 — recordUICmdBuffer 将 DrawData 写入 GPU buffer

    // ... existing submit + present ...
}
```

**6. 在 `recordCommandBuffer()` 中调用 `recordUICmdBuffer()`：**

在主渲染 pass 的 `endRendering()` 之后、提交之前，添加 UI 渲染 pass：

```cpp
void XxxRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    // ... 主渲染 pass（beginRendering → drawIndexed → endRendering）...

    commandBuffer.endRendering();          // 主渲染结束

    // 新增：UI 渲染 pass（独立 rendering，无 depth）
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
        static_cast<float>(swapChainExtent.width),
        static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUICmdBuffer(commandBuffer, imageIndex);

    // ... 后续 image layout transition 和 commandBuffer.end() ...
}
```

> `recordUICmdBuffer()` 不会调用 `commandBuffer.end()`，命令缓冲由主渲染流程的 `end()` 统一关闭。

---

## 3. 完整接入示例

以下以 `ShadowRenderer` 为例，展示最小化接入：

```cpp
// === ShadowRenderer.h ===
#pragma once
#include <Base/VulkanBase.h>

struct ShadowRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);
    bool initVulkan();
    bool initUI();              // ①
    bool prepareResource();
    void render();
    void cleanup();
    void waitIdle() { device.waitIdle(); }

private:
    void updateUIPanel() override;  // ②

    // ... 现有成员 ...
};
```

```cpp
// === ShadowRenderer.cpp ===
#include "ShadowRenderer.h"
#include <Base/VulkanBase_UI.h>

bool ShadowRenderer::initUI()          // ③
{
    return initVulkanUI();
}

void ShadowRenderer::updateUIPanel()  // ④
{
    ImGui::Begin("Shadow", nullptr, ImGuiWindowFlags_NoCollapse);
    ImGui::SliderInt("Filter Mode", &shadowFilterMode, 0, 2);
    ImGui::SliderFloat("PCF Radius", &pcfRadiusTexels, 0.5f, 8.0f);
    ImGui::End();
}

// prepareResource() 末尾：
//     if (!initUI()) return false;

// render() 中：
//     updateUIFrame(currentFrame);

// recordCommandBuffer() 末尾（endRendering 之后）：
//     commandBuffer.setViewport(...);
//     commandBuffer.setScissor(...);
//     recordUICmdBuffer(commandBuffer, currentFrame);

void ShadowRenderer::cleanup()        // ⑤
{
    device.waitIdle();
    shutdownVulkanUI();
    // ... 销毁其他资源 ...
}
```

---

## 4. UI 功能开关

如需运行时关闭 UI，置 `uiEnabled = false`：

```cpp
// 某处 UI 控制
if (ImGui::Checkbox("Show UI", &renderer.uiEnabled)) { }
```

当 `uiEnabled = false` 时，`updateUIFrame()` 和 `recordUICmdBuffer()` 均直接返回，不产生任何额外开销。

---

## 5. 注意事项

- **`updateUIPanel()` 是 protected 纯虚函数**：外部代码（如主循环）不应直接调用，通过 `updateUIFrame()` 间接触发。
- **不要在 `updateUIPanel()` 中调用 `cleanup` 代码**：`updateUIPanel()` 每帧都可能调用，清理代码应放在 `cleanup()` 或析构函数中。
- **Shader 依赖**：`imgui.spv` 必须存在于 `VK_SHADERS_DIR` 目录，入口函数为 `vertMain` 和 `fragMain`。
- **`swapChainImages.size()` vs `MAX_FRAMES_IN_FLIGHT`**：`uiFrameBuffers` 按 `swapChainImages.size()` 分配（通常为 3），而非 `MAX_FRAMES_IN_FLIGHT=2`。这是因为 `acquireNextImage` 返回的 `imageIndex` 范围是 `[0, swapChainImages.size())`，两者必须一致才不会越界。
- **`shutdownVulkanUI()` 非纯虚函数**：子类**不需要**声明或实现它，直接调用基类版本即可。
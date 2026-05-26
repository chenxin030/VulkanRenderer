# 实例化渲染

[返回目录](../../README.md)

实例化渲染：通过 UBO/SSBO 提供多个实例的变换矩阵，用一次 drawCall 渲染多个实例，。

使用的着色器：`instanced.slang`

## 传入MVP矩阵

和[基础教程](../BasicRender/README.md)不同，上一节把MVP放到了一起（3 * MVP），而实际上渲染同一个场景的同种物体，只有Model矩阵的区别，VP矩阵相同，所以变成了 3 * MVP + VP，也就是从 MVP 数组变为了 M 数组+VP，Model矩阵使用 SSBO，VP矩阵使用UBO。一共是 1SSBO + 1UBO + 1Sampler。

UBO和SSBO的区别在于：
- UBO 比较像小（通常 64KB 左右），SSBO 更大（可到数 MB 甚至更多，取决于硬件）
- UBO 主要用于只读的、频繁使用的小数据（如全局常量、VP 矩阵）。SSBO 支持更灵活的读写（虽然在渲染阶段多数也只读），适合大数组或结构体列表。
- UBO 通常使用 std140 布局，SSBO 通常使用 std430，SSBO 更紧凑。
  - std140：每个元素都必须按 vec4 对齐，结构体成员会被填充到 16 字节边界，例如一个 float 在 std140 中也被视为"占一个 vec4 槽位"（base alignment = 16），所以必须从下一个 16 字节边界开始。总共占用 32 字节，浪费了 12 字节 padding。浪费空间多，但兼容性强
  - std430：保持基本对齐，数组和结构体不会强制对齐到 16 字节，vec3 仍对齐为 16，，但之后 float 只需要按 4 字节对齐（它的自然对齐值），不需要强制对齐到 vec4。总共占用 20 字节，几乎没有浪费。数组元素不再强制 padding 到 16，更紧凑，更省内存
  
```c++
layout(std140) uniform UBO {
    vec3 a;   // 16 bytes
    float b;  // 放不下，需要新 16 字节块
};

layout(std430) buffer SSBO {
    vec3 a;   // 16 bytes
    float b;  // 可紧跟在 a 后面
};
```

创建 SSBO 和 UBO 类似（见 `createUniformBuffers` ，`createStorageBuffers` 函数）区别在参数2 `vk::BufferUsageFlags` 
```c++
void Renderer::createInstancedBuffers() {
    createUniformBuffers(globalUboResources, sizeof(GlobalUBO));
    createStorageBuffers(instancedBufferResources, sizeof(InstanceData) * instanceCount);
}
```

## 创建资源

1. 创建buffer
   - `GlobalUBO`：view / proj
   - `InstanceData[]`：每个实例的 model 矩阵
2. 创建描述符
   - binding 0：全局 UBO
   - binding 1：实例 SSBO
   - binding 2：Sampler
3. 每帧更新
   - 更新相机 UBO（VP）
   - 更新实例 SSBO（所有实例的 model）
4. recordCommand
   - 绑定 pipeline / descriptor set
   - `drawIndexed(indexCount, instanceCount, ...)`

```C++
// createInstancedDescriptorSetLayout()
// binding, descriptorType
std::vector<vk::DescriptorSetLayoutBinding> bindings = {
    {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex
    },
    {
        .binding = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex
    },
    {
        .binding = 2,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment
    }
};

// createInstancedDescriptorPool()
std::vector<vk::DescriptorPoolSize> poolSizes = {
    {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
    {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT }, // storage buffer
    {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
};

// createInstancedPipeline
vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "instanced.spv"));

// createInstancedDescriptorSets()

// 渲染循环render() -> recordCommandBuffer()
commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *instancedPipeline);
commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
// 一次drawcall对应一次描述符集绑定
commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *instancedPipelineLayout, 0, *instancedBufferResources.descriptorSets[currentFrame], nullptr);
const uint32_t instanceCount = 3;
// 区别是参数2：实例数量
commandBuffer.drawIndexed(mesh.indices.size(), instanceCount, 0, 0, 0);

```
## 着色器数据绑定
- 着色器：`shaders/instanced.slang`
- 顶点着色器使用 `SV_InstanceID` 索引 SSBO
- `globalUbo` 提供 view / proj
- `instanceBuffer[instanceID].model` 提供 model
```c++
struct GlobalUBO {
    float4x4 view;
    float4x4 proj;
};
// 对应 常量缓冲区（类似 UBO / D3D CBV）。
ConstantBuffer<GlobalUBO> globalUbo;

struct InstanceData {
    float4x4 model;
};
// 对应 结构化缓冲区（类似 SSBO / D3D SRV for structured buffer）。
StructuredBuffer<InstanceData> instanceBuffer;

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID) {
    VSOutput output;
    float4x4 model = instanceBuffer[instanceID].model;
    ...
}
```
- ConstantBuffer<T> 表示一整块结构体数据本身，所以可以直接访问成员：
cbuffer.myValue、cbuffer.viewMatrix 这种方式。`ConstantBuffer<T>` = 一个 T

- StructuredBuffer<T> 表示结构体数组，所以必须先取一个元素：
buffer[i].modelMatrix、buffer[idx].color。`StructuredBuffer<T>` = 多个 T

如果只需要一个结构体实例，用 ConstantBuffer；
如果需要大量实例（比如实例化渲染的模型矩阵数组），就必须用 StructuredBuffer 并通过下标访问。
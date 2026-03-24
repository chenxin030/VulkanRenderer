# 基础渲染

[返回目录](../../README.md)

---

对[官方新手教程](https://docs.vulkan.org/tutorial/latest/01_Overview.html)进行略微修改，仍是渲染3个带有模型和纹理的物体，重点是理解着色器如何传入数据。

使用的 shader：`shader.slang`

## 传入模型顶点、索引数据
顶点和索引都是通过 `Buffer` 传入，区别在于 `Buffer` 的 `vk::MemoryPropertyFlags`，一个是 `eVertexBuffer`，另一个是 `eIndexBuffer`;

创建 `StagingBuffer`，把顶点/索引 `memcpy` 过去，在通过 `copyBuffer` 把`StagingBuffer`的内容给到 `VertexBuffer`

一种网格只需创建一份顶点缓冲区/索引缓冲区，渲染不同位置的同种网格，区别在于各个物体的Model矩阵（Transform）。

渲染前绑定使用的顶点/索引，`Renderer_rendering.cpp` 244，245行：
```c++
commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
```

## 传入MVP矩阵 & 纹理
和顶点/索引这2个渲染管线自带的输入不同，MVP矩阵和纹理是可选的，需要通过**描述符集**传入。

MVP矩阵使用 `UniformBuffer`， 纹理使用 `Sampler`；为此需要需要：
- 创建描述符集布局
- 把描述符集布局给到渲染管线
- 创建描述符池
- 创建描述符集

### 创建描述符集布局
一个`UniformBuffer`，一个 `Sampler`，见`createDescriptorSetLayout`函数：
```c++
std::array bindings = {
    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
};
```
参数分别是绑定的位置，描述符类型，描述符数量，在管线的哪个阶段使用；对应着色器(`shader.slang`)中的：
```c++
struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;
Sampler2D texture;
```
### 把描述符集布局给到渲染管线布局
让管线知道你会用哪些描述符集，`Renderer_core.cpp` 的 `createGraphicsPipeline` 函数338行：
```c++
vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "shader.spv"));

...

vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ 
    .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, 
    .pushConstantRangeCount = 0 
};

...
```

### 创建描述符池
这里就需要**根据物体数量进行调整**，同时还受同时渲染的帧的数量的影响(`MAX_FRAMES_IN_FLIGHT`)，`createDescriptorPool`函数：
```c++
uint32_t uniformBufferCount = resourceManager->meshUniformBuffer.size();
std::array poolSize{
    vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * uniformBufferCount),
    vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * uniformBufferCount)
};
vk::DescriptorPoolCreateInfo poolInfo{
    .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
    .maxSets = MAX_FRAMES_IN_FLIGHT * uniformBufferCount,
    .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
    .pPoolSizes = poolSize.data()
};
```
场景中会有`uniformBufferCount`(==3)个物体，且同时渲染2个帧，每帧会有各自的数据，所以要 2*3 == 6 的尺寸，`Sampler`同理

### 创建描述符集
`createDescriptorSets`函数：每个帧创建各自的描述符并写入自己的 `UniformBuffer` 和 `Sampler` ，注意要对上 `dstbinding`

## 更新并使用描述符集
在`render`渲染循环中，`updateUniformBuffer` 更新UBO的数据，并通过UBO的 `BuffersMapped` 把更新内容 `memcpy` 进去；

`recordCommandBuffer` 记录绘制命令时，绑定顶点、索引、描述符集
```c++
commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
auto& mesh = resourceManager->meshes[scene->cubeMeshIndex];
commandBuffer.bindVertexBuffers(0, *mesh.vertexBuffer, { 0 });
commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexTypeValue<decltype(mesh.indices)::value_type>::value);
const uint32_t instanceCount = scene ? scene->getMeshInstanceCount(MeshTag::Cube) : 0;
for (uint32_t i = 0; i < instanceCount; ++i) {
    auto& descriptorSets = resourceManager->meshUniformBuffer[i].descriptorSets;
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(mesh.indices.size(), 1, 0, 0, 0);
}
```
这里分别对3个各进行了1次drawCall，而下一节的[实例化渲染](../Instanced/README.zh-CN.md)只需要通过一次命令就可以同时渲染3个物体，此时 `DescriptorSetLayout` 会有所变化。
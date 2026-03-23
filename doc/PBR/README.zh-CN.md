# PBR（Level 3/4）

返回目录：[README.zh-CN.md](../../README.zh-CN.md)


涵盖两个阶段：

- **Level 3：PBR Instanced（直接光照）**
- **Level 4：IBL PBR + Skybox（HDR IBL 预计算 + 天空盒）**

## Level 3：PBR Instanced（直接光照）
- 仍是实例化渲染
- 在片段着色器中实现基础金属/粗糙度 PBR（直接光照）

### 数据结构
位于 `src/Core/ResourceManager.h`
- `SceneUBO`
- `PBRInstanceData`
- `LightUBO`
```c++
struct LightUBO {
	PointLight lights[4];
};

struct PBRInstanceData {
	glm::mat4 model;
	float metallic;
	float roughness;
	alignas(16) glm::vec3 color;
};

struct SceneUBO {
	glm::mat4 projection;
	glm::mat4 view;
	glm::vec3 camPos;
};
```
### 资源与描述符绑定

- binding 0：`SceneUBO`（projection / view / camPos）
- binding 1：`PBRInstanceData[]` SSBO（model / metallic / roughness / color）
- binding 2：`LightUBO`（点光源）

注意看 `createPBRDescriptorPool` 函数如何创建 2UBO + 1SSBO
```c++
void Renderer::createPBRBuffers() {
	createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
	createUniformBuffers(lightUboResources, sizeof(LightUBO));
	if (scene != nullptr) {
		maxInstances = scene->getMaxInstances();
	}
	createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * maxInstances);
}

// createPBRDescriptorSetLayout
std::vector<vk::DescriptorSetLayoutBinding> bindings = {
	{
		.binding = 0,
		.descriptorType = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = 1,
		.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
	},
	{
		.binding = 1,
		.descriptorType = vk::DescriptorType::eStorageBuffer,
		.descriptorCount = 1,
		.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
	},
	{
		.binding = 2,
		.descriptorType = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = 1,
		.stageFlags = vk::ShaderStageFlagBits::eFragment
	}
};

// createPBRDescriptorPool
std::vector<vk::DescriptorPoolSize> poolSizes = {
	{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2 }, // Scene + Light
	{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT }
};

// createPBRPipeline
vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "pbr.spv"));

// createPBRDescriptorSets

// render->updatePBRInstanceBuffers->recordCommandBuffer
```

### PBR直接光照
使用标准的 Cook?Torrance 微表面 BRDF：
- D：法线分布（GGX）：决定高光的形状和“尖锐度”。roughness 越小，分布越集中，高光更尖；越大越宽。
- G：几何遮蔽（Smith?GGX）：遮挡衰减，描述微表面之间互相遮挡、挡住光或视线的程度。roughness 越大，遮蔽越强，镜面能量会被压低。让高光在掠射角时合理变暗。
- F：菲涅耳（Schlick）：角度反射率，描述入射角变化导致反射率变化。视线越接近掠射角（视线与法线夹角大），反射率越高。决定金属/非金属的“反射强弱”和色调（通过 F0 与 metallic 控制）。

#### Metallic 的作用
主要影响两件事：
1. 基础反射率 F0

   ` float3 F0 = lerp((0.04).xxx, albedo, metallic);`
   - 非金属：F0 ≈ 0.04（固定介电体反射率）
   - 金属：F0 = albedo（反射率来自金属本身颜色）

2. 抑制漫反射

   `float3 kD = (1.0 - kS) * (1.0 - metallic);`

金属几乎没有漫反射，因此 metallic 越高，漫反射越少。

---
#### Roughness 的作用
控制镜面高光形状和强度，在 D_GGX 中控制分布的“尖锐度

- roughness 越小 → 高光越锐利、强度集中
- roughness 越大 → 高光更宽、更柔和

## Level 4：IBL PBR + Skybox

Level 4 在 Level 3 的基础上加入 PBR的间接反射部分 IBL

- 漫反射 IBL：irradiance cubemap
- 镜面 IBL：prefiltered cubemap + BRDF LUT
- 额外天空盒渲染 pass

### 流程：预计算 + 渲染天空盒 + 渲染场景

1. 资源加载
   - 生成/加载球体网格
   - 载入 HDR `assets/textures/newport_loft.hdr`（2D float 纹理）
2. IBL 预计算（`generateIBLResources`）
   - equirectangular（2D HDR）→ 环境 cubemap
   - 环境 cubemap → irradiance cubemap（漫反射卷积）
   - 环境 cubemap → prefiltered cubemap（镜面预滤波 + mip 链）
   - 生成 BRDF LUT（2D）
3. 创建 descriptor set layout / pool / set
4. 创建 pipeline
   - `pbribl`（实例化球体）
   - `skybox`（天空盒）
5. 每帧更新
   - Scene / Light / Instance / Params / Skybox UBO
6. 记录与提交
   - 先绘制 skybox，再绘制实例化球体

### IBL 预计算细节

- `envCubemapData`（6 层，cube）
- `irradianceCubemapData`（6 层，cube）
- `prefilteredEnvMapData`（6 层，cube，带 mip 链）
- `brdfLutData`（2D 图像）

每个 cubemap 面通常通过临时 2D view（选择 layer/mip）渲染，再调用 `beginRendering()` / `endRendering()`。

### Descriptor bindings（IBL PBR）

参考 `createIBLPBRDescriptorSetLayout` 与 `shaders/pbribl.slang`：

- binding 0：`SceneUBO`
- binding 1：`PBRInstanceData[]`
- binding 2：`LightUBO`
- binding 3：irradiance cubemap sampler
- binding 4：prefiltered cubemap sampler
- binding 5：BRDF LUT（2D）
- binding 6：`ParamsUBO`（exposure / gamma）

### Descriptor bindings（Skybox）

参考 `createSkyboxDescriptorSetLayout` 与 `shaders/skybox.slang`：

- binding 0：`SkyboxUBO`（invProjection / invView）
- binding 1：`ParamsUBO`
- binding 2：environment cubemap sampler



```c++
struct ParamsUBO {
	float exposure;
	float gamma;
};

struct SkyboxUBO {
	glm::mat4 invProjection;
	glm::mat4 invView;
};

void Renderer::createIBLPBRBuffers()
{
	createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
	createUniformBuffers(lightUboResources, sizeof(LightUBO));
	createUniformBuffers(paramsUboResources, sizeof(ParamsUBO));
	createUniformBuffers(skyboxUboResources, sizeof(SkyboxUBO));
	if (scene != nullptr) {
		maxInstances = scene->getMaxInstances();
	}
	createStorageBuffers(pbrInstanceBufferResources, sizeof(PBRInstanceData) * maxInstances);
}

// 下面是主渲染循环用到的------------------------------
// createIBLPBRDescriptorSetLayout
std::vector<vk::DescriptorSetLayoutBinding> bindings = {
	{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },	// sceneUbo
	{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },	// instanceData
	{.binding = 2, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },										// lightUbo
	{.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },								// irradianceCubemapSampler
	{.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },								// prefilteredMapSampler
	{.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },								// BRDFLUTsampler
	{.binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },										// paramsUbo
};

// createIBLPBRDescriptorPool
std::vector<vk::DescriptorPoolSize> poolSizes = {
   {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
   {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
   {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 3u },
};

// createIBLPBRPipeline
vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "pbribl.spv"));

// --------------------------------------------------------
// createSkyboxDescriptorSetLayout
std::vector<vk::DescriptorSetLayoutBinding> bindings = {
	{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
};

// createSkyboxDescriptorPool
std::vector<vk::DescriptorPoolSize> poolSizes = {
	{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u },
	{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
};

// createSkyboxPipeline
vk::raii::ShaderModule shaderModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "skybox.spv"));
```

### 预计算过程
维度选择：
- 辐照度贴图（Irradiance Map）：环境贴图的卷积，相当于模糊后的环境贴图
-  预过滤环境图（Prefiltered Environment Map）：用于镜面反射光照，根据不同粗糙度进行卷积，所以需要Mipmap
-  BRDF 查找表（BRDF LUT）：用于镜面反射的积分近似
```c++
const vk::Format envFormat = vk::Format::eR16G16B16A16Sfloat;
const uint32_t envDim = 512u;
const uint32_t irradianceDim = 64u;
const uint32_t prefilterDim = 512u;
const uint32_t prefilterMipLevels = static_cast<uint32_t>(std::floor(std::log2(prefilterDim))) + 1u;
const uint32_t brdfDim = 512u;

// 如果是cubemap，ArrayLayer == 6，vk::ImageCreateFlags == eCubeCompatible。
// 因为是通过着色器计算然后输出，所以会作为纹理附件，同时会在渲染循环中采样使用：vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
createImage(envDim, envDim, 1, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
	vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, envCubemapData);
createImage(irradianceDim, irradianceDim, 1, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
	vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, irradianceCubemapData);
createImage(prefilterDim, prefilterDim, prefilterMipLevels, 6, vk::ImageCreateFlagBits::eCubeCompatible, envFormat, vk::ImageTiling::eOptimal,
	vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, prefilteredEnvMapData);
createImage(brdfDim, brdfDim, 1, 1, {}, envFormat, vk::ImageTiling::eOptimal,
	vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, brdfLutData);

// ImageView、ImageSampler类似

// 所用的着色器：
vk::raii::ShaderModule filterModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "filtercube.spv"));
vk::raii::ShaderModule irradianceModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "irradiancecube.spv"));
vk::raii::ShaderModule prefilterModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "prefilterenvmap.spv"));
vk::raii::ShaderModule brdfModule = createShaderModule(readFile(std::string(VK_SHADERS_DIR) + "genbrdflut.spv"));

// 因为是采样Cubemap，所以创建一个Cube用来获取采样向量
Mesh cubeMesh;
generateCube(cubeMesh);
createVertexBuffer(cubeMesh);
createIndexBuffer(cubeMesh);
```
### filtercube
**把HDR文件转为cubemap**：HDR和世界地图类似，把圆的地球绘制成了一个2D图片，所以看起来会有一定的拉伸。

而`filtercube`则反过来，把2D转为CubeMap，为此需要计算3维采样方向（Cubemap）所对应的2维采样UV（HDR）

---
### 辐照度图irradianceCube
**用来处理间接反射中的漫反射部分**

漫反射是各向同性的，与视角无关；当人眼/摄像机看向物体表面某个点，在该点能看到的间接光照的漫反射是该点的法线半球内所有方向的入射光的总和，所以要对法线半球内的所有方向的入射光进行积分，实际做法就算采样 Cubemap 后求和，最后积分得到的光量/采样次数，就是该点的间接光照漫反射部分的光量；因为是求和取平均，所以得到贴图就是一张模糊的环境立方体贴图。

使用的时候，用表面法线对辐照度图进行采样，就是这个地方的间接漫反射了。

---
### 镜面反射
镜面反射采用**预滤波HDR环境贴图prefilterenvmap**和**BRDF_LUT**.

对于间接镜面反射的部分，因为参数太多，直接积分不可能，所以将积分分成了环境光照部分和BRDF部分；

环境光照部分：根据表面粗糙度的多少，决定采样的模糊程度，这个用mipmap实现，越粗糙，越模糊，mipmap用得越大（图像越小）；

粗糙度影响的是镜面波瓣的大小，表面越粗糙，镜面波瓣越大，光线会从镜面波瓣的范围内进入表面；在镜面波瓣内采样时（就像上面在法线半球内采样），光线进入表面得越散，看到的越模糊——mipmap等级就越大；在镜面波瓣内采样时，会用到重要性采样，用于均匀分布采样,它比纯随机采样收敛更快。

而BRDF部分，**把菲涅尔项F提取出来**，把BRDF变成了F*scale+bias，而LUT要做的，就是预计算scale和bias放到贴图当中；

最后需要计算间接镜面反射的时候，就用视线的反射方向（R = reflect(-V, N);）采样prefilterenvmap作为反射的颜色值reflection，然后乘以BRDF部分(F*LUT.x+LUT.y)。

具体数学计算可见[LearnOpenGL CN 镜面IBL](https://learnopengl-cn.github.io/07%20PBR/03%20IBL/02%20Specular%20IBL/#hdr)

PS：对于LearnOpenGL对于镜面波瓣的图我觉得不太直观，实际上，可以把右边黄色线当做人眼看向物体表面，而他能看到的镜面反射的内容是左边镜面波瓣里的几个方向的间接光；所以制作**预滤波HDR环境贴图prefilterenvmap**的时候，是对镜面波瓣里各个方向的内容进行采样求和取平均；
那么怎么知道人眼是从哪个方向看向表面这个点呢？ Epic Games 假设视角方向——总是等于输出采样方向ωo，以作进一步近似。

---
### 代码层面
`generateIBLResources()`

filtercube、辐照度图irradianceCube和预滤波HDR环境贴图prefilterenvmap所用到的输入都是1 PushConstant + 1 Sampeler
1. filtercube输入 MVP + HDR，输出 CubeMap
2. 辐照度图irradianceCube 输入 (MVP + 采样步长) + CubeMap
3. 预滤波HDR环境贴图prefilterenvmap输入 (MVP + Roughness + 采样次数) + CubeMap

123都和cubemap有关，所以会有 `for (uint32_t face = 0; face < 6; face++) {...}`

在之前都是直接渲染到屏幕上面，所以是把 `SwapChainImage`用作颜色附件；现在是把纹理当做颜色附件，所以：
```c++
for (uint32_t face = 0; face < 6; face++)
{
	vk::raii::ImageView faceView(device, vk::ImageViewCreateInfo{
		.image = irradianceCubemapData.textureImage,
		.viewType = vk::ImageViewType::e2D,
		.format = envFormat,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, face, 1 }
		});

	vk::RenderingAttachmentInfo colorAttachment{
		.imageView = *faceView,    // 之前是swapChainImageViews[imageIndex]
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
	};
	vk::RenderingInfo renderingInfo{
		.renderArea = { {0, 0}, {irradianceDim, irradianceDim} },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachment
	};

   ...
```

因为 1 的结果（颜色附件）会作为 2，3 的输入，所以需要转换格式（1039行），后面同理：
```c++
transitionImageLayoutCmd(*cmd, envCubemapData.textureImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	0, 1, 0, 6, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);
```
最终会得到3个贴图

---
渲染循环主体使用的输入和天空盒的输入：
`createIBLPBRDescriptorSets`
`createSkyboxDescriptorSets`  

---
渲染循环主体：`render()`->`updateIBLPBRBuffers()`->`recordCommandBuffer()`

先天空盒再是场景，这样渲染场景时，深度测试就可以直接通过，不需要进行对比，直接覆盖在天空盒上面。
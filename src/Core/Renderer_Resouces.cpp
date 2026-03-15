#include "Renderer.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void Renderer::createVertexBuffer(Mesh& mesh) {

    auto& vertices = mesh.vertices;
    auto& vertexBuffer = mesh.vertexBuffer;
    auto& vertexBufferMemory = mesh.vertexBufferMemory;

    vk::DeviceSize         bufferSize = sizeof(vertices[0]) * vertices.size();
    vk::raii::Buffer       stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}

void Renderer::createIndexBuffer(Mesh& mesh)
{
    auto& indices = mesh.indices;
    auto& indexBuffer = mesh.indexBuffer;
    auto& indexBufferMemory = mesh.indexBufferMemory;

    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vk::raii::Buffer       stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
}

void Renderer::createUniformBuffers(MeshBuffer& meshResource, vk::DeviceSize size) {
    auto& uniformBuffers = meshResource.Buffers;
    auto& uniformBuffersMemory = meshResource.BuffersMemory;
    auto& uniformBuffersMapped = meshResource.BuffersMapped;

    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, size));
    }
}

void Renderer::createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size)
{
    auto& uniformBuffers = meshResource.Buffers;
    auto& uniformBuffersMemory = meshResource.BuffersMemory;
    auto& uniformBuffersMapped = meshResource.BuffersMapped;

    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, size));
    }
}

#if RENDERING_LEVEL == 1
bool Renderer::createDescriptorPool() {
    try
    {
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
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createDescriptorSets() {
    for (auto& resource : resourceManager->meshUniformBuffer) {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo        allocInfo{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        try {
            resource.descriptorSets.clear();
            resource.descriptorSets = device.allocateDescriptorSets(allocInfo);
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to allocate descriptor sets: " << e.what() << std::endl;
            throw;
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo uniformBufferInfo{ .buffer = resource.Buffers[i], .offset = 0, .range = sizeof(MVP) };
            vk::DescriptorImageInfo imageInfo{
                .sampler = resourceManager->textures[0].textureSampler,
                .imageView = resourceManager->textures[0].textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
            std::array descriptorWrites{
                vk::WriteDescriptorSet{
                    .dstSet = resource.descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &uniformBufferInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = resource.descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &imageInfo
                }
            };
            device.updateDescriptorSets(descriptorWrites, {});
        }
    }
}
#endif

void Renderer::createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {

    try {
        vk::BufferCreateInfo bufferInfo{
          .size = size,
          .usage = usage,
          .sharingMode = vk::SharingMode::eExclusive
        };

        buffer = vk::raii::Buffer(device, bufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{
          .allocationSize = memRequirements.size,
          .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);

        // Bind memory to buffer
        buffer.bindMemory(*bufferMemory, 0);
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create staging buffer: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
    try {
        vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
        vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
        commandCopyBuffer.end();
        graphicsQueue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer }, nullptr);
        graphicsQueue.waitIdle();
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to copy buffer: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::loadModels() {
    auto& meshes = resourceManager->meshes;
#if RENDERING_LEVEL == 5
    generateCube(meshes[0]);
    createVertexBuffer(meshes[0]);
    createIndexBuffer(meshes[0]);

    generateSphere(meshes[1], 1.0f, 64);
    createVertexBuffer(meshes[1]);
    createIndexBuffer(meshes[1]);
#elif RENDERING_LEVEL >= 3
    generateSphere(meshes[0], 1.0f, 100);
    createVertexBuffer(meshes[0]);
    createIndexBuffer(meshes[0]);
#endif
#if RENDERING_LEVEL == 4
    skyboxTriangleMesh.vertices = {
        { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f } },
        { {  3.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 2.0f, 0.0f } },
        { { -1.0f,  3.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 2.0f } },
    };
    skyboxTriangleMesh.indices = { 0, 1, 2 };
    createVertexBuffer(skyboxTriangleMesh);
    createIndexBuffer(skyboxTriangleMesh);
#endif
#if RENDERING_LEVEL < 3
    for (int i = 0; i < resourceManager->modelPath.size(); ++i) {
        loadModel(resourceManager->modelPath[i], meshes[i]);
        createVertexBuffer(meshes[i]);
        createIndexBuffer(meshes[i]);
    }
#endif
}

void Renderer::loadTextures() {
    for (int i = 0; i < resourceManager->texPath.size(); ++i) {
        const auto& path = resourceManager->texPath[i];
#if RENDERING_LEVEL == 4
        const bool isHdr = path.size() >= 4 && path.substr(path.size() - 4) == ".hdr";
        if (isHdr) {
            LoadHDRTextureFromFile(path, resourceManager->textures[i]);
            vk::SamplerCreateInfo samplerInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                .mipLodBias = 0.0f,
                .anisotropyEnable = vk::False,
                .maxAnisotropy = 1.0f,
                .compareEnable = vk::False,
                .compareOp = vk::CompareOp::eAlways,
                .minLod = 0.0f,
                .maxLod = 0.0f
            };
            resourceManager->textures[i].textureSampler = vk::raii::Sampler(device, samplerInfo);
            continue;
        }
#endif
        LoadTextureFromFile(path, resourceManager->textures[i]);
        createTextureSampler(resourceManager->textures[i].textureSampler);
    }
}

void Renderer::LoadHDRTextureFromFile(const std::string& path, TextureData& texData)
{
    int texWidth = 0, texHeight = 0, texChannels = 0;
    float* pixels = stbi_loadf((VK_TEXTURE_DIR + path).c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("failed to load HDR texture image: " + path + "\n");
    }

    texData.mipLevels = 1;
    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth) * static_cast<vk::DeviceSize>(texHeight) * 4u * sizeof(float);

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();

    stbi_image_free(pixels);

    createImage(static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), texData.mipLevels, vk::Format::eR32G32B32A32Sfloat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, texData);

    transitionImageLayout(texData.textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, texData.mipLevels);
    copyBufferToImage(stagingBuffer, texData.textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    transitionImageLayout(texData.textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, texData.mipLevels);

    texData.textureImageView = createImageView(texData.textureImage, vk::Format::eR32G32B32A32Sfloat, vk::ImageAspectFlagBits::eColor, texData.mipLevels);
}
void Renderer::LoadTextureFromFile(const std::string& path, TextureData& texData)
{
    int            texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load((VK_TEXTURE_DIR + path).c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;
    auto& mipLevels = texData.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image: " + path + "\n");
    }

    vk::raii::Buffer       stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, mipLevels, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, texData);

    transitionImageLayout(texData.textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(stagingBuffer, texData.textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    generateMipmaps(texData.textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);

    texData.textureImageView = createImageView(texData.textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
}

void Renderer::cleanupUBO() {
    for (auto& ubo : resourceManager->meshUniformBuffer) {
        for (size_t i = 0; i < ubo.BuffersMemory.size(); i++)
        {
            if (ubo.BuffersMapped[i] != nullptr)
            {
                ubo.BuffersMemory[i].unmapMemory();
            }
        }

        // Clear vectors to release resources
        ubo.Buffers.clear();
        ubo.BuffersMemory.clear();
        ubo.BuffersMapped.clear();
        ubo.descriptorSets.clear();
    }

#if RENDERING_LEVEL == 5
	auto unmapMeshBuffer = [](MeshBuffer& buffer) {
		for (size_t i = 0; i < buffer.BuffersMemory.size(); i++)
		{
			if (i < buffer.BuffersMapped.size() && buffer.BuffersMapped[i] != nullptr)
			{
				buffer.BuffersMemory[i].unmapMemory();
			}
		}
		buffer.Buffers.clear();
		buffer.BuffersMemory.clear();
		buffer.BuffersMapped.clear();
		buffer.descriptorSets.clear();
	};

	unmapMeshBuffer(sceneUboResources);
	unmapMeshBuffer(shadowUboResources);
	unmapMeshBuffer(shadowParamsUboResources);
	unmapMeshBuffer(shadowInstanceBufferResources);
#endif
}

void Renderer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData)
{
    createImage(width, height, mipLevels, 1, {}, format, tiling, usage, properties, texData);
}

void Renderer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t arrayLayers, vk::ImageCreateFlags flags, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData)
{
    auto& image = texData.textureImage;
    auto& imageMemory = texData.textureImageMemory;

    vk::ImageCreateInfo imageInfo{
        .flags = flags,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = mipLevels,
        .arrayLayers = arrayLayers,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive
    };

    image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
    };
    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(imageMemory, 0);
}
vk::raii::ImageView Renderer::createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
{
    vk::ImageViewCreateInfo viewInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {aspectFlags, 0, mipLevels, 0, 1} };
    return vk::raii::ImageView(device, viewInfo);
}
void Renderer::createTextureSampler(vk::raii::Sampler& textureSampler)
{
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo        samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = vk::LodClampNone
    };
    textureSampler = vk::raii::Sampler(device, samplerInfo);
}

void Renderer::transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
{
    auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{ .oldLayout = oldLayout, .newLayout = newLayout, .image = image, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1} };

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }
    commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
    endSingleTimeCommands(*commandBuffer);
}
void Renderer::copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height)
{
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
    vk::BufferImageCopy                      region{ .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0, .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, .imageOffset = {0, 0, 0}, .imageExtent = {width, height, 1} };
    commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
    endSingleTimeCommands(*commandBuffer);
}
std::unique_ptr<vk::raii::CommandBuffer> Renderer::beginSingleTimeCommands()
{
    vk::CommandBufferAllocateInfo            allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

    vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
    commandBuffer->begin(beginInfo);

    return commandBuffer;
}

void Renderer::endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
    graphicsQueue.submit(submitInfo, nullptr);
    graphicsQueue.waitIdle();
}

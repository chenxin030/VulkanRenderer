#include "VulkanBase.h"
#include <stb_image.h>

void VulkanBase::createVertexBuffer(Mesh& mesh) {
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

void VulkanBase::createSkinnedVertexBuffer(Mesh& mesh) {
    auto& vertices = mesh.vertices;
    auto& jointIndices = mesh.jointIndices;
    auto& jointWeights = mesh.jointWeights;
    auto& vertexBuffer = mesh.vertexBuffer;
    auto& vertexBufferMemory = mesh.vertexBufferMemory;

    struct SkinnedVertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::uvec4 jointIndices;
        glm::vec4 jointWeights;
    };

    std::vector<SkinnedVertex> skinnedVerts;
    skinnedVerts.reserve(vertices.size());

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        SkinnedVertex sv;
        sv.pos = vertices[i].pos;
        sv.normal = vertices[i].normal;
        if (i < jointIndices.size() && i < jointWeights.size())
        {
            sv.jointIndices = glm::uvec4(jointIndices[i][0], jointIndices[i][1],
                                         jointIndices[i][2], jointIndices[i][3]);
            sv.jointWeights = glm::vec4(jointWeights[i][0], jointWeights[i][1],
                                        jointWeights[i][2], jointWeights[i][3]);
        }
        else
        {
            sv.jointIndices = glm::uvec4(0, 0, 0, 0);
            sv.jointWeights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        }
        skinnedVerts.push_back(sv);
    }

    vk::DeviceSize bufferSize = sizeof(SkinnedVertex) * skinnedVerts.size();
    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, skinnedVerts.data(), static_cast<size_t>(bufferSize));
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}
void VulkanBase::createIndexBuffer(Mesh& mesh) {
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
void VulkanBase::createUniformBuffers(MeshBuffer& meshResource, vk::DeviceSize size) {
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
void VulkanBase::createUniformBuffers(MeshBuffer& meshResource, vk::DeviceSize size, uint32_t count) {
    auto& uniformBuffers = meshResource.Buffers;
    auto& uniformBuffersMemory = meshResource.BuffersMemory;
    auto& uniformBuffersMapped = meshResource.BuffersMapped;

    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (uint32_t i = 0; i < count; i++) {
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, size));
    }
}
void VulkanBase::createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size) {
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
void VulkanBase::createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size, vk::BufferUsageFlags usage){
    auto& uniformBuffers = meshResource.Buffers;
    auto& uniformBuffersMemory = meshResource.BuffersMemory;
    auto& uniformBuffersMapped = meshResource.BuffersMapped;

    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(size, usage, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, size));
    }
}
void VulkanBase::createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size, vk::BufferUsageFlags usage, uint32_t count){
    auto& uniformBuffers = meshResource.Buffers;
    auto& uniformBuffersMemory = meshResource.BuffersMemory;
    auto& uniformBuffersMapped = meshResource.BuffersMapped;

    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (uint32_t i = 0; i < count; i++) {
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(size, usage, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, size));
    }
}
void VulkanBase::createBuffer(
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
void VulkanBase::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
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
void VulkanBase::LoadHDRTextureFromFile(const std::string& path, TextureData& texData)
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
void VulkanBase::LoadTextureFromFile(const std::string& path, TextureData& texData) {
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
void VulkanBase::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData) {
    createImage(width, height, mipLevels, 1, {}, format, tiling, usage, properties, texData);
}
void VulkanBase::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t arrayLayers, vk::ImageCreateFlags flags, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData) {
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
vk::raii::ImageView VulkanBase::createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
    vk::ImageViewCreateInfo viewInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {aspectFlags, 0, mipLevels, 0, 1} };
    return vk::raii::ImageView(device, viewInfo);
}
void VulkanBase::createTextureSampler(vk::raii::Sampler& textureSampler)
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
void VulkanBase::transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
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
void VulkanBase::copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
    vk::BufferImageCopy                      region{ .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0, .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, .imageOffset = {0, 0, 0}, .imageExtent = {width, height, 1} };
    commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
    endSingleTimeCommands(*commandBuffer);
}
std::unique_ptr<vk::raii::CommandBuffer> VulkanBase::beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo            allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

    vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
    commandBuffer->begin(beginInfo);

    return commandBuffer;
}
void VulkanBase::endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
    graphicsQueue.submit(submitInfo, nullptr);
    graphicsQueue.waitIdle();
}

bool VulkanBase::createDepthResources()
{
    try
    {
        vk::Format depthFormat = findDepthFormat();
        vk::ImageUsageFlags depthUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

        // Create depth image + view. Sampler is not required for depth attachment usage.
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            depthFormat,
            vk::ImageTiling::eOptimal,
            depthUsage,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            depthData);

        depthData.textureImageView = createImageView(depthData.textureImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
        depthImageLayout = vk::ImageLayout::eUndefined;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create depth resources: " << e.what() << std::endl;
        return false;
    }
}
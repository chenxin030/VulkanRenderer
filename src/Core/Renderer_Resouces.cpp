#include "Renderer.h"

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
void Renderer::createResouceBuffer(std::vector<Mesh>& resources) {
	for (auto& resource : resources) {
		createVertexBuffer(resource);
        createIndexBuffer(resource);
	}
}

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
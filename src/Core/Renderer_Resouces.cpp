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

void Renderer::createUniformBuffers(EntityResource& entityResource) {
    auto& uniformBuffers = entityResource.uniformBuffers;
    auto& uniformBuffersMemory = entityResource.uniformBuffersMemory;
    auto& uniformBuffersMapped = entityResource.uniformBuffersMapped;

    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
    }
}

bool Renderer::createDescriptorPool() {
    try
    {
        uint32_t uniformBufferCount = resourceManager->entityManager.entityResource.size();
        vk::DescriptorPoolSize       poolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT * uniformBufferCount);
        vk::DescriptorPoolCreateInfo poolInfo{ .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = 1, .pPoolSizes = &poolSize };
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::createDescriptorSets() {
    auto& entityResource = resourceManager->entityManager.entityResource;
    size_t entityCount = entityResource.size();

    for (auto& resource : resourceManager->entityManager.entityResource) {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo        allocInfo{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };
        resource.descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo bufferInfo{ .buffer = resource.uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject) };
            vk::WriteDescriptorSet   descriptorWrite{ .dstSet = resource.descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo };
            device.updateDescriptorSets(descriptorWrite, {});
        }
    }
}

void Renderer::createResouceBuffer() {
    auto& meshes = resourceManager->entityManager.meshes;
    auto& entityResource = resourceManager->entityManager.entityResource;
    for (int i = 0; i < meshes.size(); ++i) {
		createVertexBuffer(meshes[i]);
        createIndexBuffer(meshes[i]);
        createUniformBuffers(entityResource[i]);
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
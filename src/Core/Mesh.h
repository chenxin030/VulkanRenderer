#pragma once

#include <array>
#include <cassert>
#include <glm/glm.hpp>

#include <vulkan/vulkan_raii.hpp>

struct Vertex {
    glm::vec2 position;
    glm::vec3 normal;

    bool operator==(const Vertex& other) const {
        return position == other.position &&
            normal == other.normal;
    }

    static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription(
            0,
            // binding
            sizeof(Vertex),
            // stride
            vk::VertexInputRate::eVertex // inputRate
        );
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        constexpr std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {
          vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(Vertex, position)
          },
          vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, normal)
          }
        };
        return attributeDescriptions;
    }
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
};
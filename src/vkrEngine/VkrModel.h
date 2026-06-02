#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

// ---- Vertex ----

struct VkrVertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent; // xyz: tangent, w: handedness sign (1 or -1)

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return { 0, sizeof(VkrVertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        return {
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VkrVertex, pos) },
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VkrVertex, normal) },
            vk::VertexInputAttributeDescription{ 2, 0, vk::Format::eR32G32Sfloat, offsetof(VkrVertex, texCoord) },
            vk::VertexInputAttributeDescription{ 3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VkrVertex, tangent) },
        };
    }
};

// ---- Sub-mesh ----

struct VkrSubMesh
{
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t  vertexOffset = 0; // base vertex offset within the shared vertex buffer
    int32_t  materialIndex = -1; // -1 = no material
};

// ---- Material (CPU-side data, GPU resources created by renderer) ----

struct VkrMaterialTexture
{
    std::string path; // texture file path, relative
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory imageMemory = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

struct VkrMaterial
{
    std::string name;

    // glTF PBR factors
    glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
    float     metallicFactor = 1.0f;
    float     roughnessFactor = 1.0f;
    glm::vec3 emissiveFactor{ 0.0f, 0.0f, 0.0f };

    // Texture paths (relative to the glTF directory)
    std::string baseColorTexturePath;
    std::string normalTexturePath;
    std::string metallicRoughnessTexturePath;
    std::string emissiveTexturePath;

    // GPU resources (created by renderer)
    VkrMaterialTexture gpuBaseColorTexture;
    VkrMaterialTexture gpuNormalTexture;
    VkrMaterialTexture gpuMetallicRoughnessTexture;
    VkrMaterialTexture gpuEmissiveTexture;
    vk::raii::Sampler textureSampler = nullptr;
    vk::raii::DescriptorSet descriptorSet = nullptr;

    // Whether GPU resources have been created
    bool gpuResourcesCreated = false;
};

// ---- Model ----

struct VkrModel
{
    std::string name;

    // Geometry
    std::vector<VkrVertex>   vertices;
    std::vector<uint32_t>    indices;
    std::vector<VkrSubMesh>  subMeshes;

    // Materials (heap-allocated to avoid copy issues with vk::raii members)
    std::vector<std::unique_ptr<VkrMaterial>> materials;

    // GPU buffers
    vk::raii::Buffer       vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer       indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    // ---- Statistics ----
    uint32_t totalTriangles = 0;
    uint32_t totalVertices = 0;
    uint32_t totalIndices = 0;

    // ---- Bounding Box (world space, computed from vertex positions) ----
    glm::vec3 aabbMin{ 0.0f };
    glm::vec3 aabbMax{ 0.0f };

    /**
     * Load a glTF 2.0 file (either .gltf or .glb).
     * @param filePath    Absolute or relative path to the glTF file
     * @param textureBasePath  Directory for resolving relative texture paths
     * @return true on success
     */
    bool loadFromFile(const std::string& filePath, const std::string& textureBasePath);

    VkrModel() = default;
    VkrModel(VkrModel&&) noexcept = default;
    VkrModel& operator=(VkrModel&&) noexcept = default;
    VkrModel(const VkrModel&) = delete;
    VkrModel& operator=(const VkrModel&) = delete;
};

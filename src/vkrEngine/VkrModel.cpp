#include "VkrModel.h"

#include <tiny_gltf.h>

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ---- Helper: compute per-vertex tangents from positions, normals, texcoords ----
static void computeTangents(std::vector<VkrVertex>& vertices, const std::vector<uint32_t>& indices)
{
    std::vector<glm::vec3> tanAccum(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitanAccum(vertices.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        uint32_t i0 = indices[i + 0];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];

        const VkrVertex& v0 = vertices[i0];
        const VkrVertex& v1 = vertices[i1];
        const VkrVertex& v2 = vertices[i2];

        glm::vec3 edge1 = v1.pos - v0.pos;
        glm::vec3 edge2 = v2.pos - v0.pos;
        glm::vec2 deltaUV1 = v1.texCoord - v0.texCoord;
        glm::vec2 deltaUV2 = v2.texCoord - v0.texCoord;

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y + 1e-8f);

        glm::vec3 tangent;
        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

        tanAccum[i0] += tangent;
        tanAccum[i1] += tangent;
        tanAccum[i2] += tangent;

        glm::vec3 bitangent;
        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

        bitanAccum[i0] += bitangent;
        bitanAccum[i1] += bitangent;
        bitanAccum[i2] += bitangent;
    }

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        glm::vec3 n = vertices[i].normal;
        glm::vec3 t = tanAccum[i];

        // Gram-Schmidt orthogonalize
        t = glm::normalize(t - n * glm::dot(n, t));
        if (glm::length(t) < 0.001f)
        {
            // Fallback: pick an arbitrary tangent perpendicular to normal
            t = (std::abs(n.x) > 0.9f)
                ? glm::normalize(glm::cross(n, glm::vec3(0.0f, 1.0f, 0.0f)))
                : glm::normalize(glm::cross(n, glm::vec3(1.0f, 0.0f, 0.0f)));
        }

        // Handedness
        float handedness = (glm::dot(glm::cross(n, t), bitanAccum[i]) < 0.0f) ? -1.0f : 1.0f;

        vertices[i].tangent = glm::vec4(t, handedness);
    }
}

// ---- glTF Loading ----

bool VkrModel::loadFromFile(const std::string& filePath, const std::string& textureBasePath)
{
    tinygltf::Model    gltfModel;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    // Detect binary vs ASCII glTF by extension
    bool isBinary = filePath.ends_with(".glb");
    bool ok = isBinary
        ? loader.LoadBinaryFromFile(&gltfModel, &err, &warn, filePath)
        : loader.LoadASCIIFromFile(&gltfModel, &err, &warn, filePath);

    if (!warn.empty()) std::cout << "[VkrModel] glTF warning: " << warn << std::endl;
    if (!err.empty())  std::cerr << "[VkrModel] glTF error: " << err << std::endl;
    if (!ok) return false;

    // Extract model name from filename
    name = fs::path(filePath).stem().string();

    // ---- Step 1: Load Textures ----
    // glTF images may be embedded (bufferView) or external files
    for (const auto& gltfImage : gltfModel.images)
    {
        // We'll resolve texture paths when processing materials
    }

    // ---- Step 2: Load Materials ----
    materials.clear();
    for (const auto& gltfMat : gltfModel.materials)
    {
        auto mat = std::make_unique<VkrMaterial>();
        mat->name = gltfMat.name;

        // PBR factors
        if (gltfMat.pbrMetallicRoughness.baseColorFactor.size() == 4)
        {
            const auto& f = gltfMat.pbrMetallicRoughness.baseColorFactor;
            mat->baseColorFactor = glm::vec4(f[0], f[1], f[2], f[3]);
        }
        mat->metallicFactor = static_cast<float>(gltfMat.pbrMetallicRoughness.metallicFactor);
        mat->roughnessFactor = static_cast<float>(gltfMat.pbrMetallicRoughness.roughnessFactor);
        if (gltfMat.emissiveFactor.size() == 3)
        {
            const auto& f = gltfMat.emissiveFactor;
            mat->emissiveFactor = glm::vec3(f[0], f[1], f[2]);
        }

        // Base color texture
        int bcIdx = gltfMat.pbrMetallicRoughness.baseColorTexture.index;
        if (bcIdx >= 0)
        {
            const auto& tex = gltfModel.textures[bcIdx];
            if (tex.source >= 0)
            {
                const auto& img = gltfModel.images[tex.source];
                if (!img.uri.empty())
                {
                    mat->baseColorTexturePath = img.uri;
                }
                // Embedded images handled separately
            }
        }

        // Normal texture
        int nIdx = gltfMat.normalTexture.index;
        if (nIdx >= 0)
        {
            const auto& tex = gltfModel.textures[nIdx];
            if (tex.source >= 0)
            {
                const auto& img = gltfModel.images[tex.source];
                if (!img.uri.empty())
                {
                    mat->normalTexturePath = img.uri;
                }
            }
        }

        // Metallic-roughness texture
        int mrIdx = gltfMat.pbrMetallicRoughness.metallicRoughnessTexture.index;
        if (mrIdx >= 0)
        {
            const auto& tex = gltfModel.textures[mrIdx];
            if (tex.source >= 0)
            {
                const auto& img = gltfModel.images[tex.source];
                if (!img.uri.empty())
                {
                    mat->metallicRoughnessTexturePath = img.uri;
                }
            }
        }

        // Emissive texture
        int eIdx = gltfMat.emissiveTexture.index;
        if (eIdx >= 0)
        {
            const auto& tex = gltfModel.textures[eIdx];
            if (tex.source >= 0)
            {
                const auto& img = gltfModel.images[tex.source];
                if (!img.uri.empty())
                {
                    mat->emissiveTexturePath = img.uri;
                }
            }
        }

        materials.push_back(std::move(mat));
    }

    // If no materials, add a default white material
    if (materials.empty())
    {
        auto defaultMat = std::make_unique<VkrMaterial>();
        defaultMat->name = "Default";
        materials.push_back(std::move(defaultMat));
    }

    // ---- Step 3: Load Meshes ----
    vertices.clear();
    indices.clear();
    subMeshes.clear();
    totalTriangles = 0;
    totalVertices = 0;
    totalIndices = 0;

    for (const auto& gltfMesh : gltfModel.meshes)
    {
        for (const auto& primitive : gltfMesh.primitives)
        {
            if (primitive.attributes.find("POSITION") == primitive.attributes.end())
                continue;

            // -- Vertex positions --
            const auto& posAcc = gltfModel.accessors[primitive.attributes.at("POSITION")];
            const auto& posBv = gltfModel.bufferViews[posAcc.bufferView];
            const auto& posBuf = gltfModel.buffers[posBv.buffer];

            size_t vertexCount = posAcc.count;
            uint32_t baseVertex = static_cast<uint32_t>(vertices.size());

            // -- Normals --
            bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
            const tinygltf::Accessor* normAcc = nullptr;
            const tinygltf::BufferView* normBv = nullptr;
            const tinygltf::Buffer* normBuf = nullptr;
            if (hasNormals)
            {
                normAcc = &gltfModel.accessors[primitive.attributes.at("NORMAL")];
                normBv = &gltfModel.bufferViews[normAcc->bufferView];
                normBuf = &gltfModel.buffers[normBv->buffer];
            }

            // -- Texcoords --
            bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            const tinygltf::Accessor* uvAcc = nullptr;
            const tinygltf::BufferView* uvBv = nullptr;
            const tinygltf::Buffer* uvBuf = nullptr;
            if (hasTexCoords)
            {
                uvAcc = &gltfModel.accessors[primitive.attributes.at("TEXCOORD_0")];
                uvBv = &gltfModel.bufferViews[uvAcc->bufferView];
                uvBuf = &gltfModel.buffers[uvBv->buffer];
            }

            for (size_t i = 0; i < vertexCount; ++i)
            {
                VkrVertex v{};
                const float* p = reinterpret_cast<const float*>(
                    posBuf.data.data() + posBv.byteOffset + posAcc.byteOffset + i * 12);
                v.pos = glm::vec3(p[0], p[1], p[2]);

                if (hasNormals)
                {
                    const float* n = reinterpret_cast<const float*>(
                        normBuf->data.data() + normBv->byteOffset + normAcc->byteOffset + i * 12);
                    v.normal = glm::vec3(n[0], n[1], n[2]);
                }
                else
                {
                    v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                }

                if (hasTexCoords)
                {
                    const float* uv = reinterpret_cast<const float*>(
                        uvBuf->data.data() + uvBv->byteOffset + uvAcc->byteOffset + i * 8);
                    v.texCoord = glm::vec2(uv[0], uv[1]);
                }
                else
                {
                    v.texCoord = glm::vec2(0.0f, 0.0f);
                }

                v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // default
                vertices.push_back(v);
            }

            // -- Indices --
            if (primitive.indices < 0) continue;

            const auto& idxAcc = gltfModel.accessors[primitive.indices];
            const auto& idxBv = gltfModel.bufferViews[idxAcc.bufferView];
            const auto& idxBuf = gltfModel.buffers[idxBv.buffer];

            uint32_t firstIndex = static_cast<uint32_t>(indices.size());
            const uint8_t* idxData = idxBuf.data.data() + idxBv.byteOffset + idxAcc.byteOffset;

            for (size_t i = 0; i < idxAcc.count; ++i)
            {
                uint32_t idx = 0;
                switch (idxAcc.componentType)
                {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  idx = reinterpret_cast<const uint8_t*>(idxData)[i]; break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: idx = reinterpret_cast<const uint16_t*>(idxData)[i]; break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   idx = reinterpret_cast<const uint32_t*>(idxData)[i]; break;
                default: continue;
                }
                indices.push_back(baseVertex + idx);
            }

            uint32_t indexCount = static_cast<uint32_t>(idxAcc.count);

            // -- Sub-mesh record --
            VkrSubMesh sub;
            sub.firstIndex = firstIndex;
            sub.indexCount = indexCount;
            sub.vertexOffset = static_cast<int32_t>(baseVertex);
            sub.materialIndex = primitive.material; // -1 if no material
            subMeshes.push_back(sub);

            totalTriangles += indexCount / 3;
        }
    }

    totalVertices = static_cast<uint32_t>(vertices.size());
    totalIndices = static_cast<uint32_t>(indices.size());

    // ---- Step 4: Compute bounding box ----
    aabbMin = glm::vec3(std::numeric_limits<float>::max());
    aabbMax = glm::vec3(std::numeric_limits<float>::lowest());
    for (const auto& v : vertices)
    {
        aabbMin = glm::min(aabbMin, v.pos);
        aabbMax = glm::max(aabbMax, v.pos);
    }

    // ---- Step 5: Compute tangents ----
    computeTangents(vertices, indices);

    std::cout << "[VkrModel] Loaded '" << name << "': "
        << subMeshes.size() << " sub-meshes, "
        << totalVertices << " vertices, "
        << totalIndices << " indices, "
        << totalTriangles << " triangles, "
        << materials.size() << " materials"
        << " AABB:(" << aabbMin.x << "," << aabbMin.y << "," << aabbMin.z << ")-("
        << aabbMax.x << "," << aabbMax.y << "," << aabbMax.z << ")"
        << std::endl;

    return true;
}

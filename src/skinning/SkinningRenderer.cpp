#include "SkinningRenderer.h"

#include <Base/Mesh.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <imgui.h>

void SkinningRenderer::initialize(Platform* _platform)
{
    VulkanBase::initialize(_platform);
}

bool SkinningRenderer::initVulkan()
{
    camera = Camera(glm::vec3(0.0f, 2.0f, 6.0f));
    return VulkanBase::initVulkan("VulkanRenderer - 13_skinning");
}

bool SkinningRenderer::prepareResource()
{
    if (!createSkinnedMesh()) return false;
    generatePlaneMesh();
    createVertexBuffer(planeMesh);
    createIndexBuffer(planeMesh);

    createUniformBuffers(sceneUboResources, sizeof(SceneUBO));
    createStorageBuffers(jointBufferResources, sizeof(glm::mat4) * MAX_JOINTS);

    generateJointHierarchy();

    if (!createSkinnedDescriptorSetLayout()) return false;
    if (!createSkinnedDescriptorPool()) return false;
    createSkinnedDescriptorSets();
    if (!createSkinnedPipeline()) return false;

    if (!initUI()) return false;

    return true;
}

void SkinningRenderer::cleanup()
{
    shutdownUI();
    device.waitIdle();
}

bool SkinningRenderer::createSkinnedMesh()
{
    generateSkinnedCharacter();
    createSkinnedVertexBuffer(characterMesh);
    createIndexBuffer(characterMesh);
    return true;
}

void SkinningRenderer::generatePlaneMesh()
{
    planeMesh.vertices = {
        Vertex{ .pos = glm::vec3(-5.0f, 0.0f, -5.0f), .normal = glm::vec3(0.0f, 1.0f, 0.0f), .texCoord = glm::vec2(0.0f, 0.0f) },
        Vertex{ .pos = glm::vec3( 5.0f, 0.0f, -5.0f), .normal = glm::vec3(0.0f, 1.0f, 0.0f), .texCoord = glm::vec2(1.0f, 0.0f) },
        Vertex{ .pos = glm::vec3( 5.0f, 0.0f,  5.0f), .normal = glm::vec3(0.0f, 1.0f, 0.0f), .texCoord = glm::vec2(1.0f, 1.0f) },
        Vertex{ .pos = glm::vec3(-5.0f, 0.0f,  5.0f), .normal = glm::vec3(0.0f, 1.0f, 0.0f), .texCoord = glm::vec2(0.0f, 1.0f) }
    };
    planeMesh.indices = { 0, 2, 1, 0, 3, 2 };
}

void SkinningRenderer::generateSkinnedCharacter()
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<std::array<uint32_t, 4>> jointIndices;
    std::vector<std::array<float, 4>> jointWeights;
    std::vector<uint32_t> indices;

    auto addBox = [&](float cx, float cy, float cz,
                      float w, float h, float d,
                      uint32_t j0, uint32_t j1, float blend, int subdiv = 4)
    {
        uint32_t baseVert = static_cast<uint32_t>(positions.size());
        float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;

        std::vector<glm::vec3> verts = {
            glm::vec3(cx - hw, cy - hh, cz - hd),
            glm::vec3(cx + hw, cy - hh, cz - hd),
            glm::vec3(cx + hw, cy + hh, cz - hd),
            glm::vec3(cx - hw, cy + hh, cz - hd),
            glm::vec3(cx - hw, cy - hh, cz + hd),
            glm::vec3(cx + hw, cy - hh, cz + hd),
            glm::vec3(cx + hw, cy + hh, cz + hd),
            glm::vec3(cx - hw, cy + hh, cz + hd),
        };

        std::vector<std::array<int, 3>> quads = {
            {0,1,2}, {0,2,3}, {4,6,5}, {4,7,6},
            {0,4,5}, {0,5,1}, {2,6,7}, {2,7,3},
            {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
        };

        for (auto& q : quads)
        {
            glm::vec3 n = glm::normalize(glm::cross(
                verts[q[1]] - verts[q[0]], verts[q[2]] - verts[q[0]]));
            for (int k = 0; k < 3; ++k)
            {
                positions.push_back(verts[q[k]]);
                normals.push_back(n);
                texCoords.push_back(glm::vec2(0.0f, 0.0f));
                std::array<uint32_t, 4> jIdx = { j0, j1, 0, 0 };
                std::array<float, 4> jW = { 1.0f - blend, blend, 0.0f, 0.0f };
                jointIndices.push_back(jIdx);
                jointWeights.push_back(jW);
            }
            indices.push_back(baseVert + 0);
            indices.push_back(baseVert + 1);
            indices.push_back(baseVert + 2);
            (void)subdiv;
        }
    };

    addBox( 0.0f,  0.3f,  0.0f, 0.35f, 0.6f,  0.2f,   0,  1, 0.0f);
    addBox( 0.0f,  1.0f,  0.0f, 0.4f,  0.4f,  0.2f,   1,  2, 0.0f);
    addBox( 0.0f,  1.65f, 0.0f, 0.5f,  0.3f,  0.25f,  2,  3, 0.0f);
    addBox( 0.55f, 1.1f,  0.0f, 0.15f, 0.5f,  0.15f,  4,  5, 0.0f);
    addBox(-0.55f, 1.1f,  0.0f, 0.15f, 0.5f,  0.15f,  6,  7, 0.0f);
    addBox( 0.5f, -0.3f,  0.0f, 0.15f, 0.55f, 0.15f,  8,  9, 0.0f);
    addBox(-0.5f, -0.3f,  0.0f, 0.15f, 0.55f, 0.15f, 10, 11, 0.0f);
    addBox( 0.2f, -0.9f,  0.0f, 0.12f, 0.45f, 0.15f, 12, 13, 0.0f);
    addBox(-0.2f, -0.9f,  0.0f, 0.12f, 0.45f, 0.15f, 14, 15, 0.0f);

    characterMesh.vertices.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i)
    {
        characterMesh.vertices[i].pos = positions[i];
        characterMesh.vertices[i].normal = normals[i];
        characterMesh.vertices[i].texCoord = texCoords[i];
    }
    characterMesh.indices.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
        characterMesh.indices[i] = static_cast<uint16_t>(indices[i]);
    characterMesh.jointIndices = jointIndices;
    characterMesh.jointWeights = jointWeights;
}

void SkinningRenderer::generateJointHierarchy()
{
    jointParents.assign(MAX_JOINTS, { -1 });

    jointParents[0]  = { -1 };
    jointParents[1]  = {  0 };
    jointParents[2]  = {  1 };
    jointParents[3]  = {  2 };
    jointParents[4]  = {  3 };
    jointParents[5]  = {  4 };
    jointParents[6]  = {  3 };
    jointParents[7]  = {  6 };
    jointParents[8]  = {  0 };
    jointParents[9]  = {  8 };
    jointParents[10] = {  0 };
    jointParents[11] = { 10 };
    jointParents[12] = {  9 };
    jointParents[13] = { 12 };
    jointParents[14] = { 11 };
    jointParents[15] = { 14 };

    currentJointMatrices.assign(MAX_JOINTS, glm::mat4(1.0f));
}

void SkinningRenderer::updateJointMatrices(float time)
{
    if (!animateEnabled)
    {
        for (auto& m : currentJointMatrices)
            m = glm::mat4(1.0f);
        return;
    }

    auto& J = currentJointMatrices;
    float t = time;

    J[0] = glm::mat4(1.0f);

    J[1] = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.6f, 0.0f))
         * glm::toMat4(glm::quat(glm::vec3(0.0f, 0.0f, 0.15f * std::sin(t * 2.0f))))
         * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.6f, 0.0f));

    J[2] = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.4f, 0.0f))
         * glm::toMat4(glm::quat(glm::vec3(0.3f * std::sin(t * 1.5f), 0.0f, 0.0f)))
         * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.4f, 0.0f));

    J[3] = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.65f, 0.0f))
         * glm::toMat4(glm::quat(glm::vec3(0.0f, 0.4f * std::sin(t * 2.5f), 0.0f)))
         * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.65f, 0.0f));

    J[4] = glm::translate(glm::mat4(1.0f), glm::vec3(0.15f, 0.2f, 0.0f))
         * glm::toMat4(glm::quat(glm::vec3(0.6f + 0.4f * std::sin(t * 2.0f), 0.0f, 0.0f)))
         * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.2f, 0.0f));

    J[6] = glm::translate(glm::mat4(1.0f), glm::vec3(-0.15f, 0.2f, 0.0f))
         * glm::toMat4(glm::quat(glm::vec3(0.6f + 0.4f * std::sin(t * 2.0f + glm::pi<float>()), 0.0f, 0.0f)))
         * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.2f, 0.0f));

    J[8] = glm::translate(glm::mat4(1.0f), glm::vec3(0.15f, -0.1f, 0.0f))
         * glm::toMat4(glm::quat(glm::vec3(-0.5f + 0.3f * std::sin(t * 1.8f), 0.0f, 0.0f)))
         * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.1f, 0.0f));

    J[10] = glm::translate(glm::mat4(1.0f), glm::vec3(-0.15f, -0.1f, 0.0f))
          * glm::toMat4(glm::quat(glm::vec3(-0.5f + 0.3f * std::sin(t * 1.8f + glm::pi<float>()), 0.0f, 0.0f)))
          * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.1f, 0.0f));

    J[12] = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.3f, 0.0f))
          * glm::toMat4(glm::quat(glm::vec3(0.4f * std::sin(t * 3.0f), 0.0f, 0.0f)))
          * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f));

    J[14] = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.3f, 0.0f))
          * glm::toMat4(glm::quat(glm::vec3(0.4f * std::sin(t * 3.0f + glm::pi<float>()), 0.0f, 0.0f)))
          * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, 0.0f));

    for (uint32_t i = BONE_COUNT; i < MAX_JOINTS; ++i)
        J[i] = glm::mat4(1.0f);

    for (uint32_t i = 1; i < BONE_COUNT; ++i)
    {
        int32_t p = jointParents[i][0];
        if (p >= 0 && static_cast<uint32_t>(p) < MAX_JOINTS && p != static_cast<int32_t>(i))
            J[i] = J[static_cast<uint32_t>(p)] * J[i];
    }
}

void SkinningRenderer::updateSceneUBO(uint32_t frameIndex)
{
    SceneUBO ubo{};
    ubo.projection = glm::perspective(glm::radians(45.0f),
        static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
        0.1f, 100.0f);
    ubo.view = camera.GetViewMatrix();
    ubo.camPos = camera.Position;
    ubo.nearZ = 0.1f;
    ubo.projection[1][1] *= -1;
    std::memcpy(sceneUboResources.BuffersMapped[frameIndex], &ubo, sizeof(ubo));
}

void SkinningRenderer::updateBuffers(uint32_t frameIndex)
{
    updateSceneUBO(frameIndex);

    static auto startTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count();
    float animTime = std::fmod(time, ANIM_TIME);

    updateJointMatrices(animTime);
    std::memcpy(jointBufferResources.BuffersMapped[frameIndex],
        currentJointMatrices.data(), sizeof(glm::mat4) * MAX_JOINTS);
}

bool SkinningRenderer::createSkinnedDescriptorSetLayout()
{
    try
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1,
              .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
            { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
              .stageFlags = vk::ShaderStageFlagBits::eVertex },
        };
        skinnedDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings = bindings.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create skinned descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

bool SkinningRenderer::createSkinnedDescriptorPool()
{
    try
    {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
            { .type = vk::DescriptorType::eStorageBuffer,  .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        };
        skinnedDescriptorPool = vk::raii::DescriptorPool(device,
            vk::DescriptorPoolCreateInfo{
                .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                .maxSets = MAX_FRAMES_IN_FLIGHT,
                .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                .pPoolSizes = poolSizes.data()
            });
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create skinned descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

void SkinningRenderer::createSkinnedDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *skinnedDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *skinnedDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };
    skinnedDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo sceneBufferInfo{
            .buffer = *sceneUboResources.Buffers[i], .offset = 0, .range = sizeof(SceneUBO) };
        vk::DescriptorBufferInfo jointBufferInfo{
            .buffer = *jointBufferResources.Buffers[i], .offset = 0,
            .range = sizeof(glm::mat4) * MAX_JOINTS };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = *skinnedDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &sceneBufferInfo },
            { .dstSet = *skinnedDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &jointBufferInfo },
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

bool SkinningRenderer::createSkinnedPipeline()
{
    try
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(
            readFile(std::string(VK_SHADERS_DIR) + "skinning.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName = "vertMain"
        };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName = "fragMain"
        };
        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::VertexInputBindingDescription skinnedBindingDesc{ .binding = 0, .stride = 64, .inputRate = vk::VertexInputRate::eVertex };
        std::array<vk::VertexInputAttributeDescription, 4> skinningAttribDescs = {{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, 16),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32B32A32Uint, 32),
            vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32A32Sfloat, 48)
        }};
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &skinnedBindingDesc,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(skinningAttribDescs.size()),
            .pVertexAttributeDescriptions = skinningAttribDescs.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False
        };
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1, .scissorCount = 1 };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };
        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        skinnedPipelineLayout = vk::raii::PipelineLayout(device,
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount = 1,
                .pSetLayouts = &*skinnedDescriptorSetLayout,
                .pushConstantRangeCount = 0
            });

        vk::Format depthFormat = findDepthFormat();
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
            {
                .stageCount = 2,
                .pStages = shaderStages,
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = skinnedPipelineLayout,
                .renderPass = nullptr
            },
            {
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = &swapChainImageFormat,
                .depthAttachmentFormat = depthFormat
            }
        };

        skinnedPipeline = vk::raii::Pipeline(device, nullptr,
            chain.get<vk::GraphicsPipelineCreateInfo>());
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create skinned pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool SkinningRenderer::initUI()
{
    if (!uiEnabled) return true;
    if (ImGui::GetCurrentContext() != nullptr) return true;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;

    unsigned char* pixels = nullptr;
    int fontW = 0, fontH = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &fontW, &fontH);

    vk::DeviceSize uploadSize = static_cast<vk::DeviceSize>(fontW) * fontH * 4u;
    vk::raii::Buffer stagingBuf({});
    vk::raii::DeviceMemory stagingMem({});
    createBuffer(uploadSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuf, stagingMem);
    void* mapped = stagingMem.mapMemory(0, uploadSize);
    memcpy(mapped, pixels, static_cast<size_t>(uploadSize));
    stagingMem.unmapMemory();

    uiFontTexture.mipLevels = 1;
    createImage(static_cast<uint32_t>(fontW), static_cast<uint32_t>(fontH), 1,
        vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, uiFontTexture);

    transitionImageLayout(uiFontTexture.textureImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    copyBufferToImage(stagingBuf, uiFontTexture.textureImage,
        static_cast<uint32_t>(fontW), static_cast<uint32_t>(fontH));
    transitionImageLayout(uiFontTexture.textureImage,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    uiFontTexture.textureImageView = createImageView(
        uiFontTexture.textureImage, vk::Format::eR8G8B8A8Unorm,
        vk::ImageAspectFlagBits::eColor, 1);

    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    };
    uiFontTexture.textureSampler = vk::raii::Sampler(device, samplerInfo);

    std::vector<vk::DescriptorSetLayoutBinding> uiBindings = {
        {.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment}
    };
    uiDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
        vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(uiBindings.size()),
            .pBindings = uiBindings.data()
        });

    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1}
    };
    uiDescriptorPool = vk::raii::DescriptorPool(device,
        vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        });

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *uiDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*uiDescriptorSetLayout
    };
    uiDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    vk::DescriptorImageInfo fontInfo{
        .sampler = uiFontTexture.textureSampler,
        .imageView = uiFontTexture.textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };
    vk::WriteDescriptorSet write{
        .dstSet = *uiDescriptorSets[0], .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &fontInfo
    };
    device.updateDescriptorSets({ write }, nullptr);

    vk::PushConstantRange pushConstRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = 16u };
    uiPipelineLayout = vk::raii::PipelineLayout(device,
        vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*uiDescriptorSetLayout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstRange
        });

    vk::raii::ShaderModule shaderModule = createShaderModule(
        readFile(std::string(VK_SHADERS_DIR) + "imgui.spv"));
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule, .pName = "vertMain"
    };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule, .pName = "fragMain"
    };
    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    vk::VertexInputBindingDescription bindingDesc{
        .binding = 0, .stride = sizeof(ImDrawVert), .inputRate = vk::VertexInputRate::eVertex };
    std::array<vk::VertexInputAttributeDescription, 3> attrDescs = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, pos)),
        vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, uv)),
        vk::VertexInputAttributeDescription(2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(ImDrawVert, col))
    };
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDesc,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size()),
        .pVertexAttributeDescriptions = attrDescs.data()
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False
    };
    vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1, .scissorCount = 1 };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::False,
        .depthWriteEnable = vk::False,
        .depthCompareOp = vk::CompareOp::eAlways,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1, .pAttachments = &colorBlendAttachment
    };

    std::vector<vk::DynamicState> dynStates = {
        vk::DynamicState::eViewport, vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
        .pDynamicStates = dynStates.data()
    };

    vk::Format depthFormat = findDepthFormat();
    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
        {
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = uiPipelineLayout,
            .renderPass = nullptr
        },
        {
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swapChainImageFormat,
            .depthAttachmentFormat = depthFormat
        }
    };

    uiPipeline = vk::raii::Pipeline(device, nullptr,
        chain.get<vk::GraphicsPipelineCreateInfo>());

    uiFrameBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    return true;
}

void SkinningRenderer::updateSkinningUI()
{
    ImGui::SetNextWindowSize(ImVec2(380.0f, 240.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("GPU Skinning", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Checkbox("Animate", &animateEnabled);
    ImGui::Separator();
    ImGui::Text("Joint Count: %u / %u", BONE_COUNT, MAX_JOINTS);
    ImGui::Text("Vertices: %zu", characterMesh.vertices.size());
    ImGui::Text("Triangles: %zu", characterMesh.indices.size() / 3u);
    ImGui::Separator();
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame: %.2f ms", frameMs);
    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::BulletText("WASD + QE: Camera");
    ImGui::BulletText("Right-click drag: Look");
    ImGui::End();
}

void SkinningRenderer::updateUIFrame()
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr) return;

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(swapChainExtent.width),
                            static_cast<float>(swapChainExtent.height));
    io.DeltaTime = platform->frameTimer > 0.0f ? platform->frameTimer : (1.0f / 60.0f);

    double mouseX = 0.0, mouseY = 0.0;
    glfwGetCursorPos(platform->window, &mouseX, &mouseY);
    io.MousePos = ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
    io.MouseDown[0] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    io.MouseDown[1] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    io.MouseDown[2] = glfwGetMouseButton(platform->window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    ImGui::NewFrame();
    updateSkinningUI();
    ImGui::Render();
}

void SkinningRenderer::recordUI(vk::raii::CommandBuffer& commandBuffer)
{
    if (!uiEnabled || ImGui::GetCurrentContext() == nullptr ||
        uiPipeline == vk::raii::Pipeline(nullptr))
        return;

    ImDrawData* drawData = ImGui::GetDrawData();
    if (drawData == nullptr || drawData->TotalVtxCount <= 0) return;

    auto& fb = uiFrameBuffers[currentFrame];
    size_t vertexBytes = static_cast<size_t>(drawData->TotalVtxCount) * sizeof(ImDrawVert);
    size_t indexBytes = static_cast<size_t>(drawData->TotalIdxCount) * sizeof(ImDrawIdx);

    if (fb.vertexBuffer == vk::raii::Buffer(nullptr) || fb.vertexSize < vertexBytes)
    {
        if (fb.vertexMapped) { fb.vertexBufferMemory.unmapMemory(); fb.vertexMapped = nullptr; }
        fb.vertexBuffer = vk::raii::Buffer(nullptr);
        fb.vertexBufferMemory = vk::raii::DeviceMemory(nullptr);
        createBuffer(vertexBytes, vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            fb.vertexBuffer, fb.vertexBufferMemory);
        fb.vertexMapped = fb.vertexBufferMemory.mapMemory(0, vertexBytes);
        fb.vertexSize = vertexBytes;
    }

    if (fb.indexBuffer == vk::raii::Buffer(nullptr) || fb.indexSize < indexBytes)
    {
        if (fb.indexMapped) { fb.indexBufferMemory.unmapMemory(); fb.indexMapped = nullptr; }
        fb.indexBuffer = vk::raii::Buffer(nullptr);
        fb.indexBufferMemory = vk::raii::DeviceMemory(nullptr);
        createBuffer(indexBytes, vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            fb.indexBuffer, fb.indexBufferMemory);
        fb.indexMapped = fb.indexBufferMemory.mapMemory(0, indexBytes);
        fb.indexSize = indexBytes;
    }

    ImDrawVert* vtxDst = reinterpret_cast<ImDrawVert*>(fb.vertexMapped);
    ImDrawIdx* idxDst = reinterpret_cast<ImDrawIdx*>(fb.indexMapped);
    for (int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        memcpy(vtxDst, cmdList->VtxBuffer.Data,
               static_cast<size_t>(cmdList->VtxBuffer.Size) * sizeof(ImDrawVert));
        memcpy(idxDst, cmdList->IdxBuffer.Data,
               static_cast<size_t>(cmdList->IdxBuffer.Size) * sizeof(ImDrawIdx));
        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *uiPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *uiPipelineLayout, 0, *uiDescriptorSets[0], nullptr);
    commandBuffer.bindVertexBuffers(0, *fb.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*fb.indexBuffer, 0,
        sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

    struct UiPushConsts { glm::vec2 scale; glm::vec2 translate; };
    UiPushConsts pc;
    pc.scale = glm::vec2(2.0f / float(drawData->DisplaySize.x),
                         2.0f / float(drawData->DisplaySize.y));
    pc.translate = glm::vec2(-1.0f, -1.0f);
    commandBuffer.pushConstants(*uiPipelineLayout, vk::ShaderStageFlagBits::eVertex,
        0, vk::ArrayProxy<const UiPushConsts>(1, &pc));

    int32_t globalVertexOffset = 0;
    uint32_t globalIndexOffset = 0;
    ImVec2 clipOff = drawData->DisplayPos;

    for (int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        uint32_t indexOffset = 0;
        for (int cmdI = 0; cmdI < cmdList->CmdBuffer.Size; cmdI++)
        {
            const ImDrawCmd* cmd = &cmdList->CmdBuffer[cmdI];
            ImVec4 clipRect;
            clipRect.x = (cmd->ClipRect.x - clipOff.x) * pc.scale.x;
            clipRect.y = (cmd->ClipRect.y - clipOff.y) * pc.scale.y;
            clipRect.z = (cmd->ClipRect.z - clipOff.x) * pc.scale.x;
            clipRect.w = (cmd->ClipRect.w - clipOff.y) * pc.scale.y;

            if (clipRect.x < float(swapChainExtent.width) &&
                clipRect.y < float(swapChainExtent.height) &&
                clipRect.z >= 0.0f && clipRect.w >= 0.0f)
            {
                vk::Rect2D scissor;
                scissor.offset.x = static_cast<int32_t>(std::max(0.0f, clipRect.x));
                scissor.offset.y = static_cast<int32_t>(std::max(0.0f, clipRect.y));
                scissor.extent.width = static_cast<uint32_t>(std::max(0.0f, clipRect.z - clipRect.x));
                scissor.extent.height = static_cast<uint32_t>(std::max(0.0f, clipRect.w - clipRect.y));
                commandBuffer.setScissor(0, scissor);
                commandBuffer.drawIndexed(cmd->ElemCount, 1,
                    globalIndexOffset + indexOffset, globalVertexOffset, 0);
            }
            indexOffset += cmd->ElemCount;
        }
        globalIndexOffset += static_cast<uint32_t>(cmdList->IdxBuffer.Size);
        globalVertexOffset += cmdList->VtxBuffer.Size;
    }
}

void SkinningRenderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[currentFrame];
    commandBuffer.begin({});

    transition_image_layout(swapChainImages[imageIndex],
        swapChainImageLayouts[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        {}, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    transition_image_layout(depthData.textureImage,
        depthImageLayout,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);
    depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    vk::RenderingAttachmentInfo swapchainAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(std::array<float, 4>{0.02f, 0.02f, 0.03f, 1.0f})
    };

    vk::RenderingAttachmentInfo depthAttachment{
        .imageView = depthData.textureImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
    };

    vk::RenderingInfo renderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &swapchainAttachment,
        .pDepthAttachment = &depthAttachment
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.setViewport(0,
        vk::Viewport(0.0f, 0.0f,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *skinnedPipeline);
    commandBuffer.bindVertexBuffers(0, *characterMesh.vertexBuffer, { 0 });
    commandBuffer.bindIndexBuffer(*characterMesh.indexBuffer, 0,
        vk::IndexTypeValue<decltype(characterMesh.indices)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *skinnedPipelineLayout, 0, *skinnedDescriptorSets[currentFrame], nullptr);
    commandBuffer.drawIndexed(
        static_cast<uint32_t>(characterMesh.indices.size()), 1, 0, 0, 0);

    commandBuffer.endRendering();

    vk::RenderingAttachmentInfo uiAttachmentInfo{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingInfo uiRenderingInfo{
        .renderArea = { .offset = {0, 0}, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uiAttachmentInfo
    };

    commandBuffer.beginRendering(uiRenderingInfo);
    commandBuffer.setViewport(0,
        vk::Viewport(0.0f, 0.0f,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    recordUI(commandBuffer);
    commandBuffer.endRendering();

    transition_image_layout(swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite, {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
    swapChainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;

    commandBuffer.end();
}

void SkinningRenderer::render()
{
    auto frameStart = std::chrono::high_resolution_clock::now();

    const auto fenceResult = device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
        throw std::runtime_error("failed to wait for fence!");

    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }

    device.resetFences(*inFlightFences[currentFrame]);
    commandBuffers[currentFrame].reset();

    updateUIFrame();
    updateBuffers(currentFrame);
    recordCommandBuffer(imageIndex);

    const vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[currentFrame],
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
    };
    graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };
    result = presentQueue.presentKHR(presentInfo);

    if (result == vk::Result::eSuboptimalKHR ||
        result == vk::Result::eErrorOutOfDateKHR ||
        framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }

    auto frameEnd = std::chrono::high_resolution_clock::now();
    frameMs = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
    fps = (frameMs > 0.0f) ? (1000.0f / frameMs) : 0.0f;

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

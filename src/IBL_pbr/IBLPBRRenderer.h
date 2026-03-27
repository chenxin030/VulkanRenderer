#pragma once

#include <Base/VulkanBase.h>
#include <glm/glm.hpp>
#include <vector>

struct IBLPBRRenderer final : VulkanBase
{
public:
    void initialize(Platform* _platform);

    bool initVulkan();
    bool prepareResource();
    void render();
    void waitIdle() { device.waitIdle(); }

private:
	vk::raii::DescriptorSetLayout iblPbrDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      iblPbrPipelineLayout = nullptr;
	vk::raii::Pipeline            iblPbrPipeline = nullptr;
	vk::raii::DescriptorPool      iblPbrDescriptorPool = nullptr;

	MeshBuffer pbrInstanceBufferResources;
	MeshBuffer sceneUboResources;
	MeshBuffer lightUboResources;
	MeshBuffer paramsUboResources;
	MeshBuffer skyboxUboResources;

	vk::raii::DescriptorSetLayout skyboxDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      skyboxPipelineLayout = nullptr;
	vk::raii::Pipeline            skyboxPipeline = nullptr;
	vk::raii::DescriptorPool      skyboxDescriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> skyboxDescriptorSets;

	Mesh skyboxTriangleMesh;
	Mesh sphereMesh;

	TextureData hdrEquirectData;
	TextureData envCubemapData;
	TextureData irradianceCubemapData;
	TextureData prefilteredEnvMapData;
	TextureData brdfLutData;

    uint32_t instanceCount = 49;

    bool createPBRDescriptorSetLayout();
    bool createPBRDescriptorPool();
    void createPBRDescriptorSets();
    bool createPBRPipeline();
    void createPBRBuffers();

	bool createSkyboxDescriptorSetLayout();
	bool createSkyboxDescriptorPool();
	void createSkyboxDescriptorSets();
	bool createSkyboxPipeline();

	void generateIBLResources();

	void updatePBRInstanceBuffers(uint32_t frameIndex);
    void recordCommandBuffer(uint32_t imageIndex);

	void transitionImageLayoutCmd(
		vk::raii::CommandBuffer& commandBuffer,
		vk::Image image,
		vk::ImageAspectFlags aspectMask,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		uint32_t baseMipLevel,
		uint32_t levelCount,
		uint32_t baseArrayLayer,
		uint32_t layerCount,
		vk::PipelineStageFlags srcStage,
		vk::PipelineStageFlags dstStage,
		vk::AccessFlags srcAccessMask,
		vk::AccessFlags dstAccessMask);
};


#include "Renderer.h"

bool Renderer::createDescriptorSetLayout(){
  try {
    //// Create binding for a uniform buffer
    //vk::DescriptorSetLayoutBinding uboLayoutBinding{
    //  .binding = 0,
    //  .descriptorType = vk::DescriptorType::eUniformBuffer,
    //  .descriptorCount = 1,
    //  .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    //  .pImmutableSamplers = nullptr
    //};

    //// Create binding for texture sampler
    //vk::DescriptorSetLayoutBinding samplerLayoutBinding{
    //  .binding = 1,
    //  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
    //  .descriptorCount = 1,
    //  .stageFlags = vk::ShaderStageFlagBits::eFragment,
    //  .pImmutableSamplers = nullptr
    //};

    //// Create a descriptor set layout
    //std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

    //vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    //layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    //layoutInfo.pBindings = bindings.data();
    //if (descriptorIndexingEnabled) {
    //  layoutInfo.flags |= vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
    //  layoutInfo.pNext = &bindingFlagsInfo;
    //}

    //descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::DescriptorSetLayoutBinding    uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
    vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = 1, .pBindings = &uboLayoutBinding };
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    return true;
  }
 catch (const std::exception& e) {
std::cerr << "Failed to create descriptor set layout: " << e.what() << std::endl;
return false;
}
}
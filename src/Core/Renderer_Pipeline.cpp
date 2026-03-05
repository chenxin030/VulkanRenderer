#include "Renderer.h"

bool Renderer::createDescriptorSetLayout(){
  try {
    std::array bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = bindings.size(), .pBindings = bindings.data()};
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    return true;
  }
 catch (const std::exception& e) {
    std::cerr << "Failed to create descriptor set layout: " << e.what() << std::endl;
    return false;
 }
}
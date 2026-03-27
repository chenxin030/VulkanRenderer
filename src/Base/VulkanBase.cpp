#include "VulkanBase.h"

VulkanBase::VulkanBase()
{
    deviceExtensions = requiredDeviceExtensions;
}

void VulkanBase::initialize(Platform* _platform, ResourceManager* _resourceManager, Scene* _scene)
{
    platform = _platform;
    resourceManager = _resourceManager;
    scene = _scene;
    maxInstances = 0;

    platform->resizeCallback = [this](int, int) {
        framebufferResized = true;
    };
}


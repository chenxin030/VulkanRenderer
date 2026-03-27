#include "VulkanBase.h"

VulkanBase::VulkanBase()
{
    deviceExtensions = requiredDeviceExtensions;
}

void VulkanBase::initialize(Platform* _platform)
{
    platform = _platform;
    platform->resizeCallback = [this](int, int) {
        framebufferResized = true;
    };

    platform->resizeCallback = [this](int, int) { framebufferResized = true; };

    platform->mouseCallback = [this](float xpos, float ypos, uint32_t) {
        if (!platform->rightMouseButtonPressed) return;
        if (platform->firstMouse) {
            platform->lastX = xpos;
            platform->lastY = ypos;
            platform->firstMouse = false;
        }

        const float xoffset = xpos - static_cast<float>(platform->lastX);
        const float yoffset = static_cast<float>(platform->lastY) - ypos;
        platform->lastX = xpos;
        platform->lastY = ypos;
        camera.ProcessMouseMovement(xoffset, yoffset);
    };

    platform->scrollCallback = [this](double, double yoffset) {
        camera.ProcessMouseScroll(static_cast<float>(yoffset));
    };

}


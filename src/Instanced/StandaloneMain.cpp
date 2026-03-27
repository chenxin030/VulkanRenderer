#include "InstancedRenderer.h"

#include <Platform.h>

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - 1_InstenceRender");
        platform.initWindow();

        InstancedRenderer renderer;
        renderer.initialize(&platform);
        if (!renderer.initVulkan())
        {
            return EXIT_FAILURE;
        }
        renderer.prepareResource();

        bool running = true;
        while (running)
        {
            if (!platform.processEvents())
            {
                running = false;
                break;
            }
            renderer.processInput(platform.frameTimer);
            renderer.render();
            platform.endFrame();
        }

        renderer.device.waitIdle();
        platform.cleanup();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


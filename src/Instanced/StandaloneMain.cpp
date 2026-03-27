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

        InstancedRenderer app;
        app.initialize(&platform, nullptr, nullptr);
        if (!app.initVulkan())
        {
            return EXIT_FAILURE;
        }
        app.prepareResource();

        bool running = true;
        while (running)
        {
            if (!platform.processEvents())
            {
                running = false;
                break;
            }
            app.render();
            platform.endFrame();
        }

        app.device.waitIdle();
        platform.cleanup();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


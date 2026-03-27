#include "TriangleRenderer.h"

#include <Core/Platform.h>

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - triangle (Standalone)");
        platform.initWindow();

        TriangleRenderer app;
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
        app.cleanup();
        platform.cleanup();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


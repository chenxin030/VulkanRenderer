#include "GIRenderer.h"

#include <Base/Platform.h>

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - 11_gi_ssao");
        platform.initWindow();

        GIRenderer renderer;
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

        renderer.waitIdle();
        renderer.cleanup();
        platform.cleanup();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#include "VkrRenderer.h"

#include <Base/Platform.h>

#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("vkrEngine - Vulkan Renderer Demo");
        platform.initWindow();

        VkrRenderer renderer;
        renderer.initialize(&platform);

        if (!renderer.initVulkan())
        {
            std::fprintf(stderr, "[vkrEngine] Vulkan initialization failed.\n");
            return EXIT_FAILURE;
        }

        if (!renderer.prepareResource())
        {
            std::fprintf(stderr, "[vkrEngine] Resource preparation failed.\n");
            return EXIT_FAILURE;
        }

        std::cout << "[vkrEngine] Entering main loop..." << std::endl;

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
        std::fprintf(stderr, "[vkrEngine] Fatal error: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

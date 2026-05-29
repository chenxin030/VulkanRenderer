#include "VolumetricRenderer.h"

#include <Base/Platform.h>

#include <cstdio>
#include <exception>
#include <memory>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - Volumetric Lighting");
        platform.initWindow();

        VolumetricRenderer renderer;
        renderer.initialize(&platform);
        if (!renderer.initVulkan())
        {
            return EXIT_FAILURE;
        }
        if (!renderer.prepareResource())
        {
            std::cerr << "FATAL: prepareResource() failed!" << std::endl;
            return EXIT_FAILURE;
        }

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

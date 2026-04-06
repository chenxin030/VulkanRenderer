#include "ParticleRenderer.h"

#include <Base/Platform.h>

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - 14_particles");
        platform.initWindow();

        ParticleRenderer renderer;
        renderer.initialize(&platform);
        if (!renderer.initVulkan())
        {
            return EXIT_FAILURE;
        }
        if (!renderer.prepareResource())
        {
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

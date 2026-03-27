#include "IBLPBRRenderer.h"

#include <Core/Platform.h>
#include <Core/ResourceManager.h>
#include "Scene.h"

#include <cstdio>
#include <exception>
#include <memory>

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - IBL pbr (Standalone)");
        platform.initWindow();

        ResourceManager resourceManager;
        Scene scene;
        const uint32_t sceneMax = Scene::getDefaultMaxInstances();
        scene.initScene(resourceManager, sceneMax);
        resourceManager.initResource(sceneMax);

        auto app = std::make_unique<IBLPBRRenderer>();
        app->initialize(&platform, &resourceManager, &scene);
        if (!app->initVulkan())
        {
            return EXIT_FAILURE;
        }
        app->prepareResource();

        bool running = true;
        while (running)
        {
            if (!platform.processEvents())
            {
                running = false;
                break;
            }
            app->processInput(platform.frameTimer);
            app->render();
            platform.endFrame();
        }

        app->waitIdle();
        app->cleanup();
        platform.cleanup();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


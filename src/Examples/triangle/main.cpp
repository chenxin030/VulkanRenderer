#include <Core/Platform.h>
#include <Core/Renderer.h>
#include <Core/ResourceManager.h>
#include <Core/Scene.h>

#include <cstdio>

#define LOGE(...)                 \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n")

int main()
{
    try
    {
        Platform platform;
        platform.SetBaseTitle("VulkanRenderer - triangle (Level1)");
        platform.initWindow();

        ResourceManager resourceManager;
        Scene scene;
        const uint32_t sceneMax = Scene::getDefaultMaxInstances();
        scene.initScene(resourceManager, sceneMax);
        resourceManager.initResource(sceneMax);

        Renderer renderer;
        renderer.initialize(&platform, &resourceManager, &scene);
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
        LOGE("%s", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


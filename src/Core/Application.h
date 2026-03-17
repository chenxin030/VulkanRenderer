#pragma once

#include <Renderer.h>

struct Application {

	Platform platform;
	Renderer renderer;
	ResourceManager resourceManager;
	Scene scene;

	bool running;

	Application() : running(true) {}
	void init() {
		platform.initWindow();
		const uint32_t sceneMax = Scene::getDefaultMaxInstances();
		scene.initScene(resourceManager, sceneMax);
		resourceManager.initResource(sceneMax);
		renderer.initialize(&platform, &resourceManager, &scene);
		renderer.initVulkan();
		renderer.loadResource();
#if RENDERING_LEVEL == 1
		renderer.createDescriptorSets();
#elif RENDERING_LEVEL == 2 
		renderer.createInstancedDescriptorSets();
#elif RENDERING_LEVEL == 3
		renderer.createPBRDescriptorSets();
#elif RENDERING_LEVEL == 4
		renderer.createIBLPBRDescriptorSets();
		renderer.createSkyboxDescriptorSets();
#elif RENDERING_LEVEL == 5 || RENDERING_LEVEL == 6 || RENDERING_LEVEL == 7
		renderer.createShadowDescriptorSets();
#endif
	}
	void run()
	{
		while (running) {
			if (!platform.processEvents()) {
				running = false;
				break;
			}
			renderer.processInput(platform.frameTimer);
			renderer.render();

			platform.endFrame();
		}
		renderer.waitIdle();

		cleanup();
	}

	void cleanup() {
		renderer.cleanup();
		platform.cleanup();
	}
};

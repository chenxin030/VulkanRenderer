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
		renderer.prepareResource();
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

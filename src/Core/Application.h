#pragma once

#include <Renderer.h>

struct Application {

	Platform platform;
	Renderer renderer;
	ResourceManager resourceManager;

	bool running;

	Application() : running(true) {}
	void init() {
		platform.initWindow();
		resourceManager.initResource(MAX_OBJECTS);
		renderer.initialize(&platform, &resourceManager);
		renderer.initVulkan();
		renderer.loadResource();
		renderer.createDescriptorSets();
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
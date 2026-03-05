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
		resourceManager.initResource(1);
		renderer.initialize(&platform, &resourceManager);
		renderer.initVulkan();
		renderer.loadResource();
	}
	void run()
	{
		while (running) {
			if (!platform.processEvents()) {
				running = false;
				break;
			}
			renderer.render();
		}
		renderer.waitIdle();
	}

	void cleanup() {
		
		platform.cleanup();
	}
};
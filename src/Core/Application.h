#pragma once

#include <Renderer.h>
#include <platform.h>
#include <ResourceManager.h>

struct Application {

	Platform platform;
	Renderer renderer;
	ResourceManager resourceManager;

	bool running;

	Application() : running(true) {}
	void init() {
		platform.initWindow();
		resourceManager.initResources();
		renderer.initialize(&platform, &resourceManager);
		renderer.initVulkan();
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
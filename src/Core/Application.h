#pragma once

#include <Renderer.h>
#include <Platform.h>
#include <ResourceManager.h>

struct Application {

	Platform platform;
	Renderer renderer;
	ResourceManager resourceManager;

	bool running;

	Application() : renderer(platform), running(true) {}
	void init() {
		platform.initWindow();
		resourceManager.initResources();
		renderer.initVulkan();
		renderer.createResouceBuffer(resourceManager.resources);
	}
	void run()
	{
		while (running) {
			if (!platform.processEvents()) {
				running = false;
				break;
			}
			renderer.render(resourceManager.resources);
		}
		renderer.waitIdle();
	}

	void cleanup() {
		platform.cleanup();
	}
};
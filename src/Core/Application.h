#pragma once

#include <Renderer.h>
#include <Platform.h>

struct Application {

	Platform platform;
	Renderer renderer;

	bool running;

	void init() {
		platform.initWindow();
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
	}

	void cleanup() {
		platform.cleanup();
	}
};
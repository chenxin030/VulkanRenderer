#pragma once

#include <GLFW/glfw3.h>
#include <functional>
#include <chrono>

class Renderer;

struct Platform
{
	GLFWwindow* window = nullptr;

	int width = 800;
	int height = 600;

	bool isFullscreen = false;
	int windowedX = 0, windowedY = 0;
	int windowedWidth = 800, windowedHeight = 600;

	bool windowResized = false;

	uint32_t frameCounter = 0;
	uint32_t lastFPS = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp;
	std::chrono::time_point<std::chrono::high_resolution_clock> frameStart;
	float frameTimer = 0.0f;

	std::string baseTitle = "Vulkan";
	std::string customTitleFormat;

	std::function<void(int, int)> resizeCallback;
	std::function<void(float, float, uint32_t)> mouseCallback;
	std::function<void(uint32_t, bool)> keyboardCallback;
	std::function<void(uint32_t)> charCallback;
	std::function<void(double, double)> scrollCallback;

	// Mouse state tracking
	double lastX = 400, lastY = 300;
	bool firstMouse = true;
	bool rightMouseButtonPressed = false;

	void initWindow()
	{
		if (!glfwInit()) {
			throw std::runtime_error("Failed to initialize GLFW");
		}

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(width, height, baseTitle.c_str(), nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			throw std::runtime_error("Failed to create GLFW window");
		}

		// Setup callbacks
		glfwSetWindowUserPointer(window, this);

		glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
			auto* plat = static_cast<Platform*>(glfwGetWindowUserPointer(window));
			if (plat->resizeCallback) {
				plat->resizeCallback(width, height);
			}
			});

		glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {
			auto* plat = static_cast<Platform*>(glfwGetWindowUserPointer(window));
			if (plat->mouseCallback) {
				plat->mouseCallback((float)xpos, (float)ypos, 0);
			}
			});

		glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
			auto* plat = static_cast<Platform*>(glfwGetWindowUserPointer(window));
			if (button == GLFW_MOUSE_BUTTON_RIGHT) {
				if (action == GLFW_PRESS) {
					plat->rightMouseButtonPressed = true;
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				}
				else if (action == GLFW_RELEASE) {
					plat->rightMouseButtonPressed = false;
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
					plat->firstMouse = true;
				}
			}
			});

		glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
			auto* plat = static_cast<Platform*>(glfwGetWindowUserPointer(window));
			if (plat->scrollCallback) {
				plat->scrollCallback(xoffset, yoffset);
			}
			});

		lastTimestamp = std::chrono::high_resolution_clock::now();
		frameStart = lastTimestamp;
	}

	void cleanup()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	bool processEvents() {
		glfwPollEvents();

		if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
			ToggleFullscreen();
		}
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}

		return !glfwWindowShouldClose(window);
	}

	void endFrame() {
		frameCounter++;

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - frameStart).count();
		frameTimer = (float)tDiff / 1000.0f; 
		frameStart = tEnd; // Update frameStart for next frame

		float fpsTimer = (float)std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count();
		if (fpsTimer > 1000.0f) {
			lastFPS = static_cast<uint32_t>((float)frameCounter * (1000.0f / fpsTimer));

			updateWindowTitle();

			frameCounter = 0;
			lastTimestamp = tEnd;
		}
	}

	void updateWindowTitle() {
		if (!window) return;

		std::string title;
		if (!customTitleFormat.empty()) {
			char buffer[256];
			snprintf(buffer, sizeof(buffer), customTitleFormat.c_str(), lastFPS, frameTimer * 1000.0f);
			title = buffer;
		}
		else {
			title = baseTitle + " - FPS: " + std::to_string(lastFPS);
		}

		glfwSetWindowTitle(window, title.c_str());
	}

	void ToggleFullscreen() {
		if (!window) return;

		if (!isFullscreen) {
			glfwGetWindowPos(window, &windowedX, &windowedY);
			glfwGetWindowSize(window, &windowedWidth, &windowedHeight);

			GLFWmonitor* monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);

			glfwSetWindowMonitor(window, monitor, 0, 0,
				mode->width, mode->height, mode->refreshRate);
			isFullscreen = true;
		}
		else {
			glfwSetWindowMonitor(window, nullptr,
				windowedX, windowedY,
				windowedWidth, windowedHeight, 0);
			isFullscreen = false;
		}
		windowResized = true;
		if (resizeCallback) {
			int w, h;
			glfwGetFramebufferSize(window, &w, &h);
			resizeCallback(w, h);
		}
	}

	void SetTitleFormat(const std::string& format) {
		customTitleFormat = format;
	}

	void SetBaseTitle(const std::string& title) {
		baseTitle = title;
		updateWindowTitle();
	}

	uint32_t GetFPS() const {
		return lastFPS;
	}

	float GetFrameTime() const {
		return frameTimer;
	}

	float GetFrameTimeMS() const {
		return frameTimer * 1000.0f;
	}

	bool CreateVulkanSurface(VkInstance instance, VkSurfaceKHR* surface) {
		if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
			return false;
		}
		return true;
	}

	int GetWindowWidth() const {
		return width;
	}
	int GetWindowHeight() const {
		return height;
	}
	void GetWindowSize(int* width, int* height) const {
		if (window) {
			glfwGetFramebufferSize(window, width, height);
		}
		else {
			*width = GetWindowWidth();
			*height = GetWindowHeight();
		}
	}

	void SetWindowTitle(const std::string& title) {
		if (window) {
			glfwSetWindowTitle(window, title.c_str());
		}
	}

};
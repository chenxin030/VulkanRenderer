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
	int windowedX = 0, windowedY = 0;  // ����ģʽλ��
	int windowedWidth = 800, windowedHeight = 600;  // ����ģʽ��С�����ڽ���ȫ��ǰ����һ�Σ�ȡ��ȫ����ص����ȫ��ǰ�Ĵ��ڴ�С��

	bool windowResized = false;

	// FPS��ʱ�����
	uint32_t frameCounter = 0;
	uint32_t lastFPS = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp;
	std::chrono::time_point<std::chrono::high_resolution_clock> frameStart;
	float frameTimer = 0.0f;

	// ���ڱ������
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

		// ��ʼ��ʱ��
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

	// ��ÿ֡��Ⱦ��ɺ���ã�����FPS��֡ʱ��
	void endFrame() {
		frameCounter++;

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - frameStart).count();
		frameTimer = (float)tDiff / 1000.0f;  // ת��Ϊ��
		frameStart = tEnd; // Update frameStart for next frame

		// ����FPS��ʾ��ÿ�룩
		float fpsTimer = (float)std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count();
		if (fpsTimer > 1000.0f) {
			lastFPS = static_cast<uint32_t>((float)frameCounter * (1000.0f / fpsTimer));

			// ���´��ڱ���
			updateWindowTitle();

			// ���ü�����
			frameCounter = 0;
			lastTimestamp = tEnd;
		}
	}

	// ���´��ڱ���
	void updateWindowTitle() {
		if (!window) return;

		std::string title;
		if (!customTitleFormat.empty()) {
			// ʹ���Զ����ʽ
			char buffer[256];
			snprintf(buffer, sizeof(buffer), customTitleFormat.c_str(), lastFPS, frameTimer * 1000.0f);
			title = buffer;
		}
		else {
			// Ĭ�ϸ�ʽ
			title = baseTitle + " - FPS: " + std::to_string(lastFPS);
		}

		glfwSetWindowTitle(window, title.c_str());
	}

	void ToggleFullscreen() {
		if (!window) return;

		if (!isFullscreen) {
			// ���洰��ģʽ��λ�úʹ�С
			glfwGetWindowPos(window, &windowedX, &windowedY);
			glfwGetWindowSize(window, &windowedWidth, &windowedHeight);

			GLFWmonitor* monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);

			// �л���ȫ��
			glfwSetWindowMonitor(window, monitor, 0, 0,
				mode->width, mode->height, mode->refreshRate);
			isFullscreen = true;
		}
		else {
			// �ָ�������ģʽ
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

	// �����Զ�������ʽ
	// ����ʹ�� %d ��ʾFPS��%f ��ʾ֡ʱ��(ms)
	// ����: "MyApp - FPS: %d | Frame Time: %.2f ms"
	void SetTitleFormat(const std::string& format) {
		customTitleFormat = format;
	}

	// ���û������⣨������FPS��
	void SetBaseTitle(const std::string& title) {
		baseTitle = title;
		updateWindowTitle();
	}

	// ��ȡ��ǰFPS
	uint32_t GetFPS() const {
		return lastFPS;
	}

	// ��ȡ��ǰ֡ʱ�䣨�룩
	float GetFrameTime() const {
		return frameTimer;
	}

	// ��ȡ��ǰ֡ʱ�䣨���룩
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
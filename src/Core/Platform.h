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
	int windowedX = 0, windowedY = 0;  // 窗口模式位置
	int windowedWidth = 800, windowedHeight = 600;  // 窗口模式大小（仅在进入全屏前更新一次，取消全屏后回到这个全屏前的窗口大小）

	bool windowResized = false;

	// FPS和时间相关
	uint32_t frameCounter = 0;
	uint32_t lastFPS = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp;
	std::chrono::time_point<std::chrono::high_resolution_clock> frameStart;
	float frameTimer = 0.0f;

	// 窗口标题相关
	std::string baseTitle = "Vulkan";
	std::string customTitleFormat;

	std::function<void(int, int)> resizeCallback;
	std::function<void(float, float, uint32_t)> mouseCallback;
	std::function<void(uint32_t, bool)> keyboardCallback;
	std::function<void(uint32_t)> charCallback;

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

		// 初始化时间
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

	// 在每帧渲染完成后调用，更新FPS和帧时间
	void endFrame() {
		frameCounter++;

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - frameStart).count();
		frameTimer = (float)tDiff / 1000.0f;  // 转换为秒

		// 更新FPS显示（每秒）
		float fpsTimer = (float)std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count();
		if (fpsTimer > 1000.0f) {
			lastFPS = static_cast<uint32_t>((float)frameCounter * (1000.0f / fpsTimer));

			// 更新窗口标题
			updateWindowTitle();

			// 重置计数器
			frameCounter = 0;
			lastTimestamp = tEnd;
		}
	}

	// 更新窗口标题
	void updateWindowTitle() {
		if (!window) return;

		std::string title;
		if (!customTitleFormat.empty()) {
			// 使用自定义格式
			char buffer[256];
			snprintf(buffer, sizeof(buffer), customTitleFormat.c_str(), lastFPS, frameTimer * 1000.0f);
			title = buffer;
		}
		else {
			// 默认格式
			title = baseTitle + " - FPS: " + std::to_string(lastFPS);
		}

		glfwSetWindowTitle(window, title.c_str());
	}

	void ToggleFullscreen() {
		if (!window) return;

		if (!isFullscreen) {
			// 保存窗口模式的位置和大小
			glfwGetWindowPos(window, &windowedX, &windowedY);
			glfwGetWindowSize(window, &windowedWidth, &windowedHeight);

			GLFWmonitor* monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);

			// 切换到全屏
			glfwSetWindowMonitor(window, monitor, 0, 0,
				mode->width, mode->height, mode->refreshRate);
			isFullscreen = true;
		}
		else {
			// 恢复到窗口模式
			glfwSetWindowMonitor(window, nullptr,
				windowedX, windowedY,
				windowedWidth, windowedHeight, 0);
			isFullscreen = false;
		}
		windowResized = true;
	}

	// 设置自定义标题格式
	// 可以使用 %d 表示FPS，%f 表示帧时间(ms)
	// 例如: "MyApp - FPS: %d | Frame Time: %.2f ms"
	void SetTitleFormat(const std::string& format) {
		customTitleFormat = format;
	}

	// 设置基础标题（不包含FPS）
	void SetBaseTitle(const std::string& title) {
		baseTitle = title;
		updateWindowTitle();
	}

	// 获取当前FPS
	uint32_t GetFPS() const {
		return lastFPS;
	}

	// 获取当前帧时间（秒）
	float GetFrameTime() const {
		return frameTimer;
	}

	// 获取当前帧时间（毫秒）
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
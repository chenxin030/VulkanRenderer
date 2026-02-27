#pragma once

#include <GLFW/glfw3.h>
#include <functional>

struct Platform
{
	GLFWwindow* window = nullptr;

	int width = 800;
	int height = 600;

	bool windowResized = false;

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
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			throw std::runtime_error("Failed to create GLFW window");
		}

		glfwSetWindowUserPointer(window, this);
		
	glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int width, int height) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(w));
		platform->WindowResizeCallback(w, width, height);
	});

	glfwSetCursorPosCallback(window, [](GLFWwindow* w, double xpos, double ypos) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(w));
		platform->MousePositionCallback(w, xpos, ypos);
	});

	glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(w));
		platform->MouseButtonCallback(w, button, action, mods);
	});

	glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(w));
		platform->KeyCallback(w, key, scancode, action, mods);
	});

	glfwSetCharCallback(window, [](GLFWwindow* w, unsigned int codepoint) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(w));
		platform->CharCallback(w, codepoint);
	});

	glfwGetFramebufferSize(window, &width, &height);
	}

	void cleanup()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	bool processEvents() {
		glfwPollEvents();
		return !glfwWindowShouldClose(window);
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
		*width = GetWindowWidth();
		*height = GetWindowHeight();
	}

	void SetResizeCallback(std::function<void(int, int)> callback) {
		resizeCallback = std::move(callback);
	}

	void SetMouseCallback(std::function < void(float, float, uint32_t) > callback) {
		mouseCallback = std::move(callback);
	}

	void SetKeyboardCallback(std::function < void(uint32_t, bool) > callback) {
		keyboardCallback = std::move(callback);
	}

	void SetCharCallback(std::function<void(uint32_t)> callback) {
		charCallback = std::move(callback);
	}

	void SetWindowTitle(const std::string& title) {
		if (window) {
			glfwSetWindowTitle(window, title.c_str());
		}
	}

	void WindowResizeCallback(GLFWwindow* window, int width, int height) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(window));
		platform->width = width;
		platform->height = height;
		platform->windowResized = true;

		if (platform->resizeCallback) {
			platform->resizeCallback(width, height);
		}
	}

	void MousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(window));
		if (platform->mouseCallback) {
			uint32_t buttons = 0;
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
				buttons |= 0x01;
			}
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
				buttons |= 0x02;
			}
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
				buttons |= 0x04;
			}
			platform->mouseCallback(static_cast<float>(xpos), static_cast<float>(ypos), buttons);
		}
	}

	void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(window));
		if (platform->mouseCallback) {
			double xpos, ypos;
			glfwGetCursorPos(window, &xpos, &ypos);
			uint32_t buttons = 0;
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
				buttons |= 0x01;
			}
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
				buttons |= 0x02;
			}
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
				buttons |= 0x04;
			}
			platform->mouseCallback(static_cast<float>(xpos), static_cast<float>(ypos), buttons);
		}
	}

	void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(window));
		if (platform->keyboardCallback) {
			platform->keyboardCallback(key, action != GLFW_RELEASE);
		}
	}

	void CharCallback(GLFWwindow* window, unsigned int codepoint) {
		auto* platform = static_cast<Platform*>(glfwGetWindowUserPointer(window));
		if (platform->charCallback) {
			platform->charCallback(codepoint);
		}
	}
};
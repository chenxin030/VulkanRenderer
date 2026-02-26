#pragma once

#include <GLFW/glfw3.h>

struct Platform
{
	GLFWwindow* window = nullptr;

	uint32_t WIDTH = 800;
	uint32_t HEIGHT = 600;

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
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
};
#pragma once

#include "Mesh.h"

struct ResourceManager {
	std::vector<Mesh> resources;

	void initResources() {
		resources.resize(1);
		resources[0].vertices.resize(4);
		resources[0].vertices = {
			{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
			{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
		};
		resources[0].indices.resize(6);
		resources[0].indices = {0, 1, 2, 2, 3, 0 };
	}
};
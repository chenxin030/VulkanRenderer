#include "Scene.h"

void Scene::initScene(const ResourceManager& resourceManager, unsigned int modelCount)
{
	world.clear();
	maxInstances = static_cast<uint32_t>(modelCount);
	cubeMeshIndex = 0;
	sphereMeshIndex = 1;

#if RENDERING_LEVEL == 5
	setCubeMeshIndex(0);
	setSphereMeshIndex(1);
	if (modelCount > 0) {
		Entity ground = world.createEntity();
		world.addTransform(ground, Transform{ {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {6.0f, 0.05f, 6.0f} });
		world.addMeshTag(ground, MeshTag::Cube);
		world.addColor(ground, glm::vec4(0.78f, 0.78f, 0.80f, 1.0f));
	}
	if (modelCount > 1) {
		Entity pillar = world.createEntity();
		world.addTransform(pillar, Transform{ {-2.0f, 0.25f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.25f, 1.20f, 0.25f} });
		world.addMeshTag(pillar, MeshTag::Cube);
		world.addColor(pillar, glm::vec4(0.35f, 0.35f, 0.35f, 1.0f));
	}
	if (modelCount > 2) {
		Entity top = world.createEntity();
		world.addTransform(top, Transform{ {-2.0f, 1.95f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.50f, 0.50f, 0.50f} });
		world.addMeshTag(top, MeshTag::Cube);
		world.addColor(top, glm::vec4(0.90f, 0.25f, 0.20f, 1.0f));
	}
	if (modelCount > 3) {
		Entity pad = world.createEntity();
		world.addTransform(pad, Transform{ {0.0f, -0.92f, -1.5f}, {0.0f, 0.0f, 0.0f}, {0.30f, 0.03f, 0.30f} });
		world.addMeshTag(pad, MeshTag::Cube);
		world.addColor(pad, glm::vec4(0.35f, 0.35f, 0.35f, 1.0f));
	}
	if (modelCount > 4) {
		Entity cube = world.createEntity();
		world.addTransform(cube, Transform{ {0.0f, -0.43f, -1.5f}, {0.0f, 0.0f, 0.0f}, {0.50f, 0.50f, 0.50f} });
		world.addMeshTag(cube, MeshTag::Cube);
		world.addColor(cube, glm::vec4(0.25f, 0.85f, 0.30f, 1.0f));
	}
	if (modelCount > 5) {
		Entity cube = world.createEntity();
		world.addTransform(cube, Transform{ {2.0f, -0.45f, 0.8f}, {0.0f, 0.0f, 0.0f}, {0.50f, 0.50f, 0.50f} });
		world.addMeshTag(cube, MeshTag::Cube);
		world.addColor(cube, glm::vec4(0.20f, 0.35f, 0.95f, 1.0f));
	}
#elif RENDERING_LEVEL == 6
	setCubeMeshIndex(0);
	setSphereMeshIndex(1);
	if (modelCount > 0) {
		Entity ground = world.createEntity();
		world.addTransform(ground, Transform{ {0.0f, -1.1f, 0.0f}, {0.0f, 0.0f, 0.0f}, {7.5f, 0.03f, 7.5f} });
		world.addMeshTag(ground, MeshTag::Cube);
		world.addColor(ground, glm::vec4(0.78f, 0.78f, 0.80f, 1.0f));
	}
	if (modelCount > 1) {
		Entity pillar = world.createEntity();
		world.addTransform(pillar, Transform{ {-3.0f, -0.2f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.08f, 2.8f, 0.08f} });
		world.addMeshTag(pillar, MeshTag::Cube);
		world.addColor(pillar, glm::vec4(0.35f, 0.35f, 0.35f, 1.0f));
	}
	if (modelCount > 2) {
		Entity pillar = world.createEntity();
		world.addTransform(pillar, Transform{ {-2.6f, 0.2f, 0.3f}, {0.0f, 0.0f, 0.0f}, {0.10f, 2.2f, 0.10f} });
		world.addMeshTag(pillar, MeshTag::Cube);
		world.addColor(pillar, glm::vec4(0.90f, 0.25f, 0.20f, 1.0f));
	}
	if (modelCount > 3) {
		taauMovingProbe = world.createEntity();
		world.addTransform(taauMovingProbe, Transform{ {2.5f, 0.35f, -1.6f}, {0.0f, 0.0f, 0.0f}, {0.35f, 0.35f, 0.35f} });
		world.addMeshTag(taauMovingProbe, MeshTag::Cube);
		world.addColor(taauMovingProbe, glm::vec4(0.20f, 0.75f, 0.95f, 1.0f));
	}
	if (modelCount > 4) {
		taauEdgeProbe = world.createEntity();
		world.addTransform(taauEdgeProbe, Transform{ {3.7f, -0.35f, -2.8f}, {0.0f, 0.0f, 0.0f}, {0.22f, 1.6f, 0.22f} });
		world.addMeshTag(taauEdgeProbe, MeshTag::Cube);
		world.addColor(taauEdgeProbe, glm::vec4(0.25f, 0.85f, 0.30f, 1.0f));
	}
	if (modelCount > 5) {
		Entity bar = world.createEntity();
		world.addTransform(bar, Transform{ {0.0f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.12f, 0.9f, 0.12f} });
		world.addMeshTag(bar, MeshTag::Cube);
		world.addColor(bar, glm::vec4(0.85f, 0.70f, 0.20f, 1.0f));
	}
	if (modelCount > 6) {
		Entity bar = world.createEntity();
		world.addTransform(bar, Transform{ {0.25f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.10f, 0.9f, 0.10f} });
		world.addMeshTag(bar, MeshTag::Cube);
		world.addColor(bar, glm::vec4(0.30f, 0.30f, 0.30f, 1.0f));
	}
	if (modelCount > 7) {
		Entity bar = world.createEntity();
		world.addTransform(bar, Transform{ {-0.25f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.14f, 0.9f, 0.14f} });
		world.addMeshTag(bar, MeshTag::Cube);
		world.addColor(bar, glm::vec4(0.40f, 0.40f, 0.40f, 1.0f));
	}
	if (modelCount > 8) {
		Entity bar = world.createEntity();
		world.addTransform(bar, Transform{ {0.55f, -0.45f, 2.8f}, {0.0f, 0.0f, 0.0f}, {0.08f, 0.9f, 0.08f} });
		world.addMeshTag(bar, MeshTag::Cube);
		world.addColor(bar, glm::vec4(0.50f, 0.50f, 0.50f, 1.0f));
	}
#elif RENDERING_LEVEL == 7
	setCubeMeshIndex(0);
	setSphereMeshIndex(1);
	if (modelCount > 0) {
		Entity ground = world.createEntity();
		world.addTransform(ground, Transform{ {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {7.0f, 0.05f, 7.0f} });
		world.addMeshTag(ground, MeshTag::Cube);
		world.addColor(ground, glm::vec4(0.78f, 0.78f, 0.80f, 1.0f));
	}
	if (modelCount > 1) {
		Entity cube = world.createEntity();
		world.addTransform(cube, Transform{ {-1.5f, 0.15f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.35f, 0.35f, 0.35f} });
		world.addMeshTag(cube, MeshTag::Cube);
		world.addColor(cube, glm::vec4(0.35f, 0.35f, 0.35f, 1.0f));
	}
	if (modelCount > 2) {
		Entity cube = world.createEntity();
		world.addTransform(cube, Transform{ {1.6f, 0.35f, -1.2f}, {0.0f, 0.0f, 0.0f}, {0.45f, 0.45f, 0.45f} });
		world.addMeshTag(cube, MeshTag::Cube);
		world.addColor(cube, glm::vec4(0.90f, 0.25f, 0.20f, 1.0f));
	}
#elif RENDERING_LEVEL == 3 || RENDERING_LEVEL == 4
	setSphereMeshIndex(0);
	const uint32_t gridSize = 7u;
	const float spacing = 1.5f;
	const float halfExtent = (static_cast<float>(gridSize - 1) * 0.5f) * spacing;
	const glm::vec2 topLeft(-halfExtent, halfExtent);
	for (uint32_t row = 0; row < gridSize; ++row) {
		for (uint32_t col = 0; col < gridSize; ++col) {
			Entity sphere = world.createEntity();
			float x = topLeft.x + static_cast<float>(col) * spacing;
			float y = topLeft.y - static_cast<float>(row) * spacing;
			world.addTransform(sphere, Transform{ {x, y, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.7f, 0.7f, 0.7f} });
			world.addMeshTag(sphere, MeshTag::Sphere);
			float roughness = glm::clamp(static_cast<float>(col) / static_cast<float>(gridSize - 1), 0.05f, 1.0f);
			float metallic = glm::clamp(1.0f - static_cast<float>(row) / static_cast<float>(gridSize - 1), 0.1f, 1.0f);
			world.addPBRMaterial(sphere, PBRMaterial{ metallic, roughness, glm::vec3(1.0f, 0.765557f, 0.336057f) });
		}
	}
#elif RENDERING_LEVEL == 1 || RENDERING_LEVEL == 2
	setCubeMeshIndex(0);
	setSphereMeshIndex(1);
	if (modelCount > 0) {
		Entity center = world.createEntity();
		world.addTransform(center, Transform{ {0.0f, 0.0f, 0.0f}, {0.0f, glm::radians(-90.0f), 0.0f}, {1.0f, 1.0f, 1.0f} });
		world.addMeshTag(center, MeshTag::Cube);
	}
	if (modelCount > 1) {
		Entity left = world.createEntity();
		world.addTransform(left, Transform{ {-2.0f, 0.0f, -1.0f}, {0.0f, glm::radians(45.0f), 0.0f}, {0.75f, 0.75f, 0.75f} });
		world.addMeshTag(left, MeshTag::Cube);
	}
	if (modelCount > 2) {
		Entity right = world.createEntity();
		world.addTransform(right, Transform{ {2.0f, 0.0f, -1.0f}, {0.0f, glm::radians(-45.0f), 0.0f}, {0.75f, 0.75f, 0.75f} });
		world.addMeshTag(right, MeshTag::Cube);
	}
#endif
}

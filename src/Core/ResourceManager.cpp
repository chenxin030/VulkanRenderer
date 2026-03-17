#include "ResourceManager.h"

void ResourceManager::initResource(unsigned int modelCount) {
#if RENDERING_LEVEL == 3
	meshes.resize(1);
	meshUniformBuffer.resize(modelCount);
	transforms.resize(modelCount);
#elif RENDERING_LEVEL == 4
	meshes.resize(1);
	textures.resize(texPath.size());
	meshUniformBuffer.resize(modelCount);
	transforms.resize(modelCount);
#elif RENDERING_LEVEL == 5
	meshes.resize(2);
	meshUniformBuffer.resize(modelCount);
	transforms.resize(modelCount);

	if (modelCount > 0) {
		transforms[0].position = { 0.0f, -1.0f, 0.0f };
		transforms[0].rotation = { 0.0f, 0.0f, 0.0f };
		transforms[0].scale = { 6.0f, 0.05f, 6.0f };
	}
	if (modelCount > 1) {
		transforms[1].position = { -2.0f, 0.25f, 0.0f };
		transforms[1].scale = { 0.25f, 1.20f, 0.25f };
	}
	if (modelCount > 2) {
		transforms[2].position = { -2.0f, 1.95f, 0.0f };
		transforms[2].scale = { 0.50f, 0.50f, 0.50f };
	}
	if (modelCount > 3) {
		transforms[3].position = { 0.0f, -0.92f, -1.5f };
		transforms[3].scale = { 0.30f, 0.03f, 0.30f };
	}
	if (modelCount > 4) {
		transforms[4].position = { 0.0f, -0.43f, -1.5f };
		transforms[4].scale = { 0.50f, 0.50f, 0.50f };
	}
	if (modelCount > 5) {
		transforms[5].position = { 2.0f, -0.45f, 0.8f };
		transforms[5].scale = { 0.50f, 0.50f, 0.50f };
	}
#elif RENDERING_LEVEL == 6
	meshes.resize(2);
	meshUniformBuffer.resize(modelCount);
	transforms.resize(modelCount);

	// TAAU test scene (ghosting / thin-line shimmer / fast motion / edge & high-frequency texture)
	if (modelCount > 0) {
		// Ground plane to expose edge shimmer
		transforms[0].position = { 0.0f, -1.1f, 0.0f };
		transforms[0].rotation = { 0.0f, 0.0f, 0.0f };
		transforms[0].scale = { 7.5f, 0.03f, 7.5f };
	}
	if (modelCount > 1) {
		// Thin pillar group for thin-line flicker
		transforms[1].position = { -3.0f, -0.2f, 0.0f };
		transforms[1].scale = { 0.08f, 2.8f, 0.08f };
	}
	if (modelCount > 2) {
		transforms[2].position = { -2.6f, 0.2f, 0.3f };
		transforms[2].scale = { 0.10f, 2.2f, 0.10f };
	}
	if (modelCount > 3) {
		// Fast moving probe (to be animated in Renderer_TAAU)
		transforms[3].position = { 2.5f, 0.35f, -1.6f };
		transforms[3].scale = { 0.35f, 0.35f, 0.35f };
	}
	if (modelCount > 4) {
		// Edge clipping probe near screen edge
		transforms[4].position = { 3.7f, -0.35f, -2.8f };
		transforms[4].scale = { 0.22f, 1.6f, 0.22f };
	}
	if (modelCount > 5) {
		// High-frequency alternating bars (dense array)
		transforms[5].position = { 0.0f, -0.45f, 2.8f };
		transforms[5].scale = { 0.12f, 0.9f, 0.12f };
	}
	if (modelCount > 6) {
		transforms[6].position = { 0.25f, -0.45f, 2.8f };
		transforms[6].scale = { 0.10f, 0.9f, 0.10f };
	}
	if (modelCount > 7) {
		transforms[7].position = { -0.25f, -0.45f, 2.8f };
		transforms[7].scale = { 0.14f, 0.9f, 0.14f };
	}
	if (modelCount > 8) {
		transforms[8].position = { 0.55f, -0.45f, 2.8f };
		transforms[8].scale = { 0.08f, 0.9f, 0.08f };
	}
#elif RENDERING_LEVEL == 7
	meshes.resize(2);
	meshUniformBuffer.resize(modelCount);
	transforms.resize(modelCount);

	if (modelCount > 0) {
		// Ground plane for reflections
		transforms[0].position = { 0.0f, -1.0f, 0.0f };
		transforms[0].rotation = { 0.0f, 0.0f, 0.0f };
		transforms[0].scale = { 7.0f, 0.05f, 7.0f };
	}
	if (modelCount > 1) {
		// Reflective cube
		transforms[1].position = { -1.5f, 0.15f, 0.0f };
		transforms[1].scale = { 0.35f, 0.35f, 0.35f };
	}
	if (modelCount > 2) {
		// Secondary cube to show secondary reflections
		transforms[2].position = { 1.6f, 0.35f, -1.2f };
		transforms[2].scale = { 0.45f, 0.45f, 0.45f };
	}
#endif
#if RENDERING_LEVEL == 1 || RENDERING_LEVEL == 2
	meshes.resize(modelPath.size());
	textures.resize(texPath.size());
	meshUniformBuffer.resize(modelCount);
	transforms.resize(modelCount);

	// Object 1 - Center
	if (modelCount > 0) {
		transforms[0].position = { 0.0f, 0.0f, 0.0f };
		transforms[0].rotation = { 0.0f, glm::radians(-90.0f), 0.0f };
		transforms[0].scale = { 1.0f, 1.0f, 1.0f };
	}

	// Object 2 - Left
	if (modelCount > 1) {
		transforms[1].position = { -2.0f, 0.0f, -1.0f };
		transforms[1].rotation = { 0.0f, glm::radians(45.0f), .0f };
		transforms[1].scale = { 0.75f, 0.75f, 0.75f };
	}

	// Object 3 - Right
	if (modelCount > 2) {
		transforms[2].position = { 2.0f, 0.0f, -1.0f };
		transforms[2].rotation = { 0.0f, glm::radians(-45.0f), 0.0f };
		transforms[2].scale = { 0.75f, 0.75f, 0.75f };
	}
#endif
}

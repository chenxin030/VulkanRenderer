cmake_minimum_required(VERSION 3.28)

# vkrEngine — Unified Vulkan Renderer Demo
add_executable(15_vkrEngine
    vkrEngine.cpp
    VkrModel.h
    VkrModel.cpp
    Scene.h
    GpuProfiler.h
    GpuProfiler.cpp
    VkrRenderer.h
    VkrRenderer.cpp
)

target_link_libraries(15_vkrEngine PRIVATE vkr_base)

# Compile Slang shaders
vkr_add_example_slang_shaders(15_vkrEngine
    "${CMAKE_SOURCE_DIR}/shaders/15_vkrEngine"
    "${CMAKE_SOURCE_DIR}/shaders"
)

# Copy GLFW DLL
vkr_copy_glfw_dll(15_vkrEngine)

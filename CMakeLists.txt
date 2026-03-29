# 项目根目录
cmake_minimum_required(VERSION 3.28)

set(NAME vulkanRenderer)
project(${NAME})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/Zm300)
endif()

# 设置解决方案文件夹属性（用于 Visual Studio）
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message(STATUS "Looking for Vulkan...")
find_package(Vulkan REQUIRED)

find_library(GLFW_LIBRARY
    NAMES glfw3 glfw3dll
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/GLFW3/lib
    NO_DEFAULT_PATH
)
if(NOT GLFW_LIBRARY)
    message(WARNING "GLFW library not found in external/GLFW3/lib")
endif()

# 全局预处理（更推荐后续迁移到 target_compile_definitions）
if(WIN32)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
endif()
add_definitions(
    -DNOMINMAX
    -D_USE_MATH_DEFINES
    -D_CRT_SECURE_NO_WARNINGS
    -DVULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    -DGLFW_INCLUDE_VULKAN
    -DPLATFORM_DESKTOP
    -DGLM_FORCE_RADIANS
    -DGLM_FORCE_DEFAULT_ALIGNED_GENTYPES
    -DGLM_FORCE_DEPTH_ZERO_TO_ONE
    -DGLM_ENABLE_EXPERIMENTAL
    -DVK_MODEL_DIR="${CMAKE_SOURCE_DIR}/assets/models/"
    -DVK_TEXTURE_DIR="${CMAKE_SOURCE_DIR}/assets/textures/"
)

option(FORCE_VALIDATION "Force validation layers at compile time" OFF)
if(FORCE_VALIDATION)
    add_definitions(-DFORCE_VALIDATION)
endif()

add_library(vkr_deps INTERFACE)
target_include_directories(vkr_deps INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/GLFW3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/external/tinygltf
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ktx/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ktx/other_include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vkr_deps INTERFACE
    Vulkan::Vulkan
    ${GLFW_LIBRARY}
)

add_library(Base STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/Platform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase.h   
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase_commands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase_swapchain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/VulkanBase_Resource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/Mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Base/Mesh.cpp
)
target_link_libraries(Base PUBLIC vkr_deps)

# 兼容各 example 现有链接名（它们都链接 vkr_base）
add_library(vkr_base ALIAS Base)

# -----------------------------
# ImGui shared target
# -----------------------------
file(GLOB VKR_IMGUI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui*.cpp
)
add_library(Imgui STATIC
    ${VKR_IMGUI_SOURCES}
)
target_link_libraries(Imgui PUBLIC vkr_deps)
source_group("ImGui" FILES ${VKR_IMGUI_SOURCES})

file(GLOB IMGUI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui*.cpp
)
file(GLOB_RECURSE SHADER_SLANG_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang
)
if(SHADER_SLANG_FILES)
    add_slang_shader_target(
        ShaderCompileTarget
        SOURCES ${SHADER_SLANG_FILES}
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    )
endif()

if(TARGET ShaderCompileTarget)
    add_dependencies(${NAME} ShaderCompileTarget)
endif()

# -----------------------------
# Examples
# -----------------------------
add_subdirectory(src)

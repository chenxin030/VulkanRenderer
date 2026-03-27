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

# 包含着色器编译模块
include(cmake/shader.cmake)

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

# -----------------------------
# Common deps target (no sources)
# 注意：Renderer/Scene/ResourceManager 仍依赖 RENDERING_LEVEL 宏，
# 现阶段不能做成“单份静态库供多个 example 复用”，否则不同 example 会冲突。
# 后续把 Renderer 拆成 VulkanBase + Example 后，再恢复真正的 vkr_base 静态库。
# -----------------------------
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vkr_deps INTERFACE
    Vulkan::Vulkan
    ${GLFW_LIBRARY}
)

# -----------------------------
# Base core target (shared VulkanBase implementation)
# -----------------------------
add_library(vkr_base STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/VulkanBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/VulkanBase_commands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/VulkanBase_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/VulkanBase_swapchain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/VulkanBase_utils.cpp
)
target_link_libraries(vkr_base PUBLIC vkr_deps)
set_target_properties(vkr_base PROPERTIES
    FOLDER base
)

# -----------------------------
# Legacy exe（保留原入口，便于对照）
# -----------------------------
option(VKR_BUILD_LEGACY "Build legacy vulkanRenderer target" OFF)

file(GLOB_RECURSE APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App/*.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App/*.cc
)
file(GLOB_RECURSE APP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App/*.hxx
)

if(VKR_BUILD_LEGACY)
    file(GLOB_RECURSE CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/*.cc
    )
    file(GLOB_RECURSE CORE_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/*.hxx
    )
    list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
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

    add_executable(${NAME})
    target_sources(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Application.h
        ${APP_SOURCES}
        ${APP_HEADERS}
        ${CORE_SOURCES}
        ${CORE_HEADERS}
        ${IMGUI_SOURCES}
        ${SHADER_SLANG_FILES}
    )
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/App
    )
    target_link_libraries(${NAME} PRIVATE vkr_deps)
    target_compile_definitions(${NAME} PRIVATE
        RENDERING_LEVEL=8
        VK_SHADERS_DIR="${CMAKE_SOURCE_DIR}/shaders/"
    )

    if(TARGET ShaderCompileTarget)
        add_dependencies(${NAME} ShaderCompileTarget)
    endif()

    if(WIN32)
        set(GLFW_DLL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/GLFW3/bin/glfw3.dll")
        if(EXISTS ${GLFW_DLL_PATH})
            add_custom_command(TARGET ${NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${GLFW_DLL_PATH}
                    $<TARGET_FILE_DIR:${NAME}>
            )
        endif()
    endif()

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${NAME})
else()
    # 默认启动一个 example
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT 1_triangle)
endif()

# -----------------------------
# Examples
# -----------------------------
add_subdirectory(src/Examples)

cmake_minimum_required(VERSION 3.28)

set(EXAMPLE_NAME 1_triangle)
set(EXAMPLE_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(EXAMPLE_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

add_executable(${EXAMPLE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/StandaloneMain.cpp
)

target_include_directories(${EXAMPLE_NAME} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/src/Examples
)

target_sources(${EXAMPLE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TriangleRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TriangleRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StandaloneMain.cpp
)

target_link_libraries(${EXAMPLE_NAME} PRIVATE vkr_deps vkr_base)

target_compile_definitions(${EXAMPLE_NAME} PRIVATE
    VK_SHADERS_DIR="${EXAMPLE_SHADER_OUTPUT_DIR}/"
)

vkr_add_example_slang_shaders(${EXAMPLE_NAME} "${EXAMPLE_SHADER_SOURCE_DIR}" "${EXAMPLE_SHADER_OUTPUT_DIR}")

vkr_copy_glfw_dll(${EXAMPLE_NAME})

set_target_properties(${EXAMPLE_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)


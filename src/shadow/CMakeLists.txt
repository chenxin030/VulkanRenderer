cmake_minimum_required(VERSION 3.28)

set(EXAMPLE_NAME 4_shadow)
set(EXAMPLE_SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders/shadow")

add_executable(${EXAMPLE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/StandaloneMain.cpp
)

target_include_directories(${EXAMPLE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_sources(${EXAMPLE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ShadowRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/StandaloneMain.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/ShadowRenderer.cpp
)

target_link_libraries(${EXAMPLE_NAME} PRIVATE vkr_deps vkr_base vkr_imgui)

vkr_copy_glfw_dll(${EXAMPLE_NAME})

set_target_properties(${EXAMPLE_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)


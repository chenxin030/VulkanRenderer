cmake_minimum_required(VERSION 3.28)

set(EXAMPLE_NAME 14_particles)

add_executable(${EXAMPLE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/StandaloneMain.cpp
)

target_include_directories(${EXAMPLE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/Base
)

target_sources(${EXAMPLE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ParticleRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ParticleRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StandaloneMain.cpp
)

file(GLOB EXAMPLE_SHADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/shaders/14_particles/*.slang"
)
if(EXAMPLE_SHADER_FILES)
    set_source_files_properties(${EXAMPLE_SHADER_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("shaders" FILES ${EXAMPLE_SHADER_FILES})
    target_sources(${EXAMPLE_NAME} PRIVATE ${EXAMPLE_SHADER_FILES})
endif()

vkr_add_example_slang_shaders(${EXAMPLE_NAME}
    "${CMAKE_SOURCE_DIR}/shaders/14_particles"
    "${CMAKE_SOURCE_DIR}/shaders"
)

target_link_libraries(${EXAMPLE_NAME} PRIVATE vkr_deps vkr_base vkr_imgui)

vkr_copy_glfw_dll(${EXAMPLE_NAME})

set_target_properties(${EXAMPLE_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

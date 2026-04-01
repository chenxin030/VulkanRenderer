cmake_minimum_required(VERSION 3.28)

function(vkr_copy_glfw_dll target_name)
    if(WIN32)
        set(GLFW_DLL_PATH "${CMAKE_SOURCE_DIR}/external/GLFW3/bin/glfw3.dll")
        if(EXISTS ${GLFW_DLL_PATH})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${GLFW_DLL_PATH}
                    $<TARGET_FILE_DIR:${target_name}>
            )
        endif()
    endif()
endfunction()

function(vkr_add_example_slang_shaders target_name shader_source_dir shader_output_dir)
    file(GLOB EXAMPLE_SHADER_SOURCES CONFIGURE_DEPENDS "${shader_source_dir}/*.slang")
    if(NOT EXAMPLE_SHADER_SOURCES)
        return()
    endif()

    # 将 shader 编译步骤直接挂在 example 目标上，避免生成独立 *_shaders 工程
    find_program(SLANGC_EXECUTABLE
        NAMES slangc slangc.exe
        HINTS ${CMAKE_SOURCE_DIR}/external/slang/bin
    )

    if(NOT SLANGC_EXECUTABLE)
        message(WARNING "Slang compiler not found. Skip shader compile for target: ${target_name}")
        return()
    endif()

    file(MAKE_DIRECTORY "${shader_output_dir}")

    foreach(SHADER_SOURCE ${EXAMPLE_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        set(FINAL_OUTPUT "${shader_output_dir}/${SHADER_NAME}.spv")

        set(SHADER_ENTRY_ARGS -entry vertMain -entry fragMain)
        if(SHADER_NAME MATCHES "_comp$" OR SHADER_NAME MATCHES "_build$")
            set(SHADER_ENTRY_ARGS -entry compMain)
        endif()

        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${SLANGC_EXECUTABLE}
                ${SHADER_SOURCE}
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                ${SHADER_ENTRY_ARGS}
                -o ${FINAL_OUTPUT}
            COMMENT "Compiling ${SHADER_NAME}.slang for ${target_name}"
            VERBATIM
        )
    endforeach()

    target_sources(${target_name} PRIVATE ${EXAMPLE_SHADER_SOURCES})
endfunction()

add_compile_definitions(
    VK_SHADERS_DIR="${CMAKE_SOURCE_DIR}/shaders/"
)

add_subdirectory(Instanced)
add_subdirectory(pbr)
add_subdirectory(IBL_pbr)
add_subdirectory(shadow)
add_subdirectory(taau)
add_subdirectory(ssr)
add_subdirectory(culling)
add_subdirectory(deferredShading)
add_subdirectory(postfx)
add_subdirectory(atmospheric)
add_subdirectory(gi)
add_subdirectory(clustered)
add_subdirectory(skinning)
add_subdirectory(particles)



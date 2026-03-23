function(add_slang_shader_target TARGET)
    cmake_parse_arguments(SHADER "" "OUTPUT_DIR" "SOURCES" ${ARGN})
    
    # 如果没有提供输出目录，使用默认值
    if(NOT SHADER_OUTPUT_DIR)
        set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    endif()
    
    set(SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders)
    
    # 创建输出目录
    file(MAKE_DIRECTORY ${SHADERS_OUTPUT_DIR})
    
    # 查找slang编译器
    find_program(SLANGC_EXECUTABLE 
        NAMES slangc slangc.exe
        HINTS ${CMAKE_SOURCE_DIR}/external/slang/bin
    )
    
    if(NOT SLANGC_EXECUTABLE)
        message(WARNING "Slang compiler not found. Shaders will not be compiled.")
        add_custom_target(${TARGET} COMMAND ${CMAKE_COMMAND} -E echo "Slang compiler not found")
        return()
    endif()
    
    set(COMPILED_SHADERS "")
    
    # 遍历所有着色器源文件
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        get_filename_component(SHADER_EXT ${SHADER_SOURCE} EXT)
        
        set(SPV_OUTPUT ${SHADERS_OUTPUT_DIR}/${SHADER_NAME}.spv)
        
        # 添加复制命令到源码目录（用于开发）
        set(FINAL_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)

        set(SHADER_ENTRY_ARGS -entry vertMain -entry fragMain)
        if(SHADER_NAME MATCHES "_comp$" OR SHADER_NAME MATCHES "_build$")
            set(SHADER_ENTRY_ARGS -entry compMain)
        endif()
        
        # 编译命令
        add_custom_command(
            OUTPUT ${SPV_OUTPUT}
            COMMAND ${SLANGC_EXECUTABLE} 
                ${SHADER_SOURCE} 
                -target spirv 
                -profile spirv_1_4           # 🔴 指定SPIR-V 1.4版本
                -emit-spirv-directly          # 🔴 直接生成SPIR-V
                -fvk-use-entrypoint-name      # 🔴 使用入口点名称
                ${SHADER_ENTRY_ARGS}
                -o ${SPV_OUTPUT}
            WORKING_DIRECTORY ${SHADERS_OUTPUT_DIR}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_NAME} shader to SPIR-V"
            VERBATIM
        )
        
        # 添加复制命令到目标目录
        add_custom_command(
            OUTPUT ${FINAL_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPV_OUTPUT} ${FINAL_OUTPUT}
            DEPENDS ${SPV_OUTPUT}
            COMMENT "Copying ${SHADER_NAME}.spv to shaders directory"
        )
        
        list(APPEND COMPILED_SHADERS ${FINAL_OUTPUT})
    endforeach()
    
    # 创建自定义目标
    add_custom_target(${TARGET} DEPENDS ${COMPILED_SHADERS})
    
    # 标记生成的文件
    set_source_files_properties(${COMPILED_SHADERS} PROPERTIES GENERATED TRUE)
    
    # 统计着色器文件数量
    list(LENGTH SHADER_SOURCES SHADER_COUNT)
    message(STATUS "Found ${SHADER_COUNT} shader files to compile")
    
endfunction()
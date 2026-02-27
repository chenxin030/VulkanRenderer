# 项目根目录
cmake_minimum_required(VERSION 3.28)

set(NAME vulkanRenderer)
project(${NAME})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置解决方案文件夹属性（用于 Visual Studio）
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(STARTUP_PROJECT_NAME ${NAME})

# 设置 VulkanRenderer 为启动项目（仅对 Visual Studio 有效）
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${NAME})

# 包含着色器编译模块
include(cmake/shader.cmake)

# Find Vulkan
message(STATUS "Looking for Vulkan...")
find_package(Vulkan REQUIRED)

# Platform define for Windows
if(WIN32)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
endif()

# Preprocessor defines
add_definitions(
    -DNOMINMAX 
    -D_USE_MATH_DEFINES
    -D_CRT_SECURE_NO_WARNINGS
    -DVULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    -DGLFW_INCLUDE_VULKAN
    -DPLATFORM_DESKTOP
    -DVK_MODEL_DIR="${CMAKE_SOURCE_DIR}/assets/models/"
    -DVK_TEXTURE_DIR="${CMAKE_SOURCE_DIR}/assets/textures/"
    -DVK_SHADERS_DIR="${CMAKE_SOURCE_DIR}/shaders/"
)

option(FORCE_VALIDATION "Force validation layers at compile time" OFF)
if(FORCE_VALIDATION)
    add_definitions(-DFORCE_VALIDATION)
endif()

# 创建可执行文件
add_executable(${NAME})

# 先查找源文件
# App 文件夹
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

# Core 文件夹
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

# 检查是否找到源文件
if(NOT APP_SOURCES AND NOT CORE_SOURCES)
    message(WARNING "No source files found in src/App or src/Core directories!")
else()
    message(STATUS "Found App sources: ${APP_SOURCES}")
    message(STATUS "Found Core sources: ${CORE_SOURCES}")
endif()

# 添加所有源文件到目标
target_sources(${NAME} PRIVATE
    ${APP_SOURCES}
    ${APP_HEADERS}
    ${CORE_SOURCES}
    ${CORE_HEADERS}
)

# 设置包含目录
target_include_directories(${NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/GLFW3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/external/tinygltf
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/App
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core
    ${Vulkan_INCLUDE_DIRS}
)

# 源文件分组
source_group("App\\Source Files" FILES ${APP_SOURCES})
source_group("App\\Header Files" FILES ${APP_HEADERS})
source_group("Core\\Source Files" FILES ${CORE_SOURCES})
source_group("Core\\Header Files" FILES ${CORE_HEADERS})

# 添加着色器文件
file(GLOB_RECURSE SHADER_SLANG_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang
)

if(SHADER_SLANG_FILES)
    list(LENGTH SHADER_SLANG_FILES SHADER_COUNT)
    
    # 调用着色器编译函数
    add_slang_shader_target(
        ShaderCompileTarget
        SOURCES ${SHADER_SLANG_FILES}
        OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    )
    
    # 将着色器编译目标设为依赖
    add_dependencies(${NAME} ShaderCompileTarget)
    
    # 在IDE中显示着色器文件
    source_group("Shaders\\Source Files" FILES ${SHADER_SLANG_FILES})
    target_sources(${NAME} PRIVATE ${SHADER_SLANG_FILES})
    
    message(STATUS "Found and added ${SHADER_COUNT} slang shader files")
else()
    message(WARNING "No .slang shader files found in ${CMAKE_CURRENT_SOURCE_DIR}/shaders/")
endif()

# 查找GLFW库
find_library(GLFW_LIBRARY 
    NAMES glfw3 glfw3dll
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/GLFW3/lib
    NO_DEFAULT_PATH
)

if(NOT GLFW_LIBRARY)
    message(WARNING "GLFW library not found in external/GLFW3/lib")
endif()

# 链接库
target_link_libraries(${NAME} PRIVATE
    Vulkan::Vulkan
    ${GLFW_LIBRARY}
)

# 设置目标属性
set_target_properties(${NAME} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 复制 glfw3.dll 到输出目录
if(WIN32)
    set(GLFW_DLL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/GLFW3/bin/glfw3.dll")
    if(EXISTS ${GLFW_DLL_PATH})
        add_custom_command(TARGET ${NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GLFW_DLL_PATH}
                $<TARGET_FILE_DIR:${NAME}>
            COMMENT "Copying glfw3.dll to output directory"
        )
        message(STATUS "Will copy glfw3.dll from ${GLFW_DLL_PATH}")
    else()
        message(WARNING "glfw3.dll not found at ${GLFW_DLL_PATH}")
    endif()
endif()
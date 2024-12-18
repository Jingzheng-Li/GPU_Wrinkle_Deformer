cmake_minimum_required(VERSION 3.18)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(library_name gipc)

# 项目配置
project(gpu_ipc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# 检查是否设置了编译类型，默认为 Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Release or Debug)" FORCE)
endif()

# 优化编译选项
# 1. Release 模式的优化选项
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
# 2. Debug 模式的调试信息
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA 架构设置，自动选择合适的架构
find_package(CUDAToolkit REQUIRED)
if (NOT DEFINED CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES 86)  # 默认设置为 86 架构
endif()

# 输出配置信息
message(STATUS "Building for CUDA architectures: ${CUDA_ARCHITECTURES}")
message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")

# 更新CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

# 查找所需库
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(CGAL REQUIRED)

# 创建可执行文件
add_executable(${library_name})

# 设置编译标准
target_compile_features(${library_name} PRIVATE cxx_std_17 cuda_std_17)

# 手动管理源文件，避免 GLOB_RECURSE 性能开销
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE GIPC_HEADER CONFIGURE_DEPENDS "src/*.h" "src/*.hpp" "src/*.cuh" "src/*.inl" "src/*.inc")

# 包含头文件目录
target_include_directories(${library_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FEMENERGY
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LSOLVER
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUIPC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/UTILS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FRONTEND
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DEFORMER
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 设置源文件
target_sources(${library_name} PRIVATE ${GIPC_SOURCE} ${GIPC_HEADER})

# 设置编译定义，一次性添加所有的宏定义
target_compile_definitions(${library_name} PRIVATE 
    DEFAULT_WARPS_PERBLOCK=8
    DEFAULT_THREADS_PERWARP=32
    DEFAULT_THREADS_PERBLOCK=256
    USE_SNK

    USE_GIPCFRICTION
    USE_DOUBLE_PRECISION
    # GPUIPC_ANIMATION
    # GPUIPC_DRAG
    # GPUIPC_HTTP
    GPUIPC_DEFORMER
)

# 链接 CUDA 库和其他依赖库
target_link_libraries(${library_name} PRIVATE 
    cusparse 
    cublas 
    cusolver 
    cufft 
    cudart_static 
    GLUT::GLUT 
    GLEW::GLEW 
    OpenGL::GLU 
    Eigen3::Eigen
    CUDA::cuda_driver 
    CUDA::cudart
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${library_name} PRIVATE OpenMP::OpenMP_CXX)
endif()

# 查找并链接 cpp-httplib
find_package(PkgConfig REQUIRED)
pkg_check_modules(HTTPLIB REQUIRED cpp-httplib)
target_include_directories(${library_name} PRIVATE ${HTTPLIB_INCLUDE_DIRS})
target_link_libraries(${library_name} PRIVATE ${HTTPLIB_LIBRARIES} pthread)

# 设置 CUDA 相关属性，只设置一次 CUDA 架构
set_target_properties(${library_name} PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}  # 仅设置一次
)

# **区分 Debug 和 Release 构建：为不同的构建类型设置不同的编译选项**
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring Debug build")
    target_compile_options(${library_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: 
            -g                    # 生成调试信息
            -G                    # 启用设备代码调试
            --use_fast_math       # 使用快速数学库
            --extended-lambda     # 支持扩展 lambda
            --expt-relaxed-constexpr # 支持宽松的 constexpr
            --default-stream=per-thread
        >
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring Release build")
    target_compile_options(${library_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: 
            -O3                   # 极致优化
            --use_fast_math       # 使用快速数学库
            --extended-lambda     # 支持扩展 lambda
            --expt-relaxed-constexpr # 支持宽松的 constexpr
            --default-stream=per-thread
        >
    )
endif()

# 设置 source_group 来整理源代码结构
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HEADER})

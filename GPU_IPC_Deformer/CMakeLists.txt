cmake_minimum_required(VERSION 3.18)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

# 项目配置
project(gpu_ipc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# 如果没有指定构建类型，默认为 Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Release or Debug)" FORCE)
endif()

# 让 CMake 优先使用带有 Config 模式的包（比如 vcpkg 安装的）
# 而不是系统 /usr/lib/cmake 下带 Module 模式的老旧包。
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# 如有必要，手动指定 freeglut 的目录，强制使用 vcpkg 安装位置
# （可在 vcpkg install freeglut 后验证 exact 路径）
# set(freeglut_DIR "/home/jingzheng/Public/Vcpkg/vcpkg/installed/x64-linux/share/freeglut")

# 使用 vcpkg 的官方 toolchain
# 这在命令行里 -DCMAKE_TOOLCHAIN_FILE=... 即可
# 不要再 append "/usr/local" 到 CMAKE_PREFIX_PATH，以免冲突
# list(APPEND CMAKE_PREFIX_PATH "/usr/local")   # <-- 建议删除

# ====================== CUDA 架构设置 ======================
find_package(CUDAToolkit REQUIRED)
if (NOT DEFINED CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES 86)  # 默认设置为 86 架构
endif()
message(STATUS "Building for CUDA architectures: ${CUDA_ARCHITECTURES}")
message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")

# ====================== 第三方库查找 ======================
# 全部使用 vcpkg 提供的 config 模式，若找不到则会抛错
find_package(Eigen3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
# find_package(freeglut CONFIG REQUIRED)  # 关键：需要链接 freeglut::freeglut
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)

# 创建可执行文件
set(library_name gipc)
add_executable(${library_name})

# 设置编译标准
target_compile_features(${library_name} PRIVATE cxx_std_17 cuda_std_17)

# 手动管理源文件，避免 GLOB_RECURSE 的缓存问题
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
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 将源文件添加到 target
target_sources(${library_name} PRIVATE ${GIPC_SOURCE} ${GIPC_HEADER})

# 设置编译定义
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

# 链接依赖的库
target_link_libraries(${library_name} PRIVATE 
    # CUDA 相关
    cusparse 
    cublas 
    cusolver 
    cufft 
    cudart_static
    CUDA::cuda_driver 
    CUDA::cudart

    # OpenGL & freeglut
    # freeglut::freeglut
    GLEW::GLEW 
    OpenGL::GLU

    # 其他
    Eigen3::Eigen
    Boost::boost
    CGAL::CGAL
    httplib::httplib
)

# 如果存在 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(${library_name} PRIVATE OpenMP::OpenMP_CXX)
endif()

# 设置 CUDA 相关属性
set_target_properties(${library_name} PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)

# Debug / Release 区分优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring Debug build")
    target_compile_options(${library_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: 
            -g  # 调试信息
            -G  # 启用设备代码调试
            --use_fast_math
            --extended-lambda
            --expt-relaxed-constexpr
            --default-stream=per-thread
        >
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring Release build")
    target_compile_options(${library_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: 
            -O3
            --use_fast_math
            --extended-lambda
            --expt-relaxed-constexpr
            --default-stream=per-thread
        >
    )
endif()

# 分组源代码
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HEADER})

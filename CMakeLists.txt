cmake_minimum_required(VERSION 3.18)
project(GPUDeepLearning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Find CUDA
find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# CUDA kernel library
add_library(cuda_kernels STATIC
    cuda_kernels/matrix_ops.cu
    cuda_kernels/activation_kernels.cu
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --ptxas-options=-v
    >
)

# C++ library
add_library(gpu_dl_core STATIC
    src/tensor.cpp
)

target_link_libraries(gpu_dl_core
    cuda_kernels
    ${CUDA_LIBRARIES}
)


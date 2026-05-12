cmake_minimum_required(VERSION 3.18)
project(aberration_cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90")

find_package(CUDAToolkit 12.8 REQUIRED)

set(CUDA_SOURCES
    src/kernels.cu
    src/operations.cu
)

add_library(aberration_cuda_kernels STATIC ${CUDA_SOURCES})

target_include_directories(aberration_cuda_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(aberration_cuda_kernels
    PUBLIC
        CUDA::cudart_static
        dolphinbackend
)

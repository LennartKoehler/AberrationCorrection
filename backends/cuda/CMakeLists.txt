cmake_minimum_required(VERSION 3.18)
project(CudaAberrationBackend VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90")

add_library(cuda_aberration_backend STATIC
    src/CUDAAberrationBackend.cpp
)

target_include_directories(cuda_aberration_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Link against Dolphin's cuda_backend — it is already built by
# add_subdirectory(lib/dolphin) in the root CMakeLists.txt.
# No need to copy or add_subdirectory Dolphin's code here.
# The PUBLIC include dirs of cuda_backend (including cuda_backend/ headers)
# propagate automatically.
add_subdirectory(cuda_kernels)
add_subdirectory(tests)

target_link_libraries(cuda_aberration_backend
    PUBLIC
        cuda_backend
        aberration_cuda_kernels
        dolphinbackend
        aberration_backend
)

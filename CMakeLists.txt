cmake_minimum_required(VERSION 3.18)
project(AberrationCorrection VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

# Prevent CMake from finding the system-installed dolphin package.
# We want to use the local source via add_subdirectory() instead.
set(CMAKE_DISABLE_FIND_PACKAGE_dolphin ON CACHE BOOL "" FORCE)

# Only build the backends + dolphin_image from the dolphin submodule.
# The main `dolphin` library (deconvolution, PSF, etc.) is not needed.
set(BUILD_DOLPHIN_LIBRARY OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(lib/dolphin)
add_subdirectory(backends/aberration_backend)
add_subdirectory(backends/cpu)
add_subdirectory(backends/cuda)

add_library(aberrationcorrection STATIC
    src/main.cpp
    src/ImageConversion.cpp
    src/ThreadPool.cpp
    src/ParameterSearch.cpp
    src/SphereDetector.cpp)



target_include_directories(aberrationcorrection
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(aberrationcorrection INTERFACE
  $<$<CONFIG:Debug>:-g;-O0;-fno-omit-frame-pointer>
)


target_link_libraries(aberrationcorrection
    PRIVATE
        dolphin_image
        dolphinbackend
        cpu_backend
        aberration_backend
        cuda_aberration_backend
        cpu_aberration_backend
        project_options
)

add_subdirectory(tests)

cmake_minimum_required(VERSION 3.18)
project(AberrationCpuBackend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

add_library(cpu_aberration_backend STATIC
    src/CPUAberrationBackend.cpp
)

target_include_directories(cpu_aberration_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_subdirectory(tests)

target_link_libraries(cpu_aberration_backend
    PUBLIC
        cpu_backend
        dolphinbackend
        aberration_backend
)

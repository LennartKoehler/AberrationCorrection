cmake_minimum_required(VERSION 3.18)
project(aberrationbackend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_library(aberration_backend INTERFACE)

target_include_directories(aberration_backend INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


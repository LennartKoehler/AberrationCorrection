add_executable(zernike_cpu_test zernike_test.cpp)
set(CMAKE_BUILD_TYPE Debug)

target_link_libraries(zernike_cpu_test
    PRIVATE
        cpu_aberration_backend
        project_options
)

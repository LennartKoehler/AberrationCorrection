add_executable(zernike_cuda_test zernike_test.cpp)

target_link_libraries(zernike_cuda_test
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        project_options
)

add_executable(aberration_pipeline_test aberration_pipeline_test.cpp)

target_link_libraries(aberration_pipeline_test
    PRIVATE
        aberration_cuda_backend
        dolphin_image
        project_options
)

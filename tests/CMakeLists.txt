add_executable(aberration_pipeline_test aberration_pipeline_test.cpp)

target_link_libraries(aberration_pipeline_test
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        dolphin_image
        project_options
)

add_executable(zernike_phase_plots zernike_phase_plots.cpp)

target_link_libraries(zernike_phase_plots
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        dolphin_image
        aberrationcorrection
        project_options
)

add_executable(apply_zernike_phase apply_zernike_phase.cpp)

target_link_libraries(apply_zernike_phase
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        dolphin_image
        aberrationcorrection
        project_options
)

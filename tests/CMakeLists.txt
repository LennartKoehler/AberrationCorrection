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

add_executable(add_subtract_zernike add_subtract_zernike.cpp)

target_link_libraries(add_subtract_zernike
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        dolphin_image
        aberrationcorrection
        project_options
)

add_executable(real_vs_complex_zernike real_vs_complex_zernike.cpp)

target_link_libraries(real_vs_complex_zernike
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        dolphin_image
        aberrationcorrection
        project_options
)

add_executable(find_zernike_coeff_test find_zernike_coeff_test.cpp)

target_link_libraries(find_zernike_coeff_test
    PRIVATE
        cuda_aberration_backend
        cpu_aberration_backend
        dolphin_image
        aberrationcorrection
        project_options
)

add_executable(image_conversion_test image_conversion_test.cpp)

target_link_libraries(image_conversion_test
    PRIVATE
        aberrationcorrection
        dolphin_image
        dolphinbackend
        cpu_backend
        cpu_aberration_backend
        cuda_aberration_backend
        project_options
)

add_executable(sphere_detector_test sphere_detector_test.cpp)

target_link_libraries(sphere_detector_test
    PRIVATE
        aberrationcorrection
        dolphin_image
        project_options
)

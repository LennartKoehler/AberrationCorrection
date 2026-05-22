
/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#include "aberrationcorrection/ImageConversion.h"
#include "aberrationcorrection/BackendFactory.h"
#include "dolphin_image/Image3D.h"
#include <cmath>
#include <cstring>
#include <itkConstantPadImageFilter.h>
#include <itkMirrorPadImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageDuplicator.h>


ComplexData ImageConversion::convertImageToComplexData(
    const Image3D& input) {

    CuboidShape shape = input.getShape();
    ComplexData result = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceComplexFull(shape);

    int index = 0;

    for (const auto& it : input) {
        result[index][0] = static_cast<real_t>(it);
        result[index][1] = 0.0;
        index ++;
    }

    return result;
}
RealData ImageConversion::convertImageToRealData(
    const Image3D& input) {

    CuboidShape shape = input.getShape();
    RealData result = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(shape);
    // since the memory layout for inplace fft is noncontiguous the image pointer cant be reinterpreted as real_t*
    // thats also why this has to be done on the cpu first, and then another copy operation of the memory
    // with correct layout to gpu

    int index = 0;

    for (const auto& it : input) {
        result[index] = static_cast<real_t>(it);
        index ++;
    }

    return result;
}

Image3D ImageConversion::convertComplexDataToImage(
        const ComplexData& input){

    Image3D output(input.getSize(), 0.0f);

    int index = 0;
    for (auto& it : output) {
        real_t real = input[index][0];
        real_t imag = input[index][1];
        // it = static_cast<float>(std::sqrt(real * real + imag * imag));
        it = static_cast<float>(real);
        index ++;
    }

    return output;
}

// TODO make this conversion part of the backend? because if data is on cuda device then this wont work
// but then the backend needs to know about Image3D -> not good
Image3D ImageConversion::convertRealDataToImage(
        const RealData& input){

    Image3D output(input.getSize(), 0.0f);
    // since the memory layout for inplace fft is noncontiguous the image pointer cant be reinterpreted as real_t*
    // thats also why this has to be done on the cpu first, and then another copy operation of the memory
    // with correct layout to gpu

    int index = 0;
    for (auto& it : output) {
        real_t real = input[index];
        it = static_cast<float>(real);
        index ++;
    }

    return output;
}


// ============================================================================
//  conversions using buffer pointer memcpy
// ============================================================================

RealData ImageConversion::convertImageToRealData(
    Image3D&& input) {

    CuboidShape shape = input.getShape();
    RealData result = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceReal(shape);
    // padding == 0 for allocateMemoryOnDeviceReal, so memory is contiguous

    const float* src = input.getItkImage()->GetBufferPointer();
    float* dst = reinterpret_cast<float*>(result.getData());
    std::size_t bytes = static_cast<std::size_t>(shape.getVolume()) * sizeof(float);
    std::memcpy(dst, src, bytes);

    return result;
}

Image3D ImageConversion::convertRealDataToImage(
    RealData&& input) {

    assert(input.getRealSize() == input.getSize() && input.getPadding() ==0);
    CuboidShape shape = input.getSize();
    Image3D output(shape, 0.0f);

    const float* src = reinterpret_cast<const float*>(input.getData());
    float* dst = output.getItkImage()->GetBufferPointer();
    std::size_t bytes = static_cast<std::size_t>(shape.getVolume()) * sizeof(float);
    std::memcpy(dst, src, bytes);

    return output;
}

ComplexData ImageConversion::convertImageToComplexData(
    Image3D&& input) {

    CuboidShape shape = input.getShape();
    ComplexData result = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceComplexFull(shape);
    // padding == 0 for allocateMemoryOnDeviceComplexFull, so memory is contiguous

    // First zero the entire complex buffer (sets both real and imaginary parts to 0)
    std::memset(result.getData(), 0, result.getDataBytes());

    // Then copy the real parts from the image buffer using raw pointers
    const float* src = input.getItkImage()->GetBufferPointer();
    complex_t* dst = result.getData();
    std::size_t volume = static_cast<std::size_t>(shape.getVolume());
    for (std::size_t i = 0; i < volume; ++i) {
        dst[i][0] = static_cast<real_t>(src[i]);
        // dst[i][1] is already 0 from memset
    }

    return result;
}

// Padding function implementations have been moved to src/image/ImagePadding.cpp
// The ImageConversion namespace now delegates to ImagePadding namespace via inline functions in the header.


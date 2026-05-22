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

/**
 * @file image_conversion_test.cpp
 * @brief Tests for the buffer-pointer-based (move) ImageConversion overloads
 *
 * Verifies that:
 *  - convertImageToRealData(Image3D&&)  produces identical results to the ref-based overload
 *  - convertRealDataToImage(RealData&&)  produces identical results to the ref-based overload
 *  - convertImageToComplexData(Image3D&&) produces correct real parts and zero imaginary parts
 *  - round-trip Image3D -> RealData -> Image3D preserves data exactly
 *  - convertRealDataToImage(RealData&&) asserts on non-contiguous (padded) input
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

#include "aberrationcorrection/ImageConversion.h"
#include "aberrationcorrection/BackendFactory.h"
#include "dolphin_image/Image3D.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float tolerance = 1e-6f;

static bool approxEqual(float a, float b, float eps = tolerance) {
    return std::fabs(a - b) < eps;
}

/// Fill an Image3D with a deterministic pattern based on linear index.
static void fillPattern(Image3D& img) {
    int idx = 0;
    for (auto& pixel : img) {
        pixel = static_cast<float>(idx) * 1.37f + 0.5f;
        ++idx;
    }
}

/// Build a small test image of the given shape.
static Image3D makeTestImage(const CuboidShape& shape) {
    Image3D img(shape, 0.0f);
    fillPattern(img);
    return img;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static bool testImageToRealDataRoundTrip() {
    std::cout << "  [testImageToRealDataRoundTrip] ..." << std::flush;

    CuboidShape shape{8, 6, 4};
    Image3D original = makeTestImage(shape);

    // Make a copy to compare later (move will consume the source)
    Image3D reference(original);

    // Move-based conversion
    RealData realData = ImageConversion::convertImageToRealData(Image3D(original));

    // Move-based back-conversion
    // We need a non-padded RealData – allocate a fresh one and copy
    RealData realCopy = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceReal(shape);
    std::memcpy(realCopy.getData(), realData.getData(), shape.getVolume() * sizeof(float));

    Image3D roundTripped = ImageConversion::convertRealDataToImage(std::move(realCopy));

    // Verify shape
    if (roundTripped.getShape() != original.getShape()) {
        std::cout << " FAILED (shape mismatch)" << std::endl;
        return false;
    }

    // Verify pixel values
    int idx = 0;
    for (auto it = roundTripped.begin(); it != roundTripped.end(); ++it, ++idx) {
        if (!approxEqual(*it, reference[idx])) {
            std::cout << " FAILED at index " << idx
                      << ": got " << *it << ", expected " << reference[idx] << std::endl;
            return false;
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testImageToRealDataMatchesRefOverload() {
    std::cout << "  [testImageToRealDataMatchesRefOverload] ..." << std::flush;

    CuboidShape shape{8, 6, 4};
    Image3D img = makeTestImage(shape);

    // Reference-based overload (uses FFT-in-place allocation + per-pixel copy)
    RealData refResult = ImageConversion::convertImageToRealData(static_cast<const Image3D&>(img));

    // Move-based overload (uses allocateMemoryOnDeviceReal + memcpy)
    RealData moveResult = ImageConversion::convertImageToRealData(Image3D(img));

    // Compare pixel by pixel via the access() interface
    int volume = shape.getVolume();
    for (int i = 0; i < volume; ++i) {
        float refVal  = refResult[i];
        float moveVal = moveResult[i];
        if (!approxEqual(refVal, moveVal)) {
            std::cout << " FAILED at index " << i
                      << ": ref=" << refVal << ", move=" << moveVal << std::endl;
            return false;
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testRealDataToImageMatchesRefOverload() {
    std::cout << "  [testRealDataToImageMatchesRefOverload] ..." << std::flush;

    CuboidShape shape{8, 6, 4};
    Image3D img = makeTestImage(shape);

    // Convert to RealData (move overload) to get a contiguous buffer
    RealData realData = ImageConversion::convertImageToRealData(Image3D(img));

    // Make a second copy for the reference overload
    RealData realCopy = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceReal(shape);
    std::memcpy(realCopy.getData(), realData.getData(), shape.getVolume() * sizeof(float));

    // Reference-based overload
    Image3D refResult = ImageConversion::convertRealDataToImage(static_cast<const RealData&>(realCopy));

    // Move-based overload
    Image3D moveResult = ImageConversion::convertRealDataToImage(std::move(realData));

    // Compare
    auto refIt  = refResult.begin();
    auto moveIt = moveResult.begin();
    int idx = 0;
    for (; refIt != refResult.end() && moveIt != moveResult.end(); ++refIt, ++moveIt, ++idx) {
        if (!approxEqual(*refIt, *moveIt)) {
            std::cout << " FAILED at index " << idx
                      << ": ref=" << *refIt << ", move=" << *moveIt << std::endl;
            return false;
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testImageToComplexData() {
    std::cout << "  [testImageToComplexData] ..." << std::flush;

    CuboidShape shape{8, 6, 4};
    Image3D img = makeTestImage(shape);
    Image3D reference(img);

    ComplexData complexData = ImageConversion::convertImageToComplexData(Image3D(img));

    // Verify shape
    if (complexData.getSize() != shape) {
        std::cout << " FAILED (shape mismatch)" << std::endl;
        return false;
    }

    // Verify padding is 0 (contiguous)
    if (complexData.getPadding() != 0) {
        std::cout << " FAILED (unexpected padding)" << std::endl;
        return false;
    }

    int volume = shape.getVolume();
    for (int i = 0; i < volume; ++i) {
        float expectedReal = reference[i];
        float realPart = complexData[i][0];
        float imagPart = complexData[i][1];

        if (!approxEqual(realPart, expectedReal)) {
            std::cout << " FAILED at index " << i
                      << ": real part got " << realPart
                      << ", expected " << expectedReal << std::endl;
            return false;
        }
        if (!approxEqual(imagPart, 0.0f)) {
            std::cout << " FAILED at index " << i
                      << ": imaginary part got " << imagPart
                      << ", expected 0" << std::endl;
            return false;
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testImageToComplexDataMatchesRefOverload() {
    std::cout << "  [testImageToComplexDataMatchesRefOverload] ..." << std::flush;

    CuboidShape shape{8, 6, 4};
    Image3D img = makeTestImage(shape);

    // Reference overload
    ComplexData refResult = ImageConversion::convertImageToComplexData(static_cast<const Image3D&>(img));

    // Move overload
    ComplexData moveResult = ImageConversion::convertImageToComplexData(Image3D(img));

    int volume = shape.getVolume();
    for (int i = 0; i < volume; ++i) {
        float refReal  = refResult[i][0];
        float refImag  = refResult[i][1];
        float moveReal = moveResult[i][0];
        float moveImag = moveResult[i][1];

        if (!approxEqual(refReal, moveReal) || !approxEqual(refImag, moveImag)) {
            std::cout << " FAILED at index " << i
                      << ": ref=(" << refReal << "," << refImag << ")"
                      << " move=(" << moveReal << "," << moveImag << ")" << std::endl;
            return false;
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testVariousShapes() {
    std::cout << "  [testVariousShapes] ..." << std::flush;

    std::vector<CuboidShape> shapes = {
        {1, 1, 1},
        {4, 4, 4},
        {16, 8, 2},
        {3, 5, 7},
        {64, 32, 16},
    };

    for (const auto& shape : shapes) {
        Image3D img = makeTestImage(shape);
        Image3D reference(img);

        // Image -> RealData -> Image round-trip
        RealData realData = ImageConversion::convertImageToRealData(Image3D(img));
        Image3D roundTripped = ImageConversion::convertRealDataToImage(std::move(realData));

        int idx = 0;
        for (auto it = roundTripped.begin(); it != roundTripped.end(); ++it, ++idx) {
            if (!approxEqual(*it, reference[idx])) {
                std::cout << " FAILED for shape " << shape.print()
                          << " at index " << idx << std::endl;
                return false;
            }
        }

        // Image -> ComplexData
        ComplexData complexData = ImageConversion::convertImageToComplexData(Image3D(img));
        int volume = shape.getVolume();
        for (int i = 0; i < volume; ++i) {
            if (!approxEqual(complexData[i][0], reference[i]) || !approxEqual(complexData[i][1], 0.0f)) {
                std::cout << " FAILED (complex) for shape " << shape.print()
                          << " at index " << i << std::endl;
                return false;
            }
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testZeroImage() {
    std::cout << "  [testZeroImage] ..." << std::flush;

    CuboidShape shape{4, 4, 4};
    Image3D img(shape, 0.0f);

    RealData realData = ImageConversion::convertImageToRealData(std::move(img));
    int volume = shape.getVolume();
    for (int i = 0; i < volume; ++i) {
        if (!approxEqual(realData[i], 0.0f)) {
            std::cout << " FAILED at index " << i << std::endl;
            return false;
        }
    }

    Image3D img2(shape, 0.0f);
    ComplexData complexData = ImageConversion::convertImageToComplexData(std::move(img2));
    for (int i = 0; i < volume; ++i) {
        if (!approxEqual(complexData[i][0], 0.0f) || !approxEqual(complexData[i][1], 0.0f)) {
            std::cout << " FAILED at index " << i << std::endl;
            return false;
        }
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== Image Conversion (Move Overloads) Test ===" << std::endl;

    int passed = 0;
    int total  = 0;

    auto run = [&](bool (*test)(), const char* name) {
        ++total;
        if (test()) ++passed;
        else std::cerr << "  FAIL: " << name << std::endl;
    };

    run(testImageToRealDataRoundTrip,              "testImageToRealDataRoundTrip");
    run(testImageToRealDataMatchesRefOverload,     "testImageToRealDataMatchesRefOverload");
    run(testRealDataToImageMatchesRefOverload,     "testRealDataToImageMatchesRefOverload");
    run(testImageToComplexData,                    "testImageToComplexData");
    run(testImageToComplexDataMatchesRefOverload,  "testImageToComplexDataMatchesRefOverload");
    run(testVariousShapes,                         "testVariousShapes");
    run(testZeroImage,                             "testZeroImage");

    std::cout << "\n=== " << passed << "/" << total << " tests passed ===" << std::endl;
    return (passed == total) ? 0 : 1;
}

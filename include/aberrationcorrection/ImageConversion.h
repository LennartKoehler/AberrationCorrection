#pragma once
#include "dolphinbackend/ComplexData.h"
#include "dolphin_image/Image3D.h"

namespace ImageConversion{
    ComplexData convertImageToComplexData(const Image3D& image);
    Image3D convertComplexDataToImage(const ComplexData& data);

    RealData convertImageToRealData(const Image3D& image);
    Image3D convertRealDataToImage(const RealData& data);

    // Fast conversions using buffer pointer memcpy (no per-pixel iteration)
    // These require contiguous memory (no padding) and matching element sizes.
    RealData convertImageToRealData(Image3D&& image);
    Image3D convertRealDataToImage(RealData&& data);
    ComplexData convertImageToComplexData(Image3D&& image);
}

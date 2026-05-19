#pragma once
#include "dolphinbackend/ComplexData.h"
#include "dolphin_image/Image3D.h"

namespace ImageConversion{
    ComplexData convertImageToComplexData(const Image3D& image);
    Image3D convertComplexDataToImage(const ComplexData& data);

    RealData convertImageToRealData(const Image3D& image);
    Image3D convertRealDataToImage(const RealData& data);
}

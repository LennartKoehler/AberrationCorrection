#pragma once

#include "dolphin_image/Image3D.h"
#include <optional>

struct SphereDetection {
    float centerX;
    float centerY;
    float centerZ;
    float radius;
    int voxelCount;
};

namespace SphereDetector {
    std::optional<SphereDetection> detectSphere(const Image3D& image);
}

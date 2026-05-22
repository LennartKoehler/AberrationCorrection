
#include "aberrationcorrection/SphereDetector.h"
#include "dolphin_image/ImageOperations.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

namespace SphereDetector {

static constexpr int HISTOGRAM_BINS = 256;
static constexpr double MIN_SPHERE_FRACTION = 0.001;
static constexpr double MAX_SPHERE_FRACTION = 0.8;
static constexpr double MIN_SEPARABILITY = 0.3;

struct OtsuResult {
    int threshold;
    double interClassVariance;
};

static OtsuResult computeOtsuThreshold(const std::vector<int>& histogram, int totalVoxels) {
    double sumAll = 0.0;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        sumAll += static_cast<double>(i) * histogram[i];
    }

    double sumBackground = 0.0;
    int weightBackground = 0;
    double maxVariance = -1.0;
    int bestThreshold = 0;

    for (int t = 0; t < HISTOGRAM_BINS - 1; ++t) {
        weightBackground += histogram[t];
        if (weightBackground == 0) continue;

        int weightForeground = totalVoxels - weightBackground;
        if (weightForeground == 0) break;

        sumBackground += static_cast<double>(t) * histogram[t];

        double meanBackground = sumBackground / weightBackground;
        double meanForeground = (sumAll - sumBackground) / weightForeground;

        double variance = static_cast<double>(weightBackground) *
                          static_cast<double>(weightForeground) *
                          (meanBackground - meanForeground) *
                          (meanBackground - meanForeground);

        if (variance > maxVariance) {
            maxVariance = variance;
            bestThreshold = t;
        }
    }

    return {bestThreshold, maxVariance};
}

std::optional<SphereDetection> detectSphere(const Image3D& input) {
    Image3D image(input);
    ImageOperations::normalizeChannel(image);

    CuboidShape shape = image.getShape();
    int totalVoxels = shape.getVolume();

    std::vector<int> histogram(HISTOGRAM_BINS, 0);
    for (auto it = image.cbegin(); it != image.cend(); ++it) {
        int bin = static_cast<int>(*it * (HISTOGRAM_BINS - 1));
        bin = std::clamp(bin, 0, HISTOGRAM_BINS - 1);
        histogram[bin]++;
    }

    OtsuResult otsu = computeOtsuThreshold(histogram, totalVoxels);
    float thresholdValue = static_cast<float>(otsu.threshold) / (HISTOGRAM_BINS - 1);

    double histogramTotalVariance = 0.0;
    double histogramMean = 0.0;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        histogramMean += static_cast<double>(i) * histogram[i];
    }
    histogramMean /= totalVoxels;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        double diff = static_cast<double>(i) - histogramMean;
        histogramTotalVariance += diff * diff * histogram[i];
    }
    histogramTotalVariance /= totalVoxels;

    if (histogramTotalVariance > 0.0 &&
        otsu.interClassVariance / (totalVoxels * totalVoxels * histogramTotalVariance) < MIN_SEPARABILITY) {
        return std::nullopt;
    }

    int countAbove = 0;
    int countBelow = 0;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        if (i <= otsu.threshold) {
            countBelow += histogram[i];
        } else {
            countAbove += histogram[i];
        }
    }

    bool sphereIsAbove = (countAbove <= countBelow);
    int sphereVoxelCount = sphereIsAbove ? countAbove : countBelow;

    double minVoxels = totalVoxels * MIN_SPHERE_FRACTION;
    double maxVoxels = totalVoxels * MAX_SPHERE_FRACTION;
    if (sphereVoxelCount < minVoxels || sphereVoxelCount > maxVoxels) {
        return std::nullopt;
    }

    double sphereSum = 0.0;
    int sphereCount = 0;
    double bgSum = 0.0;
    int bgCount = 0;
    for (auto it = image.cbegin(); it != image.cend(); ++it) {
        bool isAbove = (*it > thresholdValue);
        bool isSphereClass = (isAbove == sphereIsAbove);
        if (isSphereClass) {
            sphereSum += *it;
            sphereCount++;
        } else {
            bgSum += *it;
            bgCount++;
        }
    }

    double sphereMean = (sphereCount > 0) ? sphereSum / sphereCount : 0.0;
    double bgMean = (bgCount > 0) ? bgSum / bgCount : 0.0;

    double sumX = 0.0, sumY = 0.0, sumZ = 0.0, sumWeight = 0.0;
    int countedVoxels = 0;

    for (auto it = image.cbegin(); it != image.cend(); ++it) {
        bool isAbove = (*it > thresholdValue);
        bool isSphereClass = (isAbove == sphereIsAbove);
        if (!isSphereClass) continue;

        float weight = std::abs(*it - static_cast<float>(bgMean));
        if (weight < 1e-8f) continue;

        int x, y, z;
        it.getCoordinates(x, y, z);

        sumX += x * weight;
        sumY += y * weight;
        sumZ += z * weight;
        sumWeight += weight;
        countedVoxels++;
    }

    if (sumWeight < 1e-12 || countedVoxels == 0) {
        return std::nullopt;
    }

    float cx = static_cast<float>(sumX / sumWeight);
    float cy = static_cast<float>(sumY / sumWeight);
    float cz = static_cast<float>(sumZ / sumWeight);

    double volume = static_cast<double>(countedVoxels);
    float radius = static_cast<float>(std::cbrt(3.0 * volume / (4.0 * M_PI)));

    return SphereDetection{cx, cy, cz, radius, countedVoxels};
}

}

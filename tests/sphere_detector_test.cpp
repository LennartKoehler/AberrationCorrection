#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>

#include "dolphin_image/Image3D.h"
#include "aberrationcorrection/SphereDetector.h"

static float tolerance_center = 2.0f;
static float tolerance_radius = 2.0f;

static Image3D createHardSphere(
    int nx, int ny, int nz,
    float cx, float cy, float cz, float radius,
    float fgValue, float bgValue)
{
    Image3D img(CuboidShape{nx, ny, nz}, bgValue);
    float rSq = radius * radius;

    for (auto it = img.begin(); it != img.end(); ++it) {
        int x, y, z;
        it.getCoordinates(x, y, z);
        float dx = x - cx;
        float dy = y - cy;
        float dz = z - cz;
        if (dx * dx + dy * dy + dz * dz <= rSq) {
            *it = fgValue;
        }
    }
    return img;
}

static Image3D createBlurrySphere(
    int nx, int ny, int nz,
    float cx, float cy, float cz, float radius, float sigma,
    float fgValue, float bgValue)
{
    Image3D img(CuboidShape{nx, ny, nz}, bgValue);

    for (auto it = img.begin(); it != img.end(); ++it) {
        int x, y, z;
        it.getCoordinates(x, y, z);
        float dx = x - cx;
        float dy = y - cy;
        float dz = z - cz;
        float r = std::sqrt(dx * dx + dy * dy + dz * dz);
        float t = 0.5f * (1.0f - std::tanh((r - radius) / sigma));
        *it = bgValue + (fgValue - bgValue) * t;
    }
    return img;
}

static void addGaussianNoise(Image3D& img, float mean, float stddev, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (auto it = img.begin(); it != img.end(); ++it) {
        *it += dist(rng);
    }
}

static bool approxEqual(float a, float b, float eps) {
    return std::fabs(a - b) < eps;
}

static bool testHardSphereBrightOnDark() {
    std::cout << "  [testHardSphereBrightOnDark] ..." << std::flush;

    int nx = 64, ny = 64, nz = 64;
    float cx = 30.0f, cy = 32.0f, cz = 34.0f;
    float radius = 15.0f;

    Image3D img = createHardSphere(nx, ny, nz, cx, cy, cz, radius, 1000.0f, 10.0f);
    addGaussianNoise(img, 0.0f, 30.0f);

    auto result = SphereDetector::detectSphere(img);
    if (!result) {
        std::cout << " FAILED (no sphere detected)" << std::endl;
        return false;
    }

    if (!approxEqual(result->centerX, cx, tolerance_center) ||
        !approxEqual(result->centerY, cy, tolerance_center) ||
        !approxEqual(result->centerZ, cz, tolerance_center)) {
        std::cout << " FAILED (center: got ("
                  << result->centerX << "," << result->centerY << "," << result->centerZ
                  << "), expected (" << cx << "," << cy << "," << cz << "))" << std::endl;
        return false;
    }

    if (!approxEqual(result->radius, radius, tolerance_radius)) {
        std::cout << " FAILED (radius: got " << result->radius
                  << ", expected " << radius << ")" << std::endl;
        return false;
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testHardSphereDarkOnBright() {
    std::cout << "  [testHardSphereDarkOnBright] ..." << std::flush;

    int nx = 64, ny = 64, nz = 64;
    float cx = 30.0f, cy = 32.0f, cz = 34.0f;
    float radius = 15.0f;

    Image3D img = createHardSphere(nx, ny, nz, cx, cy, cz, radius, 10.0f, 1000.0f);
    addGaussianNoise(img, 0.0f, 30.0f);

    auto result = SphereDetector::detectSphere(img);
    if (!result) {
        std::cout << " FAILED (no sphere detected)" << std::endl;
        return false;
    }

    if (!approxEqual(result->centerX, cx, tolerance_center) ||
        !approxEqual(result->centerY, cy, tolerance_center) ||
        !approxEqual(result->centerZ, cz, tolerance_center)) {
        std::cout << " FAILED (center: got ("
                  << result->centerX << "," << result->centerY << "," << result->centerZ
                  << "), expected (" << cx << "," << cy << "," << cz << "))" << std::endl;
        return false;
    }

    if (!approxEqual(result->radius, radius, tolerance_radius)) {
        std::cout << " FAILED (radius: got " << result->radius
                  << ", expected " << radius << ")" << std::endl;
        return false;
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testBlurrySphere() {
    std::cout << "  [testBlurrySphere] ..." << std::flush;

    int nx = 64, ny = 64, nz = 64;
    float cx = 32.0f, cy = 32.0f, cz = 32.0f;
    float radius = 15.0f;
    float sigma = 2.0f;

    Image3D img = createBlurrySphere(nx, ny, nz, cx, cy, cz, radius, sigma, 1000.0f, 10.0f);
    addGaussianNoise(img, 0.0f, 20.0f);

    auto result = SphereDetector::detectSphere(img);
    if (!result) {
        std::cout << " FAILED (no sphere detected)" << std::endl;
        return false;
    }

    if (!approxEqual(result->centerX, cx, tolerance_center) ||
        !approxEqual(result->centerY, cy, tolerance_center) ||
        !approxEqual(result->centerZ, cz, tolerance_center)) {
        std::cout << " FAILED (center: got ("
                  << result->centerX << "," << result->centerY << "," << result->centerZ
                  << "), expected (" << cx << "," << cy << "," << cz << "))" << std::endl;
        return false;
    }

    float blurry_tolerance = 3.0f;
    if (!approxEqual(result->radius, radius, blurry_tolerance)) {
        std::cout << " FAILED (radius: got " << result->radius
                  << ", expected ~" << radius << " (tolerance " << blurry_tolerance << "))" << std::endl;
        return false;
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testSmallSphere() {
    std::cout << "  [testSmallSphere] ..." << std::flush;

    int nx = 32, ny = 32, nz = 32;
    float cx = 16.0f, cy = 16.0f, cz = 16.0f;
    float radius = 5.0f;

    Image3D img = createHardSphere(nx, ny, nz, cx, cy, cz, radius, 500.0f, 5.0f);
    addGaussianNoise(img, 0.0f, 15.0f);

    auto result = SphereDetector::detectSphere(img);
    if (!result) {
        std::cout << " FAILED (no sphere detected)" << std::endl;
        return false;
    }

    if (!approxEqual(result->centerX, cx, tolerance_center) ||
        !approxEqual(result->centerY, cy, tolerance_center) ||
        !approxEqual(result->centerZ, cz, tolerance_center)) {
        std::cout << " FAILED (center: got ("
                  << result->centerX << "," << result->centerY << "," << result->centerZ
                  << "), expected (" << cx << "," << cy << "," << cz << "))" << std::endl;
        return false;
    }

    float small_tolerance = 2.5f;
    if (!approxEqual(result->radius, radius, small_tolerance)) {
        std::cout << " FAILED (radius: got " << result->radius
                  << ", expected " << radius << ")" << std::endl;
        return false;
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

static bool testNoSphere() {
    std::cout << "  [testNoSphere] ..." << std::flush;

    Image3D img(CuboidShape{32, 32, 32}, 100.0f);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (auto it = img.begin(); it != img.end(); ++it) {
        *it += dist(rng);
    }

    auto result = SphereDetector::detectSphere(img);
    if (result.has_value()) {
        std::cout << " FAILED (detected sphere in uniform noise)" << std::endl;
        return false;
    }

    std::cout << " PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Sphere Detector Test ===" << std::endl;

    int passed = 0;
    int total  = 0;

    auto run = [&](bool (*test)(), const char* name) {
        ++total;
        if (test()) ++passed;
        else std::cerr << "  FAIL: " << name << std::endl;
    };

    run(testHardSphereBrightOnDark, "testHardSphereBrightOnDark");
    run(testHardSphereDarkOnBright, "testHardSphereDarkOnBright");
    run(testBlurrySphere,           "testBlurrySphere");
    run(testSmallSphere,            "testSmallSphere");
    run(testNoSphere,               "testNoSphere");

    std::cout << "\n=== " << passed << "/" << total << " tests passed ===" << std::endl;
    return (passed == total) ? 0 : 1;
}

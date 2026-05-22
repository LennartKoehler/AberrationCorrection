#pragma once

struct ZernikeCoefficients {
    static constexpr int Size = 5;

    bool anyLarger(const ZernikeCoefficients& other);
    ZernikeCoefficients operator-(float value);
    ZernikeCoefficients operator+(float value);
    ZernikeCoefficients operator-(const ZernikeCoefficients& other);
    ZernikeCoefficients operator+(const ZernikeCoefficients& other);
    void clamp(const ZernikeCoefficients& other);
    void floor(const ZernikeCoefficients& other);
    float c[Size];
};

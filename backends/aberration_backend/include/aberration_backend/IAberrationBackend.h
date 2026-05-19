#pragma once
#include "dolphinbackend/ComplexData.h"
#include "aberration_backend/ZernikeCoefficients.h"


class IAberrationBackend{
public:
    virtual ~IAberrationBackend() = default;
    virtual void computeAberration(const ComplexData& data, ZernikeCoefficients coeff) const = 0;
    virtual void zernikePhaseTestFunction(const ComplexData& output, ZernikeCoefficients coeff) const = 0;

};

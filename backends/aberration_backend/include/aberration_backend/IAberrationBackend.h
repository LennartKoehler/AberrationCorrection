#pragma once
#include "dolphinbackend/IComputeBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"


class IAberrationBackend : public virtual IComputeBackend{
public:
    virtual ~IAberrationBackend() = default;
    virtual void subtractZernikePhase(ComplexData& data, ZernikeCoefficients coeff) const = 0;
    virtual void addZernikePhase(ComplexData& data, ZernikeCoefficients coeff) const = 0;

    virtual void zernikePhaseTestFunction(ComplexData& output, ZernikeCoefficients coeff) const = 0;

};



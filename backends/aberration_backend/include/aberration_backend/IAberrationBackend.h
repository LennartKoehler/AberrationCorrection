#pragma once
#include "dolphinbackend/ComplexData.h"

using ZernikeCoefficients = float[4];


class IAberrationBackend{
public:
    virtual ~IAberrationBackend() = default;
    virtual void computeAberration(const ComplexData& data) const = 0;

};

#pragma once

#include "aberration_backend/IAberrationBackend.h"

class IErrorFunction{
public:

    virtual float getError(ComplexData&& current, const IAberrationBackend& backend) = 0;

};

class BasicErrorFunction : public IErrorFunction{
public:

    BasicErrorFunction(const ComplexData& target):target(target){}
    virtual float getError(ComplexData&& current, const IAberrationBackend& backend) override{
        real_t error = 0;
        backend.meanSquareError(current, target, &error);
        return error;
    }

    ComplexData target;

};

class RealSpaceErrorFunction : public IErrorFunction{
public:

    RealSpaceErrorFunction(const RealData& target, RealData& scratch):target(target), scratch(scratch){}

    virtual float getError(ComplexData&& current, const IAberrationBackend& backend) override{
        backend.backwardFFT(current, scratch);



        return 0.1;
    }

    RealData scratch;
    RealData target;

};

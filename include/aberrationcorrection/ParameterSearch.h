#pragma once

#include "ErrorFunction.h"
#include "aberration_backend/ZernikeCoefficients.h"

class ComputeContext;
class ISearchAlgorithm;
class IErrorFunction;
class ZernikeCoefficients;

class ParameterSearch{
public:
    ParameterSearch() = default;
    virtual ~ParameterSearch() = default;

    std::pair<ZernikeCoefficients, ComplexData> search(
        const ComplexData& input,
        ISearchAlgorithm& searchAlgorithm,
        IErrorFunction& errorFn,
        ComputeContext context);
private:

    bool searchFinished();
    bool searchFinished_ = false;
    size_t iterationCounter;

};

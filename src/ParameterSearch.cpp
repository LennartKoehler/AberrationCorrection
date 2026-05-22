#include "aberration_backend/IAberrationBackend.h"
#include "aberrationcorrection/ParameterSearch.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "aberrationcorrection/ErrorFunction.h"
#include "aberrationcorrection/ComputeContext.h"
#include "aberrationcorrection/ErrorFunction.h"
#include "aberrationcorrection/SearchAlgorithm.h"
#include <iostream>

std::pair<ZernikeCoefficients, ComplexData> ParameterSearch::search(
    const ComplexData& input,
    ISearchAlgorithm& searchAlgorithm,
    IErrorFunction& errorFn,
    ComputeContext context)
{
    float error;
    iterationCounter = 0;
    ZernikeCoefficients currentParameters = searchAlgorithm.getInitial();
    ComplexData currentData = context.memoryManager->allocateMemoryOnDeviceComplexFull(input.getSize());

    while (!searchFinished()){
        context.memoryManager->memCopy(input, currentData); // TODO because the subtract zernikePhase is inplace, not optimized

        context.computeBackend->subtractZernikePhase(currentData, currentParameters);

        error = errorFn.getError(std::move(currentData), *context.computeBackend);

        std::pair<bool, ZernikeCoefficients> update = searchAlgorithm.nextGuess(currentParameters, error);
        searchFinished_ = update.first;
        currentParameters = update.second;

        iterationCounter ++;
        std::cout << "Iteration: " << iterationCounter  << "\tError: "<< error <<std::endl;

    }
    return std::pair<ZernikeCoefficients, ComplexData>(currentParameters, std::move(currentData));

}

bool ParameterSearch::searchFinished(){
    return searchFinished_;
}


#pragma once

#include "aberration_backend/ZernikeCoefficients.h"
#include <utility>
#include <cassert>


class ISearchAlgorithm{
public:
    ISearchAlgorithm(
        const ZernikeCoefficients& initial,
		const ZernikeCoefficients& min,
		const ZernikeCoefficients& max)
    : initial(initial),
    min(min),
    max(max),
    bestParameters(initial),
    smallestError(999)
    {}

    virtual std::pair<bool, ZernikeCoefficients> nextGuess(ZernikeCoefficients previousGuess, float error) = 0;
    const ZernikeCoefficients& getInitial(){return initial;}
    ZernikeCoefficients getBestParameteters(){return bestParameters;}

protected:

    void updateBest(ZernikeCoefficients previousGuess, float error){
        if (error < smallestError){
            smallestError = error;
            bestParameters = previousGuess;
        }
    }
    ZernikeCoefficients initial;
	ZernikeCoefficients min;
	ZernikeCoefficients max;
    ZernikeCoefficients bestParameters;
    float smallestError;
};

class NaiveSearchAlgorithm : public ISearchAlgorithm{
public:

    NaiveSearchAlgorithm(const ZernikeCoefficients& initial, const ZernikeCoefficients& min, const ZernikeCoefficients& max)
    : ISearchAlgorithm(initial, min, max){}

    void init(float stepSize){ this->stepSize = stepSize;}

    std::pair<bool, ZernikeCoefficients> nextGuess(ZernikeCoefficients previousGuess, float error) override{

        updateBest(previousGuess, error);

        ZernikeCoefficients newGuess = previousGuess;

        bool done = false;
        // Odometer-style carry: increment c[0], and carry into higher indices
        for (int i = 0; i < ZernikeCoefficients::Size; i++){
            newGuess.c[i] += stepSize;
            if (newGuess.c[i] <= max.c[i]){
                break; // no carry needed, done
            }
            // Overflow: reset this digit to min
            newGuess.c[i] = min.c[i];
            // Loop continues to carry into c[i+1]
            if (i >= ZernikeCoefficients::Size - 1){
                done = true; //if at last index and carry is needed, this would wrap around, so were done
            }
        }

        return std::pair<bool, ZernikeCoefficients>(done, newGuess);
    }

private:

    float stepSize = 0.1;
};


// TODO make the stepSize per parameter and more dynamic
class GridSearchAlgorithm : public ISearchAlgorithm{
public:

    GridSearchAlgorithm(const ZernikeCoefficients& initial, const ZernikeCoefficients& min, const ZernikeCoefficients& max)
    : ISearchAlgorithm(initial, min, max), currentGridDepth(0), currentMin(min), currentMax(max){}

    // if stepReductionSize > 1 then the deeper it goes the more extensive it searched, if < 1 then the deeper the less extensive
    // if stepReductionSize = 1 then each depth should have appproximately the same number of iterations
    void init(float stepSize, float stepReductionSize, int gridDepth){
        assert(stepSize < 0.5 * ((max - min).c[0]));
        this->stepSize = stepSize;
        this->initialStepSize = stepSize;
        this->stepReductionSize = stepReductionSize;
        this->maxGridDepth = gridDepth;}

    std::pair<bool, ZernikeCoefficients> nextGuess(ZernikeCoefficients previousGuess, float error) override{

        updateBest(previousGuess, error);

        ZernikeCoefficients newGuess = previousGuess;
        bool done = false;

        // Odometer-style carry: increment c[0], and carry into higher indices
        for (int i = 0; i < ZernikeCoefficients::Size; i++){
            newGuess.c[i] += stepSize;
            if (newGuess.c[i] <= currentMax.c[i]){
                break; // no carry needed, done
            }
            // Overflow: reset this digit to min
            newGuess.c[i] = currentMin.c[i];
            // Loop continues to carry into c[i+1]
            if (i >= ZernikeCoefficients::Size - 1){
                if (currentGridDepth >= maxGridDepth){
                    done = true; //if at last index and carry is needed, and its at max grid depth
                }
                else {
                    newGuess = enterNewGridDepth();
                }

            }
        }

        return std::pair<bool, ZernikeCoefficients>(done, newGuess);
    }

private:
    ZernikeCoefficients enterNewGridDepth(){
        currentGridDepth++;
        ZernikeCoefficients tempMin = bestParameters - stepSize;
        ZernikeCoefficients tempMax = bestParameters + stepSize;
        tempMin.floor(min); // these can be lower than the previous min, but never lower than "global min"
        tempMax.clamp(max);
        currentMin = tempMin;
        currentMax = tempMax;
        stepSize = 2 * stepSize * stepReductionSize * initialStepSize;
        return currentMin;
    }

    int maxGridDepth;
    int currentGridDepth;
    float stepReductionSize;
    float stepSize;
    float initialStepSize;
	ZernikeCoefficients currentMax; // for current depth, the ISearchAlgorithm min and max are global
	ZernikeCoefficients currentMin;
};

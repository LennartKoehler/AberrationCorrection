#pragma once

#include "kernels.cuh"

namespace ABERR {

    cudaError_t applyPhaseCorrection(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ, cudaStream_t stream = 0);

    cudaError_t subtractZernikePhase(int dataNx, int dataNy, int dataNz, int realNx, int realNy, int realNz, complex_t* data, ZernikeCoefficients coeff, cudaStream_t stream = 0);
    cudaError_t addZernikePhase(int dataNx, int dataNy, int dataNz, int realNx, int realNy, int realNz, complex_t* data, ZernikeCoefficients coeff, cudaStream_t stream = 0);

    namespace TEST{

        cudaError_t zernikePhaseTest(int dataNx, int dataNy, int dataNz, int realNx, int realNy, int realNz, complex_t* data, ZernikeCoefficients coeff, cudaStream_t stream = 0);
    }

}



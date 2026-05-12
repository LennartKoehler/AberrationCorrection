#pragma once

#include "kernels.cuh"

namespace ABERR {

    cudaError_t applyPhaseCorrection(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ, cudaStream_t stream = 0);

}

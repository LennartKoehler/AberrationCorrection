#pragma once

#include <cuda_runtime_api.h>
#include "aberration_backend/ZernikeCoefficients.h"

#ifdef DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

typedef real_t complex_t[2];

__global__ void applyPhaseCorrectionGlobal(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ);

__global__ void applyZernikePolynomialsGlobal(int Nx, int Ny, int Nz, complex_t* data, ZernikeCoefficients coeff);

__global__ void getPhaseCorrectionGlobal(real_t* deltaPhi, int index, int x, int y, int z, int Nx, int Ny, int Nz, ZernikeCoefficients coeff);

__global__ void zernikePhaseTestGlobal(int Nx, int Ny, int Nz, complex_t* data, ZernikeCoefficients coeff);

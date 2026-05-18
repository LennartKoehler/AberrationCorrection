#pragma once

#include <cuda_runtime_api.h>

#ifdef DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

typedef real_t complex_t[2];

struct ZernikeCoefficients {
    real_t c[4]; // Z0: Piston, Z1: Tilt X, Z2: Tilt Y, Z3: Defocus
};

__global__ void applyPhaseCorrectionGlobal(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ);

__global__ void applyZernikePolynomialsGlobal(int Nx, int Ny, int Nz, complex_t* data, ZernikeCoefficients coeff);

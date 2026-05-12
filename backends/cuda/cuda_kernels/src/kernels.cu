#include "kernels.cuh"

__global__
void applyPhaseCorrectionGlobal(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        real_t phase = phaseX[x] + phaseY[y] + phaseZ[z];
        real_t cosPhase = cos(phase);
        real_t sinPhase = sin(phase);

        real_t re = data[index][0];
        real_t im = data[index][1];

        data[index][0] = re * cosPhase - im * sinPhase;
        data[index][1] = re * sinPhase + im * cosPhase;
    }
}

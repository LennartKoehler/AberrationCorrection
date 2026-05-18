#include "operations.cuh"
#include "kernels.cuh"

const dim3 GLOBAL_THREADS_PER_BLOCK(4, 8, 8);

inline dim3 computeBlocksPerGrid(int Nx, int Ny, int Nz) {
    return dim3(
        (Nx + GLOBAL_THREADS_PER_BLOCK.x - 1) / GLOBAL_THREADS_PER_BLOCK.x,
        (Ny + GLOBAL_THREADS_PER_BLOCK.y - 1) / GLOBAL_THREADS_PER_BLOCK.y,
        (Nz + GLOBAL_THREADS_PER_BLOCK.z - 1) / GLOBAL_THREADS_PER_BLOCK.z
    );
}

#define CUDA_CHECK_KERNEL(kernel_launch, stream) \
    do { \
        cudaEvent_t _event; \
        cudaError_t _err = cudaEventCreate(&_event); \
        if (_err != cudaSuccess) { \
            return _err; \
        } \
        kernel_launch; \
        _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
            cudaEventDestroy(_event); \
            return _err; \
        } \
        _err = cudaEventRecord(_event, stream); \
        if (_err != cudaSuccess) { \
            cudaEventDestroy(_event); \
            return _err; \
        } \
        _err = cudaEventSynchronize(_event); \
        if (_err != cudaSuccess) { \
            cudaEventDestroy(_event); \
            return _err; \
        } \
        cudaEventDestroy(_event); \
    } while(0)

namespace ABERR {

    cudaError_t applyPhaseCorrection(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ, cudaStream_t stream) {
        if (!data || !phaseX || !phaseY || !phaseZ) {
            return cudaErrorInvalidValue;
        }

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
            (applyPhaseCorrectionGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, data, phaseX, phaseY, phaseZ)),
            stream);
        return cudaSuccess;
    }

    cudaError_t applyZernikeCorrection(int Nx, int Ny, int Nz, complex_t* data, ZernikeCoefficients coeff, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
            (applyZernikePolynomialsGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, data, coeff)),
            stream);
        return cudaSuccess;
    }

}

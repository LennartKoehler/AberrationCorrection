#include "cuda/CUDAAberrationBackend.h"
#include "aberration_backend/IAberrationBackend.h"
#include "cuda_kernels/operations.cuh"

void CUDAAberrationBackend::computeAberration(const ComplexData& image) const{

    CuboidShape dataSize = image.getSize();
    int Nx = dataSize.width;
    int Ny = dataSize.height;
    int Nz = dataSize.depth;
    complex_t* data = data.getData();
    ZernikeCoefficients coeff = {1,2,3,4};
    cudaStream_t stream = 0;
    ABERR::applyZernikeCorrection(Nx, Ny, Nz, data, coeff, stream);

}

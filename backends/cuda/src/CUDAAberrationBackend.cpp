#include "cuda/CUDAAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "operations.cuh"
#include <stdexcept>

void CUDAAberrationBackend::computeAberration(const ComplexData& image, ZernikeCoefficients coefficients) const{

    CuboidShape dataSize = image.getSize();
    int Nx = dataSize.width;
    int Ny = dataSize.height;
    int Nz = dataSize.depth;
    complex_t* data = image.getData();
    cudaStream_t stream = 0;
    ABERR::applyZernikeCorrection(Nx, Ny, Nz, data, coefficients, stream);

}

void CUDAAberrationBackend::zernikePhaseTestFunction(const ComplexData& output, ZernikeCoefficients coeff) const {

    CuboidShape dataSize = output.getSize();
    int Nx = dataSize.width;
    int Ny = dataSize.height;
    int Nz = dataSize.depth;
    complex_t* data = output.getData();
    cudaStream_t stream = 0;
    ABERR::TEST::zernikePhaseTest(Nx, Ny, Nz, data, coeff, stream);
}

std::unique_ptr<CUDAComputeBackend> CUDAAberrationBackendManager::createComputeBackend(CUDABackendConfig config) {
    return std::move(std::make_unique<CUDAAberrationBackend>(config));
}

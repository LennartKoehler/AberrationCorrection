#include "cuda/CUDAAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "operations.cuh"
#include <stdexcept>

void CUDAAberrationBackend::addZernikePhase(ComplexData& image, ZernikeCoefficients coefficients) const{
    CuboidShape dataSize = image.getSize();
    CuboidShape realSize = image.getRealSize();
    complex_t* data = image.getData();
    cudaStream_t stream = 0;
    ABERR::addZernikePhase(dataSize.width, dataSize.height, dataSize.depth, realSize.width, realSize.height, realSize.depth, data, coefficients, stream);
}

void CUDAAberrationBackend::subtractZernikePhase(ComplexData& image, ZernikeCoefficients coefficients) const{
    CuboidShape dataSize = image.getSize();
    CuboidShape realSize = image.getRealSize();
    complex_t* data = image.getData();
    cudaStream_t stream = 0;
    ABERR::subtractZernikePhase(dataSize.width, dataSize.height, dataSize.depth, realSize.width, realSize.height, realSize.depth, data, coefficients, stream);
}

void CUDAAberrationBackend::zernikePhaseTestFunction(ComplexData& output, ZernikeCoefficients coeff) const {
    CuboidShape dataSize = output.getSize();
    CuboidShape realSize = output.getRealSize();
    complex_t* data = output.getData();
    cudaStream_t stream = 0;
    ABERR::TEST::zernikePhaseTest(dataSize.width, dataSize.height, dataSize.depth, realSize.width, realSize.height, realSize.depth, data, coeff, stream);
}

std::unique_ptr<CUDAComputeBackend> CUDAAberrationBackendManager::createComputeBackend(CUDABackendConfig config) {
    return std::move(std::make_unique<CUDAAberrationBackend>(config));
}

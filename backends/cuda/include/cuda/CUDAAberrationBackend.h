#pragma once

#include "aberration_backend/ZernikeCoefficients.h"
#include "cuda_backend/CUDABackend.h"
#include "cuda_backend/CUDABackendManager.h"
#include "aberration_backend/IAberrationBackend.h"

class CUDAAberrationBackend : public CUDAComputeBackend, public virtual IAberrationBackend{
public:
    CUDAAberrationBackend(CUDABackendConfig config) : CUDAComputeBackend(config){}
    ~CUDAAberrationBackend() override = default;

    void subtractZernikePhase(ComplexData& data, ZernikeCoefficients coeff) const override;
    void addZernikePhase(ComplexData& data, ZernikeCoefficients coeff) const override;
    void zernikePhaseTestFunction(ComplexData& output, ZernikeCoefficients coeff) const override;
};

class CUDAAberrationBackendManager : public CUDABackendManager {
public:
    CUDAAberrationBackendManager() = default;
    ~CUDAAberrationBackendManager() override = default;


    std::unique_ptr<CUDAComputeBackend> createComputeBackend(CUDABackendConfig config) override;
};

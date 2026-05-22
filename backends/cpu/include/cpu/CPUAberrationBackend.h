#pragma once

#include "aberration_backend/ZernikeCoefficients.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"
#include "aberration_backend/IAberrationBackend.h"


class CPUAberrationBackend : public IAberrationBackend ,public CPUComputeBackend{
public:

    CPUAberrationBackend(CPUBackendConfig config, FFTWManager& manager) : CPUComputeBackend(config, manager){}
    ~CPUAberrationBackend() override = default;

    void subtractZernikePhase(ComplexData& data, ZernikeCoefficients coeff) const override;
    void addZernikePhase(ComplexData& data, ZernikeCoefficients coeff) const override;

    void zernikePhaseTestFunction(ComplexData& output, ZernikeCoefficients coeff) const override;

private:

    void computeZernikePhase(real_t* correction, int x, int y, int Nx, int Ny, ZernikeCoefficients coeff) const ;
    void subtractZernikePhase_(CuboidShape dataSize, CuboidShape realSize, complex_t* data, const ZernikeCoefficients& coeff) const;
    void addZernikePhase_(CuboidShape dataSize, CuboidShape realSize, complex_t* data, const ZernikeCoefficients& coeff) const;

};

class CPUAberrationBackendManager : public CPUBackendManager {
public:
    std::unique_ptr<CPUComputeBackend> createComputeBackend(CPUBackendConfig config) override;
};


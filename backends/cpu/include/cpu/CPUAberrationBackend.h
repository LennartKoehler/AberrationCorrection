#pragma once

#include "aberration_backend/ZernikeCoefficients.h"
#include "cpu_backend/CPUBackend.h"
#include "cpu_backend/CPUBackendManager.h"
#include "aberration_backend/IAberrationBackend.h"


class CPUAberrationBackend : public CPUComputeBackend, public IAberrationBackend {
public:

    CPUAberrationBackend(CPUBackendConfig config, FFTWManager& manager) : CPUComputeBackend(config, manager){}
    ~CPUAberrationBackend() override = default;

    void computeAberration(const ComplexData& data, ZernikeCoefficients coeff) const override;

    void applyZernikeCorrection(int Nx, int Ny, int Nz, complex_t* data, const ZernikeCoefficients& coeff) const;
    void zernikePhaseTestFunction(const ComplexData& output, ZernikeCoefficients coeff) const override;

private:

    void computeZernikePhase(real_t* correction, int x, int y, int Nx, int Ny, ZernikeCoefficients coeff) const ;

};

class CPUAberrationBackendManager : public CPUBackendManager {
public:
    std::unique_ptr<CPUComputeBackend> createComputeBackend(CPUBackendConfig config) override;
};


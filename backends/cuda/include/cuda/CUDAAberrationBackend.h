#pragma once

#include "cuda_backend/CUDABackend.h"
#include "cuda_backend/CUDABackendManager.h"
#include "aberration_backend/IAberrationBackend.h"

class CUDAAberrationBackend : public CUDAComputeBackend, public IAberrationBackend{
public:
    using CUDAComputeBackend::CUDAComputeBackend;

    ~CUDAAberrationBackend() override = default;

    void computeAberration(const ComplexData& data) const override;
};


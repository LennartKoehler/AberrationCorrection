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


class CUDAAberrationBackendManager : public CUDABackendManager{

public:
    IComputeBackend& getComputeBackend(const BackendConfig& config) override{
        auto deconv = std::make_unique<CUDAAberrationBackend>(configToConfig(config));
        std::unique_lock<std::mutex> lock(mutex_);
        computeBackends.push_back(std::move(deconv));
        return *computeBackends.back();
    }
};

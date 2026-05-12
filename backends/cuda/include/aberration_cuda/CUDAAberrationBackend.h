#pragma once

#include "cuda_backend/CUDABackend.h"
#include "cuda_backend/CUDABackendManager.h"

class CUDAAberrationBackend : public CUDAComputeBackend{
public:
    using CUDAComputeBackend::CUDAComputeBackend;
};

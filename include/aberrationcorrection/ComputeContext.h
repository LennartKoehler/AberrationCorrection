#pragma once
#include "aberration_backend/IAberrationBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"
#include <memory>

struct ComputeContext{

    std::shared_ptr<IAberrationBackend> computeBackend;
    std::shared_ptr<IBackendMemoryManager> memoryManager;


};

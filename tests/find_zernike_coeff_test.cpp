#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <memory>

#include "cpu_backend/CPUBackend.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "aberration_backend/IAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "aberrationcorrection/BackendFactory.h"
#include "aberrationcorrection/ParameterSearch.h"
#include "aberrationcorrection/SearchAlgorithm.h"
#include "aberrationcorrection/ErrorFunction.h"
#include "aberrationcorrection/ComputeContext.h"

#include "cuda/CUDAAberrationBackend.h"
#include "cpu/CPUAberrationBackend.h"

static IAberrationBackend& getAberrationBackend(IBackend& backend) {
    IComputeBackend& compute = backend.mutableComputeManager();
    return dynamic_cast<IAberrationBackend&>(compute);
}

static void fillComplexData(ComplexData& data, int volume) {
    auto* ptr = data.getData();
    for (int i = 0; i < volume; ++i) {
        // Fill with deterministic complex values (real + i*imag)
        float x = static_cast<float>(i % 64) - 32.0f;
        float y = static_cast<float>((i / 64) % 64) - 32.0f;
        float sigma = 10.0f;
        float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
        ptr[i][0] = val;       // real part
        ptr[i][1] = val * 0.5f; // imaginary part
    }
}


static void findCoefficients(IBackend& backend, const ComplexData& input, const ComplexData& target){

    ZernikeCoefficients initial{{0.0,0.0,0.0,0.0,0.0}};
    ZernikeCoefficients min{{0.0,0.0,0.0,0.0,0.0}};
    ZernikeCoefficients max{{1.0,1.0,1.0,1.0,1.0}};
    GridSearchAlgorithm searchAlgo{initial, min, max};
    float stepSize = 0.1;
    float stepRedunctionSize = 1.5;
    float gridDepth = 3;
    searchAlgo.init(stepSize, stepRedunctionSize, gridDepth);
    IAberrationBackend* ab = dynamic_cast<IAberrationBackend*>(&backend.mutableComputeManager());
    IBackendMemoryManager* mm = dynamic_cast<IBackendMemoryManager*>(&backend.mutableMemoryManager());

    RealData scratch = mm->allocateMemoryOnDeviceReal(input.getRealSize());
    RealData targetReal = mm->allocateMemoryOnDeviceReal(input.getRealSize());
    ab->backwardFFT(target, targetReal);
    // BasicErrorFunction errorFn{target};
    RealSpaceErrorFunction errorFn{targetReal, scratch};

    ComputeContext context{
        std::shared_ptr<IAberrationBackend>(ab),
        std::shared_ptr<IBackendMemoryManager>(mm)};

    ParameterSearch search;
    std::pair<ZernikeCoefficients, ComplexData> result= search.search(
        input,
        searchAlgo,
        errorFn,
        context);


    std::cout << " Following coefficients were found as ideal: Z0=" << result.first.c[0]
              << " Z1=" << result.first.c[1]
              << " Z2=" << result.first.c[2]
              << " Z3=" << result.first.c[3]
              << " Z12=" << result.first.c[4] << std::endl;
}

static void runWithManager(IBackendManager& manager,
                           IBackendMemoryManager& memMgr,
                           IBackend& backend,
                           ZernikeCoefficients coeff) {

    const int Nx = 64, Ny = 64, Nz = 64;
    CuboidShape shape(Nx, Ny, Nz);
    int volume = shape.getVolume();
    std::cout << "    Shape: " << Nx << " x " << Ny << " x " << Nz << std::endl;

    std::cout << "[1] Allocating ComplexData on device..." << std::endl;
    ComplexData referenceOnHost = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceComplexFull(shape);

    std::cout << "[2] Filling with test data..." << std::endl;
    fillComplexData(referenceOnHost, volume);

    // Save a copy of the original data on the host for comparison
    ComplexData inputOnDevice = memMgr.copyDataToDevice(referenceOnHost);
    ComplexData referenceOnDevice = memMgr.copyDataToDevice(referenceOnHost);

    std::cout << "[3] Applying Zernike aberration correction..." << std::endl;
    std::cout << "    Coefficients: Z0=" << coeff.c[0]
              << " Z1=" << coeff.c[1]
              << " Z2=" << coeff.c[2]
              << " Z3=" << coeff.c[3]
              << " Z12=" << coeff.c[4] << std::endl;
    IAberrationBackend& aberrationBackend = getAberrationBackend(backend);
    aberrationBackend.addZernikePhase(inputOnDevice, coeff);
    backend.sync();

    findCoefficients(backend, inputOnDevice, referenceOnDevice);



}

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cerr << "Usage: " << argv[0]
                  << " [backend=cpu|cuda] [z0 z1 z2 z3 z12]"
                  << std::endl;
        return 0;
    }

    std::string backendName = (argc > 1) ? argv[1] : "cuda";

    ZernikeCoefficients coeff{{0.0f, 0.0f, 0.0f, 0.5f, 0.0f}};
    if (argc > 6) {
        coeff.c[0] = static_cast<float>(std::atof(argv[2]));
        coeff.c[1] = static_cast<float>(std::atof(argv[3]));
        coeff.c[2] = static_cast<float>(std::atof(argv[4]));
        coeff.c[3] = static_cast<float>(std::atof(argv[5]));
        coeff.c[4] = static_cast<float>(std::atof(argv[6]));
    }

    auto logCallback = [](const std::string& msg, LogLevel level) {
        if (level >= LogLevel::INFO) {
            std::cout << "  [Backend] " << msg << std::endl;
        }
    };

    try {
        BackendConfig config;
        config.nThreads = 1;

        if (backendName == "cuda") {
            std::cout << "Using CUDA backend" << std::endl;
            CUDAAberrationBackendManager manager;
            manager.init(logCallback);
            IBackend& backend = manager.getBackend(config);
            IBackendMemoryManager& memMgr = backend.mutableMemoryManager();
            runWithManager(manager, memMgr, backend, coeff);
        } else if (backendName == "cpu") {
            std::cout << "Using CPU backend" << std::endl;
            CPUAberrationBackendManager manager;
            manager.init(logCallback);
            IBackend& backend = manager.getBackend(config);
            IBackendMemoryManager& memMgr = backend.mutableMemoryManager();
            runWithManager(manager, memMgr, backend, coeff);
        } else {
            std::cerr << "Unknown backend: " << backendName << ". Use 'cpu' or 'cuda'." << std::endl;
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

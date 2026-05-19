#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "aberration_backend/IAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "aberrationcorrection/ImageConversion.h"
#include "aberrationcorrection/BackendFactory.h"

#include "cuda/CUDAAberrationBackend.h"
#include "cpu/CPUAberrationBackend.h"

static IAberrationBackend& getAberrationBackend(IBackend& backend) {
    IComputeBackend& compute = backend.mutableComputeManager();
    return dynamic_cast<IAberrationBackend&>(compute);
}

static void runWithManager(IBackendManager& manager,
                           IBackendMemoryManager& memMgr,
                           IBackend& backend,
                           const std::string& outputFile,
                           ZernikeCoefficients coeff) {

    CuboidShape shape{32, 32, 32};
    ComplexData inputOnDevice = memMgr.allocateMemoryOnDeviceComplexFull(shape);


    std::cout << "Applying Zernike aberration correction..." << std::endl;
    std::cout << "    Coefficients: Z0=" << coeff.c[0]
              << " Z1=" << coeff.c[1]
              << " Z2=" << coeff.c[2]
              << " Z3=" << coeff.c[3]
              << " Z4=" << coeff.c[4] << std::endl;
    IAberrationBackend& aberrationBackend = getAberrationBackend(backend);
    aberrationBackend.zernikePhaseTestFunction(inputOnDevice, coeff);
    backend.sync();

    ComplexData dataOnHost = memMgr.moveDataFromDevice(inputOnDevice, BackendFactory::getInstance().getDefaultBackendMemoryManager());

    Image3D outputImage = ImageConversion::convertComplexDataToImage(dataOnHost);

    if (!TiffWriter::writeToFile(outputFile, outputImage)) {
        throw std::runtime_error("Failed to write output: " + outputFile);
    }

    std::cout << "Done. Output written to: " << outputFile << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <output.tif> [backend=cpu|cuda] [z0 z1 z2 z3]"
                  << std::endl;
        return 1;
    }

    std::string outputFile = argv[1];
    std::string backendName = (argc > 2) ? argv[2] : "cuda";

    ZernikeCoefficients coeff{{0.0f, 0.0f, 0.0f, 0.5f, 0.0f}};
    if (argc > 7) {
        coeff.c[0] = static_cast<float>(std::atof(argv[3]));
        coeff.c[1] = static_cast<float>(std::atof(argv[4]));
        coeff.c[2] = static_cast<float>(std::atof(argv[5]));
        coeff.c[3] = static_cast<float>(std::atof(argv[6]));
        coeff.c[4] = static_cast<float>(std::atof(argv[7]));
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
            runWithManager(manager, memMgr, backend, outputFile, coeff);
        } else if (backendName == "cpu") {
            std::cout << "Using CPU backend" << std::endl;
            CPUAberrationBackendManager manager;
            manager.init(logCallback);
            IBackend& backend = manager.getBackend(config);
            IBackendMemoryManager& memMgr = backend.mutableMemoryManager();
            runWithManager(manager, memMgr, backend, outputFile, coeff);
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

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "aberration_backend/IAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "aberrationcorrection/BackendFactory.h"
#include "aberrationcorrection/ImageConversion.h"
#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffWriter.h"

#include "cuda/CUDAAberrationBackend.h"
#include "cpu/CPUAberrationBackend.h"

static IAberrationBackend& getAberrationBackend(IBackend& backend) {
    IComputeBackend& compute = backend.mutableComputeManager();
    return dynamic_cast<IAberrationBackend&>(compute);
}

static void fillData(ComplexData& data, int volume) {
    auto* ptr = data.getData();
    for (int i = 0; i < volume; ++i) {
        // Fill with deterministic complex values (real + i*imag)
        float x = static_cast<float>(i % 64) - 32.0f;
        float y = static_cast<float>((i / 64) % 64) - 32.0f;
        float sigma = 10.0f;
        float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
        ptr[i][0] = val;       // real part
        ptr[i][1] = 0.0;       // im part
    }
}

static void fillData(RealData& data, int volume) {
    auto* ptr = data.getData();
    for (int i = 0; i < volume; ++i) {
        // Fill with deterministic complex values (real + i*imag)
        float x = static_cast<float>(i % 64) - 32.0f;
        float y = static_cast<float>((i / 64) % 64) - 32.0f;
        float sigma = 10.0f;
        float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
        ptr[i] = val;       // real part
    }
}

static bool compareData(const Image3D& a, const Image3D& b, float tolerance = 1e-5f) {
    if (a.getShape().getVolume() != b.getShape().getVolume()) return false;
    return a.isEqual(b, tolerance);
}

static Image3D runComplex(IBackendManager& manager,
                           IBackendMemoryManager& memMgr,
                           IBackend& backend,
                           ZernikeCoefficients coeff) {

    const int Nx = 64, Ny = 64, Nz = 32;
    CuboidShape shape(Nx, Ny, Nz);
    int volume = shape.getVolume();
    std::cout << "    Shape: " << Nx << " x " << Ny << " x " << Nz << std::endl;

    IAberrationBackend& aberrationBackend = getAberrationBackend(backend);

    std::cout << "[1] Allocating ComplexData on device..." << std::endl;
    ComplexData referenceOnHost = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceComplexFull(shape);

    std::cout << "[2] Filling with test data..." << std::endl;
    fillData(referenceOnHost, volume);

    // Save a copy of the original data on the host for comparison
    ComplexData inputOnDevice = memMgr.copyDataToDevice(referenceOnHost);
    aberrationBackend.forwardFFT(inputOnDevice, inputOnDevice);

    std::cout << "[3] Applying Zernike aberration correction..." << std::endl;
    std::cout << "    Coefficients: Z0=" << coeff.c[0]
              << " Z1=" << coeff.c[1]
              << " Z2=" << coeff.c[2]
              << " Z3=" << coeff.c[3]
              << " Z12=" << coeff.c[4] << std::endl;
    // aberrationBackend.addZernikePhase(inputOnDevice, coeff);
    // backend.sync();

    // aberrationBackend.backwardFFT(inputOnDevice, inputOnDevice);

    ComplexData resultOnHost = memMgr.moveDataFromDevice(inputOnDevice, BackendFactory::getInstance().getDefaultBackendMemoryManager());
    Image3D result = ImageConversion::convertComplexDataToImage(resultOnHost);
    return result;

}

static Image3D runReal(IBackendManager& manager,
                           IBackendMemoryManager& memMgr,
                           IBackend& backend,
                           ZernikeCoefficients coeff) {

    const int Nx = 64, Ny = 64, Nz = 32;
    CuboidShape shape(Nx, Ny, Nz);
    int volume = shape.getVolume();
    std::cout << "    Shape: " << Nx << " x " << Ny << " x " << Nz << std::endl;

    IAberrationBackend& aberrationBackend = getAberrationBackend(backend);

    std::cout << "[1] Allocating ComplexData on device..." << std::endl;
    RealData referenceOnHost = BackendFactory::getInstance().getDefaultBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(shape);

    std::cout << "[2] Filling with test data..." << std::endl;
    fillData(referenceOnHost, volume);

    // Save a copy of the original data on the host for comparison
    RealData inputOnDevice = memMgr.copyDataToDevice(referenceOnHost);
    DataView<complex_t> complexInterpretation = memMgr.reinterpret(inputOnDevice);
    aberrationBackend.forwardFFT(inputOnDevice, complexInterpretation);

    std::cout << "[3] Applying Zernike aberration correction..." << std::endl;
    std::cout << "    Coefficients: Z0=" << coeff.c[0]
              << " Z1=" << coeff.c[1]
              << " Z2=" << coeff.c[2]
              << " Z3=" << coeff.c[3]
              << " Z12=" << coeff.c[4] << std::endl;
    // aberrationBackend.addZernikePhase(complexInterpretation, coeff);
    // backend.sync();

    // aberrationBackend.backwardFFT(complexInterpretation, inputOnDevice);

    ComplexData resultOnHost = memMgr.moveDataFromDevice(complexInterpretation, BackendFactory::getInstance().getDefaultBackendMemoryManager());

    Image3D result = ImageConversion::convertComplexDataToImage(resultOnHost);
    return result;

}

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cerr << "Usage: " << argv[0]
                  << " [backend=cpu|cuda] [z0 z1 z2 z3 z12]"
                  << std::endl;
        return 0;
    }
    throw std::runtime_error("this test cant succeed currently, ignore");

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

        Image3D imReal;
        Image3D imComplex;
        if (backendName == "cuda") {
            std::cout << "Using CUDA backend" << std::endl;
            CUDAAberrationBackendManager manager;
            manager.init(logCallback);
            IBackend& backend = manager.getBackend(config);
            IBackendMemoryManager& memMgr = backend.mutableMemoryManager();
            imComplex = runComplex(manager, memMgr, backend, coeff);
            imReal = runReal(manager, memMgr, backend, coeff);
        } else if (backendName == "cpu") {
            std::cout << "Using CPU backend" << std::endl;
            CPUAberrationBackendManager manager;
            manager.init(logCallback);
            IBackend& backend = manager.getBackend(config);
            IBackendMemoryManager& memMgr = backend.mutableMemoryManager();
            imComplex = runComplex(manager, memMgr, backend, coeff);
            imReal = runReal(manager, memMgr, backend, coeff);
        } else {
            std::cerr << "Unknown backend: " << backendName << ". Use 'cpu' or 'cuda'." << std::endl;
            return 1;
        }
        TiffWriter::writeToFile("testReal.tif", imReal);
        TiffWriter::writeToFile("testComplex.tif", imComplex);
        if(compareData(imReal, imComplex)){
            std::cout << "Test successfull, using r2c and c2c are the same" << std::endl;
        }
        else std::cout << "ERROR: Test not successfull, using r2c and c2c are not the same" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

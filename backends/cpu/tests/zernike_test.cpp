#include <iostream>
#include <vector>
#include <cstring>

#include "aberration_backend/IAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "cpu/CPUAberrationBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"

static void fillData(ComplexData& data, int volume) {
    auto* ptr = data.getData();
    for (int i = 0; i < volume; ++i) {
        ptr[i][0] = static_cast<real_t>((i * 7 + 3) % 100) / 10.0f;
        ptr[i][1] = static_cast<real_t>((i * 13 + 5) % 100) / 10.0f;
    }
}

static bool dataChanged(const complex_t* before, const complex_t* after, int volume) {
    for (int i = 0; i < volume; ++i) {
        if (before[i][0] != after[i][0] || before[i][1] != after[i][1]) {
            return true;
        }
    }
    return false;
}
static void testFn(IAberrationBackend& backend, ComplexData& data, ZernikeCoefficients coeff){
    std::cout << "Starting testFn" << std::endl;
    for (int i = 0; i < 100000; i++){
        backend.computeAberration(data, coeff);
    }
    std::cout << "Finished testFn" << std::endl;
}


int main() {
    try {
        std::cout << "[1] Creating CPU backend..." << std::endl;

        CPUAberrationBackendManager manager;
        manager.init([](const std::string& msg, LogLevel level) {
            if (level >= LogLevel::INFO) std::cout << "  [CPU] " << msg << std::endl;
        });

        BackendConfig config;
        config.nThreads = 1;
        CPUAberrationBackend& cpuBackend = dynamic_cast<CPUAberrationBackend&>(manager.getComputeBackend(config));
        IBackendMemoryManager& memMgr = manager.getBackendMemoryManager(config);

        const int Nx = 32, Ny = 32, Nz = 32;
        CuboidShape shape(Nx, Ny, Nz);
        int volume = shape.getVolume();

        int passed = 0;
        int total = 0;

        std::cout << "[2] Running Zernike smoke tests on " << Nx << "x" << Ny << "x" << Nz << " dataset..." << std::endl;

        {
            ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData before = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            volume = data.getSize().getVolume();
            fillData(data, volume);

            memMgr.memCopy(data, before);

            ZernikeCoefficients coeff{{0.0f, 0.0f, 0.0f, 0.0f}};
            testFn(cpuBackend, data, coeff);

            bool ok = !dataChanged(before.getData(), data.getData(), volume);
            ++total; if (ok) ++passed;
            std::cout << "  Zero coefficients (identity): " << (ok ? "PASSED" : "FAILED") << std::endl;
        }

        {
            ComplexData data = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            ComplexData before = memMgr.allocateMemoryOnDeviceComplexFull(shape);
            fillData(before, volume);

            memMgr.memCopy(before, data);
            ZernikeCoefficients coeff{{0.5f, 0.3f, -0.2f, 0.1f}};
            cpuBackend.computeAberration(data, coeff);

            bool ok = dataChanged(before.getData(), data.getData(), volume);
            ++total; if (ok) ++passed;
            std::cout << "  Non-zero coefficients (data changed): " << (ok ? "PASSED" : "FAILED") << std::endl;
        }

        std::cout << "\nResults: " << passed << "/" << total << " tests passed" << std::endl;
        if (passed == total) {
            std::cout << "ALL TESTS PASSED" << std::endl;
            return 0;
        } else {
            std::cout << "SOME TESTS FAILED" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

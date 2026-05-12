#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

#include "dolphin_image/Image3D.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include "cuda_backend/CUDABackendManager.h"

namespace ABERR {
    cudaError_t applyPhaseCorrection(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ, cudaStream_t stream = 0);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.tif> <output.tif> [channel]" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int channel = 0;
    if (argc > 3) {
        channel = std::stoi(argv[3]);
    }

    try {
        std::cout << "[1] Reading input image..." << std::endl;
        auto maybeImage = TiffReader::readTiffFile(inputFile, channel);
        if (!maybeImage.has_value()) {
            std::cerr << "Failed to read image: " << inputFile << std::endl;
            return 1;
        }
        Image3D image = std::move(*maybeImage);
        CuboidShape shape = image.getShape();
        std::cout << "    Shape: " << shape.width << " x " << shape.height << " x " << shape.depth << std::endl;

        std::cout << "[2] Creating CUDA backend via CUDABackendManager..." << std::endl;
        CUDABackendManager manager;
        manager.init([](const std::string& msg, LogLevel level) {
            if (level >= LogLevel::INFO) {
                std::cout << "  [CUDA] " << msg << std::endl;
            }
        });

        BackendConfig config;
        config.nThreads = 1;
        IBackend& backend = manager.getBackend(config);
        IComputeBackend& computeBackend = backend.mutableComputeManager();
        IBackendMemoryManager& memoryManager = backend.mutableMemoryManager();
        std::cout << "    Device: " << backend.getDeviceString() << std::endl;

        std::cout << "[3] Converting image to RealData on host..." << std::endl;
        size_t volume = shape.getVolume();
        size_t bytes = volume * sizeof(real_t);
        std::vector<real_t> hostData(volume);
        int idx = 0;
        for (const auto& pixel : image) {
            hostData[idx++] = static_cast<real_t>(pixel);
        }
        RealData inputReal(nullptr, hostData.data(), shape, shape, bytes, 0);

        std::cout << "[4] Copying data to device..." << std::endl;
        RealData inputOnDevice = memoryManager.copyDataToDevice(inputReal);

        std::cout << "[5] Forward FFT (R2C)..." << std::endl;
        ComplexData complexOnDevice = memoryManager.allocateMemoryOnDeviceComplex(shape);
        computeBackend.forwardFFT(inputOnDevice, complexOnDevice);
        backend.sync();
        std::cout << "    Complex shape: " << complexOnDevice.getSize().width << " x "
                  << complexOnDevice.getSize().height << " x " << complexOnDevice.getSize().depth << std::endl;

        std::cout << "[6] Applying aberration correction..." << std::endl;
        CuboidShape complexShape = complexOnDevice.getSize();

        real_t* d_phaseX = nullptr;
        real_t* d_phaseY = nullptr;
        real_t* d_phaseZ = nullptr;
        cudaMalloc(&d_phaseX, complexShape.width * sizeof(real_t));
        cudaMalloc(&d_phaseY, complexShape.height * sizeof(real_t));
        cudaMalloc(&d_phaseZ, complexShape.depth * sizeof(real_t));
        cudaMemset(d_phaseX, 0, complexShape.width * sizeof(real_t));
        cudaMemset(d_phaseY, 0, complexShape.height * sizeof(real_t));
        cudaMemset(d_phaseZ, 0, complexShape.depth * sizeof(real_t));

        cudaError_t err = ABERR::applyPhaseCorrection(
            complexShape.width, complexShape.height, complexShape.depth,
            complexOnDevice.getData(),
            d_phaseX, d_phaseY, d_phaseZ);
        if (err != cudaSuccess) {
            std::cerr << "applyPhaseCorrection failed: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        std::cout << "[7] Inverse FFT (C2R)..." << std::endl;
        RealData outputOnDevice = memoryManager.allocateMemoryOnDeviceReal(shape);
        computeBackend.backwardFFT(complexOnDevice, outputOnDevice);
        backend.sync();

        std::cout << "[8] Copying result to host..." << std::endl;
        RealData outputReal = memoryManager.moveDataFromDevice(
            outputOnDevice,
            memoryManager);

        std::cout << "[9] Writing output..." << std::endl;
        Image3D outputImage(shape, 0.0f);
        idx = 0;
        for (auto& pixel : outputImage) {
            pixel = static_cast<float>(outputReal[idx]);
            ++idx;
        }

        if (!TiffWriter::writeToFile(outputFile, outputImage)) {
            std::cerr << "Failed to write output: " << outputFile << std::endl;
            return 1;
        }

        cudaFree(d_phaseX);
        cudaFree(d_phaseY);
        cudaFree(d_phaseZ);

        std::cout << "Output written to: " << outputFile << std::endl;
        std::cout << "PASSED" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

#include "cpu/CPUAberrationBackend.h"
#include "aberration_backend/ZernikeCoefficients.h"
#include <cmath>
#include <fftw3.h>


void CPUAberrationBackend::computeZernikePhase(real_t* correction, int x, int y, int Nx, int Ny, ZernikeCoefficients coeff) const {


    // Map pixel coordinates to normalized coordinates
    // The center of the image maps to (0, 0), and the unit circle
    // circumscribes the image (pixels outside the disk get no phase).
    // real_t rho_x = (real_t(2) * x - Nx + 1) / (Nx - 1);
    // real_t rho_x = static_cast<real_t>(x - Nx + 1)/(Nx - 1);
    real_t rho_x = (real_t(2) * x - Nx + 1) / (Nx - 1);
    real_t rho_y = (real_t(2) * y - Ny + 1) / (Ny - 1);
    real_t rho_sq = rho_x * rho_x + rho_y * rho_y;

    //TODO only inside unit disk?
    real_t rho = sqrt(rho_sq);
    real_t theta = atan2(rho_y, rho_x);

    real_t Z0 = 1;
    real_t Z1 = 2 * rho * cos(theta);
    real_t Z2 = 2 * rho * sin(theta);
    real_t Z3 = sqrt(real_t(3)) * (2 * rho_sq - 1);
    real_t Z12 = sqrt(real_t(5)) * (6 * rho_sq * rho_sq * rho_sq - 6 * rho_sq + 1);

    *correction = coeff.c[0] * Z0 + coeff.c[1] * Z1 + coeff.c[2] * Z2 + coeff.c[3] * Z3 + coeff.c[4] * Z12;
}


void CPUAberrationBackend::addZernikePhase(ComplexData& image, ZernikeCoefficients coeff) const {
    // assert(image.getSize() == image.getRealSize());// only works for full complex values
    CuboidShape dataSize = image.getSize();
    CuboidShape realSize = image.getRealSize();
    addZernikePhase_(dataSize, realSize, image.getData(), coeff);
}


void CPUAberrationBackend::addZernikePhase_(CuboidShape dataSize, CuboidShape realSize, complex_t* data, const ZernikeCoefficients& coeff) const {
    int Nx = dataSize.width;
    int Ny = dataSize.height;
    int Nz = dataSize.depth;
    int realNx = realSize.width;
    int realNy = realSize.height;
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int index = z * (Nx * Ny) + y * Nx + x;

                real_t phase = 0.0;

                computeZernikePhase(&phase, x, y, realNx, realNy, coeff);

                real_t cosPhase = cos(phase);
                real_t sinPhase = sin(phase);
                real_t re = data[index][0];
                real_t im = data[index][1];
                data[index][0] = re * cosPhase - im * sinPhase;
                data[index][1] = re * sinPhase + im * cosPhase;
            }
        }
    }
}



void CPUAberrationBackend::subtractZernikePhase(ComplexData& image, ZernikeCoefficients coeff) const {
    CuboidShape dataSize = image.getSize();
    CuboidShape realSize = image.getRealSize();
    subtractZernikePhase_(dataSize, realSize, image.getData(), coeff);
}


void CPUAberrationBackend::subtractZernikePhase_(CuboidShape dataSize, CuboidShape realSize, complex_t* data, const ZernikeCoefficients& coeff) const {
    int Nx = dataSize.width;
    int Ny = dataSize.height;
    int Nz = dataSize.depth;
    int realNx = realSize.width;
    int realNy = realSize.height;
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int index = z * (Nx * Ny) + y * Nx + x;

                real_t phase = 0.0;

                computeZernikePhase(&phase, x, y, realNx, realNy, coeff);

                real_t cosPhase = cos(phase);
                real_t sinPhase = sin(phase);
                real_t re = data[index][0];
                real_t im = data[index][1];
                data[index][0] = re * cosPhase + im * sinPhase;
                data[index][1] = -re * sinPhase + im * cosPhase;
            }
        }
    }
}


void CPUAberrationBackend::zernikePhaseTestFunction(ComplexData& output, ZernikeCoefficients coeff) const {
    CuboidShape dataSize = output.getSize();
    CuboidShape realSize = output.getRealSize();
    int Nx = dataSize.width;
    int Ny = dataSize.height;
    int Nz = dataSize.depth;
    int realNx = realSize.width;
    int realNy = realSize.height;

    complex_t* rawData = output.getData();

    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int index = z * (Nx * Ny) + y * Nx + x;

                real_t phase = 0.0;

                computeZernikePhase(&phase, x, y, realNx, realNy, coeff);

                rawData[index][0] = phase;
                rawData[index][1] = 0.0;
            }
        }
    }
}

std::unique_ptr<CPUComputeBackend> CPUAberrationBackendManager::createComputeBackend(CPUBackendConfig config) {
    return std::move(std::make_unique<CPUAberrationBackend>(config, *fftwManager));
}


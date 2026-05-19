#include "kernels.cuh"

__global__
void applyPhaseCorrectionGlobal(int Nx, int Ny, int Nz, complex_t* data, const real_t* phaseX, const real_t* phaseY, const real_t* phaseZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        real_t phase = phaseX[x] + phaseY[y] + phaseZ[z];
        real_t cosPhase = cos(phase);
        real_t sinPhase = sin(phase);

        real_t re = data[index][0];
        real_t im = data[index][1];

        data[index][0] = re * cosPhase - im * sinPhase;
        data[index][1] = re * sinPhase + im * cosPhase;
    }
}


__device__
void computeZernikePhase(real_t* correction, int x, int y, int Nx, int Ny, ZernikeCoefficients coeff){

    // Map pixel coordinates to normalized coordinates
    // The center of the image maps to (0, 0), and the unit circle
    // circumscribes the image (pixels outside the disk get no phase).
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


__global__
void applyZernikePolynomialsGlobal(int Nx, int Ny, int Nz, complex_t* image, ZernikeCoefficients coeff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        // Compute the Zernike aberration phase δφ at this pixel
        real_t deltaPhi = 0;
        computeZernikePhase(&deltaPhi, x, y, Nx, Ny, coeff);

        // Apply correction by multiplying with e^{-i·δφ}:
        //   (re + i·im) · (cos(δφ) - i·sin(δφ))
        //   = (re·cos(δφ) + im·sin(δφ)) + i·(im·cos(δφ) - re·sin(δφ))
        // This subtracts the aberration phase without extracting atan2,
        // which is more numerically stable and avoids branch/roundtrip errors.
        real_t cosP = cos(deltaPhi);
        real_t sinP = sin(deltaPhi);

        real_t re = image[index][0];
        real_t im = image[index][1];

        image[index][0] = re * cosP + im * sinP;
        image[index][1] = im * cosP - re * sinP;
    }
}

__global__
void zernikePhaseTestGlobal(int Nx, int Ny, int Nz, complex_t* image, ZernikeCoefficients coeff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        // Compute the Zernike aberration phase δφ at this pixel
        real_t deltaPhi = 0;
        computeZernikePhase(&deltaPhi, x, y, Nx, Ny, coeff);

        image[index][0] = deltaPhi;
        image[index][1] = 0.0;
    }
}

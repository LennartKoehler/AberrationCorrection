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

__global__
void applyZernikePolynomialsGlobal(int Nx, int Ny, int Nz, complex_t* data, ZernikeCoefficients coeff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        // Map pixel coordinates to normalized coordinates on the unit disk [-1, 1]
        // The center of the image maps to (0, 0), and the unit circle
        // circumscribes the image (pixels outside the disk get no phase).
        real_t rho_x = (real_t(2) * x - Nx + 1) / (Nx - 1);
        real_t rho_y = (real_t(2) * y - Ny + 1) / (Ny - 1);
        real_t rho_z = (real_t(2) * z - Nz + 1) / (Nz - 1);
        real_t rho_sq = rho_x * rho_x + rho_y * rho_y + rho_z * rho_z;

        // Only apply aberration within the unit disk
        real_t phase = 0;
        if (rho_sq <= 1.0) {
            real_t rho = sqrt(rho_sq);
            real_t theta = atan2(rho_y, rho_x);

            // Zernike polynomials (Noll ordering):
            // Z0: Piston      = 1
            // Z1: Tilt X       = 2 * rho * cos(theta)
            // Z2: Tilt Y       = 2 * rho * sin(theta)
            // Z3: Defocus      = sqrt(3) * (2 * rho^2 - 1)
            real_t Z0 = 1;
            real_t Z1 = 2 * rho * cos(theta);
            real_t Z2 = 2 * rho * sin(theta);
            real_t Z3 = sqrt(3) * (2 * rho_sq - 1);

            phase = coeff.c[0] * Z0 + coeff.c[1] * Z1 + coeff.c[2] * Z2 + coeff.c[3] * Z3;
        }

        real_t cosPhase = cos(phase);
        real_t sinPhase = sin(phase);

        real_t re = data[index][0];
        real_t im = data[index][1];

        data[index][0] = re * cosPhase - im * sinPhase;
        data[index][1] = re * sinPhase + im * cosPhase;
    }
}

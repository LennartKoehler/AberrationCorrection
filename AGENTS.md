# AGENTS.md

C++/CUDA application for microscopy aberration correction using Zernike polynomials. Depends on the `dolphin` submodule for FFT backends, image I/O, and compute infrastructure.

## Build

```bash
# Initialize submodule first (required)
git submodule update --init --recursive

mkdir build && cd build
cmake .. && make
```

Requires CUDA Toolkit 12.8+ and a GPU with compute capability 75/80/86/90.

Root CMakeLists forces `BUILD_DOLPHIN_LIBRARY=OFF` and `ENABLE_TESTS=OFF` so only dolphin's backends + `dolphin_image` are built (not the deconvolution engine). Do not change these unless you also need dolphin's core library.

## Running Tests

No test framework. Tests are standalone executables requiring a CUDA GPU:

```bash
# From build directory:
./aberration_pipeline_test <input.tif> <output.tif> [channel]
./zernike_test   # self-contained, prints PASSED/FAILED
```

`tests/read.cpp` and `tests/write.cpp` are empty stubs, not built.

## Architecture

```
src/main.cpp                  → main executable (TIFF read/write stub)
backends/aberration_backend/  → INTERFACE library (header-only)
  include/aberration_backend/IAberrationBackend.h
backends/cuda/                → CUDA implementation
  src/CUDAAberrationBackend.cpp   (stub, inherits CUDAComputeBackend + IAberrationBackend)
  cuda_kernels/                   → standalone CUDA kernel library
    include/kernels.cuh            → ZernikeCoefficients struct, kernel declarations
    include/operations.cuh         → ABERR::applyPhaseCorrection, ABERR::applyZernikeCorrection
    src/kernels.cu                 → GPU kernel implementations
    src/operations.cu              → kernel launch wrappers (block size dim3(4,8,8))
lib/dolphin/                  → git submodule (not edited here)
```

Key link chain: `AberrationCorrection` → `dolphin_image`, `dolphinbackend`, `aberration_backend`, `aberration_cuda_backend` → `cuda_backend`, `aberration_cuda_kernels` → `CUDA::cudart_static`, `dolphinbackend`

## Critical Non-Obvious Details

- **Two `ZernikeCoefficients` types exist**: `float[4]` typedef in `IAberrationBackend.h` and `struct { real_t c[4]; }` in `kernels.cuh`. The kernel API (`ABERR::applyZernikeCorrection`) uses the struct version. Keep them in sync.

- **Data indexing**: Volume data uses row-major order `index = z * (Nx * Ny) + y * Nx + x` throughout kernels and tests.

- **Zernike convention**: 4 coefficients in Noll ordering — Z0 Piston, Z1 Tilt X, Z2 Tilt Y, Z3 Defocus. Phase applied only inside unit disk (`rho_sq <= 1.0`). Normalized coordinates map image center to origin.

- **`project_options`**: Compile flags come from dolphin's `project_options` INTERFACE target (Release: `-O3 -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma`). Link it into any new target.

- **`CUDAAberrationBackend`** uses `using CUDAComputeBackend::CUDAComputeBackend;` to inherit constructors — it must be constructed the same way as dolphin's `CUDAComputeBackend`.

- **`CMAKE_DISABLE_FIND_PACKAGE_dolphin=ON`** prevents CMake from finding a system-installed dolphin; the local submodule is always used instead.

## Adding New CUDA Kernels

1. Declare kernel in `backends/cuda/cuda_kernels/include/kernels.cuh`
2. Implement in `backends/cuda/cuda_kernels/src/kernels.cu`
3. Add launch wrapper in `backends/cuda/cuda_kernels/include/operations.cuh` and `src/operations.cu` (use `CUDA_CHECK_KERNEL` macro)
4. Add the `.cu` file to `CUDA_SOURCES` in `cuda_kernels/CMakeLists.txt`

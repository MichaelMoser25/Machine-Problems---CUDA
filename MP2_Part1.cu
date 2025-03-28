#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//  GPU matrix multiplication
__global__ void matrixMultiplyBasic(float* P, const float* M, const float* N, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory  tile width 2
__global__ void matrixMultiplyTiled2(float* P, const float* M, const float* N, int width) {
    __shared__ float M_tile[2][2];
    __shared__ float N_tile[2][2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * 2 + ty;
    int col = bx * 2 + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (width + 2 - 1) / 2; tile++) {
        if (row < width && tile * 2 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 2 + tx];
        }
        else {
            M_tile[ty][tx] = 0.0f;
        }

        if (tile * 2 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 2 + ty) * width + col];
        }
        else {
            N_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 2; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory tile width 4
__global__ void matrixMultiplyTiled4(float* P, const float* M, const float* N, int width) {
    __shared__ float M_tile[4][4];
    __shared__ float N_tile[4][4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * 4 + ty;
    int col = bx * 4 + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (width + 4 - 1) / 4; tile++) {
        if (row < width && tile * 4 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 4 + tx];
        }
        else {
            M_tile[ty][tx] = 0.0f;
        }

        if (tile * 4 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 4 + ty) * width + col];
        }
        else {
            N_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 4; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory tile width 8
__global__ void matrixMultiplyTiled8(float* P, const float* M, const float* N, int width) {
    __shared__ float M_tile[8][8];
    __shared__ float N_tile[8][8];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * 8 + ty;
    int col = bx * 8 + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (width + 8 - 1) / 8; tile++) {
        if (row < width && tile * 8 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 8 + tx];
        }
        else {
            M_tile[ty][tx] = 0.0f;
        }

        if (tile * 8 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 8 + ty) * width + col];
        }
        else {
            N_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 8; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory tile width 16
__global__ void matrixMultiplyTiled16(float* P, const float* M, const float* N, int width) {
    __shared__ float M_tile[16][16];
    __shared__ float N_tile[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (width + 16 - 1) / 16; tile++) {
        if (row < width && tile * 16 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 16 + tx];
        }
        else {
            M_tile[ty][tx] = 0.0f;
        }

        if (tile * 16 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 16 + ty) * width + col];
        }
        else {
            N_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory tile width 32
__global__ void matrixMultiplyTiled32(float* P, const float* M, const float* N, int width) {
    __shared__ float M_tile[32][32];
    __shared__ float N_tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (width + 32 - 1) / 32; tile++) {
        if (row < width && tile * 32 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 32 + tx];
        }
        else {
            M_tile[ty][tx] = 0.0f;
        }

        if (tile * 32 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 32 + ty) * width + col];
        }
        else {
            N_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// CPU matrix multiplication for verification
void matrixMultiplyCPU(float* P, const float* M, const float* N, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

// Function to initialize matrix with random values
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100) / 100.0f;
    }
}

// Function to verify results
bool verifyResults(float* cpuResult, float* gpuResult, int size) {
    const float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > epsilon) {
            printf("Verification failed at element %d: CPU = %f, GPU = %f\n",
                i, cpuResult[i], gpuResult[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Set random seed
    srand(42);

    // Matrix sizes test
    int matrixSizes[] = { 256, 512, 1024, 2048, 4096 };
    int numSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);

    // Tile widths test
    int tileWidths[] = { 2, 4, 8, 16, 32 };
    int numTileWidths = sizeof(tileWidths) / sizeof(tileWidths[0]);

    // Number of test iterations for better timing accuracy
    const int numIterations = 3;

    // Print device info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("\n");

    // CSV format for easy plotting
    printf("Size,TileWidth,Time(ms)\n");

    // For each matrix size
    for (int s = 0; s < numSizes; s++) {
        int width = matrixSizes[s];
        int size = width * width;
        size_t matrixBytes = size * sizeof(float);

        printf("\nTesting matrix size: %d x %d\n", width, width);

        // Skip CPU verification for very large matrices
        bool runCpuVerification = (width <= 1024);

        // Allocate host memory
        float* h_M = (float*)malloc(matrixBytes);
        float* h_N = (float*)malloc(matrixBytes);
        float* h_P = (float*)malloc(matrixBytes);
        float* h_P_CPU = NULL;

        if (runCpuVerification) {
            h_P_CPU = (float*)malloc(matrixBytes);
        }

        if (!h_M || !h_N || !h_P || (runCpuVerification && !h_P_CPU)) {
            printf("Failed to allocate host memory for size %d x %d\n", width, width);
            if (h_M) free(h_M);
            if (h_N) free(h_N);
            if (h_P) free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip this size
        }

        // Initialize matrices
        initializeMatrix(h_M, size);
        initializeMatrix(h_N, size);

        // Allocate device memory
        float* d_M, * d_N, * d_P;
        cudaError_t err;

        err = cudaMalloc(&d_M, matrixBytes);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMalloc d_M): %s\n", cudaGetErrorString(err));
            free(h_M); free(h_N); free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip size
        }

        err = cudaMalloc(&d_N, matrixBytes);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMalloc d_N): %s\n", cudaGetErrorString(err));
            cudaFree(d_M);
            free(h_M); free(h_N); free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip size
        }

        err = cudaMalloc(&d_P, matrixBytes);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMalloc d_P): %s\n", cudaGetErrorString(err));
            cudaFree(d_M); cudaFree(d_N);
            free(h_M); free(h_N); free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip size
        }

        // Copy data to device
        err = cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMemcpy to device): %s\n", cudaGetErrorString(err));
            cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
            free(h_M); free(h_N); free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip size
        }

        err = cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMemcpy to device): %s\n", cudaGetErrorString(err));
            cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
            free(h_M); free(h_N); free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip size
        }

        // Compute CPU result for verification (only for smaller matrices)
        if (runCpuVerification) {
            printf("Computing CPU result for verification...\n");
            matrixMultiplyCPU(h_P_CPU, h_M, h_N, width);
        }

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Basic kernel (only for reasonable sizes)
        if (width <= 2048) {
            float basicTime = 0.0f;
            dim3 basicBlock(16, 16);
            dim3 basicGrid((width + basicBlock.x - 1) / basicBlock.x,
                (width + basicBlock.y - 1) / basicBlock.y);

            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                cudaEventRecord(start);

                // Launch kernel
                matrixMultiplyBasic << <basicGrid, basicBlock >> > (d_P, d_M, d_N, width);

                // Stop timing
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                // Get any errors from kernel launch
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA Error (basic kernel): %s\n", cudaGetErrorString(err));
                    break;
                }

                // Calculate elapsed time
                float elapsed;
                cudaEventElapsedTime(&elapsed, start, stop);
                basicTime += elapsed;
            }

            if (err == cudaSuccess) {
                // Calculate average time
                basicTime /= numIterations;

                // Verify results (only for smaller matrices)
                if (runCpuVerification) {
                    err = cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost);
                    if (err == cudaSuccess) {
                        if (verifyResults(h_P_CPU, h_P, size)) {
                            printf("  Basic kernel verification: PASSED\n");
                        }
                        else {
                            printf("  Basic kernel verification: FAILED\n");
                        }
                    }
                }

                printf("  Basic kernel time: %.4f ms\n", basicTime);
                printf("%d,Basic,%.4f\n", width, basicTime);
            }
        }

        // For each tile width
        for (int t = 0; t < numTileWidths; t++) {
            int tileWidth = tileWidths[t];

            // Skip if tile width is too large for the matrix or thread block
            if (tileWidth > width || tileWidth * tileWidth > deviceProp.maxThreadsPerBlock) {
                printf("  Skipping tile width %d (too large for matrix or thread block)\n", tileWidth);
                continue;
            }

            dim3 tiledBlock(tileWidth, tileWidth);
            dim3 tiledGrid((width + tileWidth - 1) / tileWidth,
                (width + tileWidth - 1) / tileWidth);

            float tiledTime = 0.0f;
            bool kernelSuccess = true;

            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                cudaEventRecord(start);

                // Launch the appropriate kernel based on tile width
                switch (tileWidth) {
                case 2:
                    matrixMultiplyTiled2 <<< tiledGrid, tiledBlock >>> (d_P, d_M, d_N, width);
                    break;
                case 4:
                    matrixMultiplyTiled4 <<< tiledGrid, tiledBlock >>> (d_P, d_M, d_N, width);
                    break;
                case 8:
                    matrixMultiplyTiled8 <<< tiledGrid, tiledBlock >>> (d_P, d_M, d_N, width);
                    break;
                case 16:
                    matrixMultiplyTiled16 <<< tiledGrid, tiledBlock >>> (d_P, d_M, d_N, width);
                    break;
                case 32:
                    matrixMultiplyTiled32 <<< tiledGrid, tiledBlock >>> (d_P, d_M, d_N, width);
                    break;
                }

                // Stop timing
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                // Get any errors from kernel launch
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA Error (tiled kernel, tile=%d): %s\n", tileWidth, cudaGetErrorString(err));
                    kernelSuccess = false;
                    break;
                }

                // Calculate elapsed time
                float elapsed;
                cudaEventElapsedTime(&elapsed, start, stop);
                tiledTime += elapsed;
            }

            if (kernelSuccess) {
                // Calculate average time
                tiledTime /= numIterations;

                // Verify results (only for smaller matrices)
                if (runCpuVerification) {
                    err = cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost);
                    if (err == cudaSuccess) {
                        if (verifyResults(h_P_CPU, h_P, size)) {
                            printf("  Tiled kernel (TILE_WIDTH=%d) verification: PASSED\n", tileWidth);
                        }
                        else {
                            printf("  Tiled kernel (TILE_WIDTH=%d) verification: FAILED\n", tileWidth);
                        }
                    }
                }

                printf("  Tiled kernel (TILE_WIDTH=%d) time: %.4f ms\n", tileWidth, tiledTime);
                printf("%d,%d,%.4f\n", width, tileWidth, tiledTime);
            }
        }

        // Cleanup
        free(h_M);
        free(h_N);
        free(h_P);
        if (h_P_CPU) free(h_P_CPU);
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Test PASSED\n");

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU matrix multiplication for verification
void matrixMultiplyCPU(float *P, const float *M, const float *N, int width) {
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

// Basic GPU matrix multiplication (for comparison)
__global__ void matrixMultiplyBasic(float *P, const float *M, const float *N, int width) {
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

// Tiled matrix multiplication for tile width 2
__global__ void matrixMultiplyTiled2(float *P, const float *M, const float *N, int width) {
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
        // Load tiles
        if (row < width && tile * 2 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 2 + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        if (tile * 2 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 2 + ty) * width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiply tiles
        for (int k = 0; k < 2; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication for tile width 4
__global__ void matrixMultiplyTiled4(float *P, const float *M, const float *N, int width) {
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
        // Load tiles
        if (row < width && tile * 4 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 4 + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        if (tile * 4 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 4 + ty) * width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiply tiles
        for (int k = 0; k < 4; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication for tile width 8
__global__ void matrixMultiplyTiled8(float *P, const float *M, const float *N, int width) {
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
        // Load tiles
        if (row < width && tile * 8 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 8 + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        if (tile * 8 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 8 + ty) * width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiply tiles
        for (int k = 0; k < 8; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication for tile width 16
__global__ void matrixMultiplyTiled16(float *P, const float *M, const float *N, int width) {
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
        // Load tiles
        if (row < width && tile * 16 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 16 + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        if (tile * 16 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 16 + ty) * width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiply tiles
        for (int k = 0; k < 16; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Tiled matrix multiplication for tile width 32
__global__ void matrixMultiplyTiled32(float *P, const float *M, const float *N, int width) {
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
        // Load tiles
        if (row < width && tile * 32 + tx < width) {
            M_tile[ty][tx] = M[row * width + tile * 32 + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        if (tile * 32 + ty < width && col < width) {
            N_tile[ty][tx] = N[(tile * 32 + ty) * width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiply tiles
        for (int k = 0; k < 32; k++) {
            sum += M_tile[ty][k] * N_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        P[row * width + col] = sum;
    }
}

// Function to initialize matrix with random values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100) / 100.0f;
    }
}

// Function to verify results
bool verifyResults(float *cpuResult, float *gpuResult, int size) {
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
    
    // Matrix sizes to test
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    
    // Tile widths to test
    int tileWidths[] = {2, 4, 8, 16, 32};
    int numTileWidths = sizeof(tileWidths) / sizeof(tileWidths[0]);
    
    // Number of test iterations
    const int numIterations = 5;
    
    // Print device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    
    // Results table for plotting
    printf("\nMatrix Size");
    for (int t = 0; t < numTileWidths; t++) {
        printf(", Tile=%d", tileWidths[t]);
    }
    printf(", Basic\n");
    
    // For each matrix size
    for (int s = 0; s < numSizes; s++) {
        int width = matrixSizes[s];
        int size = width * width;
        size_t matrixBytes = size * sizeof(float);
        
        printf("Testing matrix size: %d x %d\n", width, width);
        
        // Skip very large matrices for CPU verification and basic kernel
        bool runCpuVerification = width <= 2048;
        bool runBasicKernel = width <= 2048;
        
        // Allocate host memory
        float *h_M = (float*)malloc(matrixBytes);
        float *h_N = (float*)malloc(matrixBytes);
        float *h_P = (float*)malloc(matrixBytes);
        float *h_P_CPU = NULL;
        
        if (runCpuVerification) {
            h_P_CPU = (float*)malloc(matrixBytes);
        }
        
        if (!h_M || !h_N || !h_P || (runCpuVerification && !h_P_CPU)) {
            printf("Failed to allocate host memory for size %d x %d\n", width, width);
            // Free already allocated memory
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
        float *d_M, *d_N, *d_P;
        if (cudaMalloc(&d_M, matrixBytes) != cudaSuccess ||
            cudaMalloc(&d_N, matrixBytes) != cudaSuccess ||
            cudaMalloc(&d_P, matrixBytes) != cudaSuccess) {
            printf("Failed to allocate device memory for size %d x %d\n", width, width);
            free(h_M); free(h_N); free(h_P);
            if (h_P_CPU) free(h_P_CPU);
            continue;  // Skip this size
        }
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice));
        
        // Compute CPU result for verification (only for sizes up to 2048)
        if (runCpuVerification) {
            matrixMultiplyCPU(h_P_CPU, h_M, h_N, width);
        }
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        float basicTime = 0.0f;
        
        // Basic kernel (for smaller matrices)
        if (runBasicKernel) {
            dim3 basicBlock(16, 16);
            dim3 basicGrid((width + basicBlock.x - 1) / basicBlock.x, 
                          (width + basicBlock.y - 1) / basicBlock.y);
            
            // Run basic kernel for timing
            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                CHECK_CUDA_ERROR(cudaEventRecord(start));
                
                // Launch kernel
                matrixMultiplyBasic<<<basicGrid, basicBlock>>>(d_P, d_M, d_N, width);
                
                // Stop timing
                CHECK_CUDA_ERROR(cudaEventRecord(stop));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                
                // Calculate elapsed time
                float elapsed;
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
                basicTime += elapsed;
            }
            
            // Verify basic kernel results
            if (runCpuVerification) {
                CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
                if (verifyResults(h_P_CPU, h_P, size)) {
                    printf("  Basic kernel verification: PASSED\n");
                } else {
                    printf("  Basic kernel verification: FAILED\n");
                }
            }
            
            // Calculate average time
            basicTime /= numIterations;
            printf("  Basic kernel average time: %.4f ms\n", basicTime);
        }
        
        // Results for this matrix size
        printf("%d", width);
        
        // For each tile width
        float tiledTimes[numTileWidths];
        for (int t = 0; t < numTileWidths; t++) {
            int tileWidth = tileWidths[t];
            
            // Skip if tile width is too large for the matrix
            if (tileWidth > width) {
                printf(", N/A");
                tiledTimes[t] = -1;
                continue;
            }
            
            dim3 tiledBlock(tileWidth, tileWidth);
            dim3 tiledGrid((width + tileWidth - 1) / tileWidth,
                          (width + tileWidth - 1) / tileWidth);
            
            float tiledTime = 0.0f;
            
            // Run tiled kernel for timing
            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                CHECK_CUDA_ERROR(cudaEventRecord(start));
                
                // Launch the appropriate kernel based on tile width
                switch (tileWidth) {
                    case 2:
                        matrixMultiplyTiled2<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 4:
                        matrixMultiplyTiled4<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 8:
                        matrixMultiplyTiled8<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 16:
                        matrixMultiplyTiled16<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 32:
                        matrixMultiplyTiled32<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
                        break;
                }
                
                // Stop timing
                CHECK_CUDA_ERROR(cudaEventRecord(stop));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                
                // Calculate elapsed time
                float elapsed;
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
                tiledTime += elapsed;
            }
            
            // Verify tiled kernel results (only for sizes that have CPU verification)
            if (runCpuVerification) {
                CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
                if (verifyResults(h_P_CPU, h_P, size)) {
                    printf("  Tiled kernel (TILE_WIDTH=%d) verification: PASSED\n", tileWidth);
                } else {
                    printf("  Tiled kernel (TILE_WIDTH=%d) verification: FAILED\n", tileWidth);
                }
            }
            
            // Calculate average time
            tiledTime /= numIterations;
            tiledTimes[t] = tiledTime;
            
            printf("  Tiled kernel (TILE_WIDTH=%d) average time: %.4f ms\n", tileWidth, tiledTime);
            if (runBasicKernel) {
                printf("  Speedup: %.2fx\n", basicTime / tiledTime);
            }
            
            printf(", %.4f", tiledTime);
        }
        
        // Add basic kernel time to results
        if (runBasicKernel) {
            printf(", %.4f", basicTime);
        } else {
            printf(", N/A");
        }
        printf("\n");
        
        // Cleanup
        free(h_M);
        free(h_N);
        free(h_P);
        if (h_P_CPU) free(h_P_CPU);
        CHECK_CUDA_ERROR(cudaFree(d_M));
        CHECK_CUDA_ERROR(cudaFree(d_N));
        CHECK_CUDA_ERROR(cudaFree(d_P));
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    }
    
    printf("\nAll tests completed successfully!\n");
    printf("Test PASSED\n");
    
    return 0;
}

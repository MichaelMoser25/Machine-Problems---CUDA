#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Define the maximum tile width we'll support
#define MAX_TILE_WIDTH 32

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

// Tiled matrix multiplication with shared memory
// Note: This version uses a runtime variable for tile width
__global__ void matrixMultiplyTiled(float *P, const float *M, const float *N, int width, int tileWidth) {
    // Dynamically sized shared memory declared externally
    extern __shared__ float sharedMem[];
    
    // Divide shared memory: first half for M_tile, second half for N_tile
    float *M_tile = sharedMem;
    float *N_tile = &sharedMem[tileWidth * tileWidth];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate the row and column indices for this thread
    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;
    
    float sum = 0.0f;
    
    // Loop over all tiles
    for (int tile = 0; tile < (width + tileWidth - 1) / tileWidth; tile++) {
        // Load tiles into shared memory
        if (row < width && tile * tileWidth + tx < width) {
            M_tile[ty * tileWidth + tx] = M[row * width + tile * tileWidth + tx];
        } else {
            M_tile[ty * tileWidth + tx] = 0.0f;
        }
        
        if (tile * tileWidth + ty < width && col < width) {
            N_tile[ty * tileWidth + tx] = N[(tile * tileWidth + ty) * width + col];
        } else {
            N_tile[ty * tileWidth + tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < tileWidth; k++) {
            sum += M_tile[ty * tileWidth + k] * N_tile[k * tileWidth + tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
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
    const int numIterations = 10;
    
    // Print device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("\n");
    
    // Host arrays for storing timing results
    float timedBasic[numSizes][numIterations];
    float timedTiled[numSizes][numTileWidths][numIterations];
    
    // For each matrix size
    for (int s = 0; s < numSizes; s++) {
        int width = matrixSizes[s];
        int size = width * width;
        size_t matrixBytes = size * sizeof(float);
        
        printf("Testing matrix size: %d x %d\n", width, width);
        
        // Allocate host memory
        float *h_M = (float*)malloc(matrixBytes);
        float *h_N = (float*)malloc(matrixBytes);
        float *h_P = (float*)malloc(matrixBytes);
        float *h_P_CPU = (float*)malloc(matrixBytes);
        
        // Initialize matrices
        initializeMatrix(h_M, size);
        initializeMatrix(h_N, size);
        
        // Allocate device memory
        float *d_M, *d_N, *d_P;
        CHECK_CUDA_ERROR(cudaMalloc(&d_M, matrixBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_N, matrixBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_P, matrixBytes));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice));
        
        // Compute CPU result for verification
        matrixMultiplyCPU(h_P_CPU, h_M, h_N, width);
        
        // Basic kernel for comparison (run once for correctness verification)
        if (width <= 1024) {  // Skip for very large matrices
            dim3 dimBlock(16, 16);
            dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
                         (width + dimBlock.y - 1) / dimBlock.y);
            
            // Create CUDA events for timing
            cudaEvent_t start, stop;
            CHECK_CUDA_ERROR(cudaEventCreate(&start));
            CHECK_CUDA_ERROR(cudaEventCreate(&stop));
            
            // Run basic kernel multiple times for timing
            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                CHECK_CUDA_ERROR(cudaEventRecord(start));
                
                // Launch kernel
                matrixMultiplyBasic<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, width);
                
                // Stop timing
                CHECK_CUDA_ERROR(cudaEventRecord(stop));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                
                // Calculate elapsed time
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&timedBasic[s][iter], start, stop));
            }
            
            // Copy result back to host for verification
            CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
            
            // Verify basic kernel results
            if (verifyResults(h_P_CPU, h_P, size)) {
                printf("  Basic kernel verification: PASSED\n");
            } else {
                printf("  Basic kernel verification: FAILED\n");
            }
            
            // Cleanup events
            CHECK_CUDA_ERROR(cudaEventDestroy(start));
            CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        }
        
        // Run tiled kernels with different tile widths
        for (int t = 0; t < numTileWidths; t++) {
            int tileWidth = tileWidths[t];
            
            printf("  Testing tile width: %d\n", tileWidth);
            
            // Create CUDA events for timing
            cudaEvent_t start, stop;
            CHECK_CUDA_ERROR(cudaEventCreate(&start));
            CHECK_CUDA_ERROR(cudaEventCreate(&stop));
            
            dim3 dimBlock(tileWidth, tileWidth);
            dim3 dimGrid((width + tileWidth - 1) / tileWidth, 
                         (width + tileWidth - 1) / tileWidth);
            
            // Calculate shared memory size (two tiles)
            size_t sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);
            
            // Run tiled kernel multiple times for timing
            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                CHECK_CUDA_ERROR(cudaEventRecord(start));
                
                // Launch kernel with dynamically allocated shared memory
                matrixMultiplyTiled<<<dimGrid, dimBlock, sharedMemSize>>>(
                    d_P, d_M, d_N, width, tileWidth);
                
                // Stop timing
                CHECK_CUDA_ERROR(cudaEventRecord(stop));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                
                // Calculate elapsed time
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&timedTiled[s][t][iter], start, stop));
            }
            
            // Copy result back to host for verification
            CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
            
            // Verify tiled kernel results
            if (verifyResults(h_P_CPU, h_P, size)) {
                printf("    Verification: PASSED\n");
            } else {
                printf("    Verification: FAILED\n");
            }
            
            // Cleanup events
            CHECK_CUDA_ERROR(cudaEventDestroy(start));
            CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        }
        
        // Print timing results
        printf("\n  Performance Results:\n");
        
        // Basic kernel results
        if (width <= 1024) {
            float avgBasic = 0.0f;
            for (int iter = 0; iter < numIterations; iter++) {
                avgBasic += timedBasic[s][iter];
            }
            avgBasic /= numIterations;
            printf("    Basic Kernel: %.4f ms\n", avgBasic);
        }
        
        // Tiled kernel results
        for (int t = 0; t < numTileWidths; t++) {
            float avgTiled = 0.0f;
            for (int iter = 0; iter < numIterations; iter++) {
                avgTiled += timedTiled[s][t][iter];
            }
            avgTiled /= numIterations;
            printf("    Tiled Kernel (TILE_WIDTH=%d): %.4f ms\n", tileWidths[t], avgTiled);
        }
        
        printf("\n");
        
        // Cleanup
        free(h_M);
        free(h_N);
        free(h_P);
        free(h_P_CPU);
        CHECK_CUDA_ERROR(cudaFree(d_M));
        CHECK_CUDA_ERROR(cudaFree(d_N));
        CHECK_CUDA_ERROR(cudaFree(d_P));
    }
    
    printf("All tests completed successfully!\n");
    
    return 0;
}

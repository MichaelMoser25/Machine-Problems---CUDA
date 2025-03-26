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

// Tiled matrix multiplication with 2x2 tiles
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

// Tiled matrix multiplication with 4x4 tiles
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

// Tiled matrix multiplication with 8x8 tiles
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

// Tiled matrix multiplication with 16x16 tiles
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

// Tiled matrix multiplication with 32x32 tiles
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
    const int numIterations = 10;
    
    // Print device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Max blocks per SM: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("\n");
    
    // Arrays for storing timing results
    float timedBasic[numSizes][numIterations];
    float timedTiled[numSizes][numTileWidths][numIterations];
    
    // For each matrix size
    for (int s = 0; s < numSizes; s++) {
        int width = matrixSizes[s];
        int size = width * width;
        size_t matrixBytes = size * sizeof(float);
        
        printf("Testing matrix size: %d x %d\n", width, width);
        
        // Skip very large matrices for basic kernel
        bool runBasicKernel = width <= 2048;
        
        // Allocate host memory
        float *h_M = (float*)malloc(matrixBytes);
        float *h_N = (float*)malloc(matrixBytes);
        float *h_P = (float*)malloc(matrixBytes);
        float *h_P_CPU = (float*)malloc(matrixBytes);
        
        if (!h_M || !h_N || !h_P || !h_P_CPU) {
            printf("Failed to allocate host memory for size %d x %d\n", width, width);
            continue;  // Skip this size if allocation failed
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
            free(h_M); free(h_N); free(h_P); free(h_P_CPU);
            continue;  // Skip this size if allocation failed
        }
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice));
        
        // Compute CPU result for verification (only for sizes up to 2048)
        if (width <= 2048) {
            printf("  Computing CPU result for verification...\n");
            matrixMultiplyCPU(h_P_CPU, h_M, h_N, width);
        }
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Basic kernel for comparison (run for smaller matrices)
        if (runBasicKernel) {
            dim3 basicBlock(16, 16);
            dim3 basicGrid((width + basicBlock.x - 1) / basicBlock.x, 
                          (width + basicBlock.y - 1) / basicBlock.y);
            
            printf("  Running basic kernel...\n");
            
            // Run basic kernel multiple times for timing
            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                CHECK_CUDA_ERROR(cudaEventRecord(start));
                
                // Launch kernel
                matrixMultiplyBasic<<<basicGrid, basicBlock>>>(d_P, d_M, d_N, width);
                
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
        }
        
        // Run tiled kernels with different tile widths
        for (int t = 0; t < numTileWidths; t++) {
            int tileWidth = tileWidths[t];
            
            // Skip larger tile sizes for smaller matrices
            if (tileWidth > width) {
                printf("  Skipping tile width %d (larger than matrix dimension)\n", tileWidth);
                for (int iter = 0; iter < numIterations; iter++) {
                    timedTiled[s][t][iter] = -1.0f;  // Mark as invalid
                }
                continue;
            }
            
            printf("  Testing tile width: %d\n", tileWidth);
            
            dim3 dimBlock(tileWidth, tileWidth);
            dim3 dimGrid((width + tileWidth - 1) / tileWidth, 
                         (width + tileWidth - 1) / tileWidth);
            
            // Run tiled kernel multiple times for timing
            for (int iter = 0; iter < numIterations; iter++) {
                // Start timing
                CHECK_CUDA_ERROR(cudaEventRecord(start));
                
                // Launch appropriate kernel based on tile width
                switch (tileWidth) {
                    case 2:
                        matrixMultiplyTiled2<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 4:
                        matrixMultiplyTiled4<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 8:
                        matrixMultiplyTiled8<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 16:
                        matrixMultiplyTiled16<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, width);
                        break;
                    case 32:
                        matrixMultiplyTiled32<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, width);
                        break;
                }
                
                // Stop timing
                CHECK_CUDA_ERROR(cudaEventRecord(stop));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                
                // Calculate elapsed time
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&timedTiled[s][t][iter], start, stop));
            }
            
            // Copy result back to host for verification (only for sizes up to 2048)
            if (width <= 2048) {
                CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
                
                // Verify tiled kernel results
                if (verifyResults(h_P_CPU, h_P, size)) {
                    printf("    Verification: PASSED\n");
                } else {
                    printf("    Verification: FAILED\n");
                }
            } else {
                printf("    Verification: SKIPPED (matrix too large for CPU comparison)\n");
            }
        }
        
        // Print timing results for this matrix size
        printf("\n  Performance Results for %d x %d:\n", width, width);
        
        // Basic kernel results (if run)
        if (runBasicKernel) {
            float avgBasic = 0.0f;
            for (int iter = 0; iter < numIterations; iter++) {
                avgBasic += timedBasic[s][iter];
            }
            avgBasic /= numIterations;
            printf("    Basic Kernel: %.4f ms\n", avgBasic);
        }
        
        // Tiled kernel results
        for (int t = 0; t < numTileWidths; t++) {
            int tileWidth = tileWidths[t];
            
            // Skip if this tile width was not run
            if (tileWidth > width) continue;
            
            float avgTiled = 0.0f;
            for (int iter = 0; iter < numIterations; iter++) {
                avgTiled += timedTiled[s][t][iter];
            }
            avgTiled /= numIterations;
            
            printf("    Tiled Kernel (TILE_WIDTH=%d): %.4f ms", tileWidth, avgTiled);
            
            if (runBasicKernel) {
                float avgBasic = 0.0f;
                for (int iter = 0; iter < numIterations; iter++) {
                    avgBasic += timedBasic[s][iter];
                }
                avgBasic /= numIterations;
                
                float speedup = avgBasic / avgTiled;
                printf(" (Speedup: %.2fx)", speedup);
            }
            
            printf("\n");
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
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    }
    
    // Print summary results for plotting
    printf("\nSummary Results for Plotting:\n");
    printf("Matrix Size");
    for (int t = 0; t < numTileWidths; t++) {
        printf(", Tile=%d", tileWidths[t]);
    }
    printf(", Basic\n");
    
    for (int s = 0; s < numSizes; s++) {
        int width = matrixSizes[s];
        printf("%d", width);
        
        // Tiled results
        for (int t = 0; t < numTileWidths; t++) {
            if (tileWidths[t] > width) {
                printf(", N/A");
                continue;
            }
            
            float avgTiled = 0.0f;
            for (int iter = 0; iter < numIterations; iter++) {
                avgTiled += timedTiled[s][t][iter];
            }
            avgTiled /= numIterations;
            
            printf(", %.4f", avgTiled);
        }
        
        // Basic kernel results
        if (width <= 2048) {
            float avgBasic = 0.0f;
            for (int iter = 0; iter < numIterations; iter++) {
                avgBasic += timedBasic[s][iter];
            }
            avgBasic /= numIterations;
            
            printf(", %.4f", avgBasic);
        } else {
            printf(", N/A");
        }
        
        printf("\n");
    }
    
    // Answer to Question 1
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int numSMs = deviceProp.multiProcessorCount;
    int totalThreads = maxThreadsPerSM * numSMs;
    
    printf("\nAnswer to Question 1:\n");
    printf("Maximum threads that can be simultaneously scheduled on the device: %d\n", totalThreads);
    printf("This is calculated as (max threads per SM: %d) * (number of SMs: %d)\n", 
           maxThreadsPerSM, numSMs);
    
    // Resource usage estimates for Question 2
    printf("\nAnswer to Question 2 (Resource Usage Estimates):\n");
    printf("Tile Width, Registers/Thread (est.), Shared Memory/Block, Blocks/SM (est.), Max Threads/SM\n");
    
    for (int t = 0; t < numTileWidths; t++) {
        int tileWidth = tileWidths[t];
        
        // Estimate register usage (this is a rough estimate)
        int regsPerThread = 16;  // This is an estimate, use nvcc with --ptxas-options=-v to get actual value
        
        // Calculate shared memory usage
        int sharedMemoryPerBlock = 2 * tileWidth * tileWidth * sizeof(float);
        
        // Estimate blocks per SM based on shared memory and thread count limitations
        int maxBlocksPerSM_sharedMem = deviceProp.sharedMemPerMultiprocessor / sharedMemoryPerBlock;
        int maxBlocksPerSM_threads = deviceProp.maxThreadsPerMultiProcessor / (tileWidth * tileWidth);
        int maxBlocksPerSM_limit = deviceProp.maxBlocksPerMultiProcessor;
        int maxBlocksPerSM = min(min(maxBlocksPerSM_sharedMem, maxBlocksPerSM_threads), maxBlocksPerSM_limit);
        
        // Calculate max threads that can be scheduled
        int maxThreads = maxBlocksPerSM * tileWidth * tileWidth;
        
        printf("%d, %d, %d bytes, %d, %d\n", 
               tileWidth, regsPerThread, sharedMemoryPerBlock, maxBlocksPerSM, maxThreads);
    }
    
    printf("\nNote: For accurate register usage, compile with nvcc --ptxas-options=-v\n");
    printf("All tests completed. Data is ready for plotting and analysis.\n");
    
    return 0;
}

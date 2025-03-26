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
    int matrixSizes[] = {256, 512, 1024};
    int numSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    
    // Number of test iterations
    const int numIterations = 5;
    
    // Print device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    
    // For each matrix size
    for (int s = 0; s < numSizes; s++) {
        int width = matrixSizes[s];
        int size = width * width;
        size_t matrixBytes = size * sizeof(float);
        
        printf("\nTesting matrix size: %d x %d\n", width, width);
        
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
        
        // Basic kernel
        dim3 basicBlock(16, 16);
        dim3 basicGrid((width + basicBlock.x - 1) / basicBlock.x, 
                       (width + basicBlock.y - 1) / basicBlock.y);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        float basicTime = 0.0f;
        
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
        
        // Copy result back for verification
        CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
        
        // Verify basic kernel results
        if (verifyResults(h_P_CPU, h_P, size)) {
            printf("  Basic kernel verification: PASSED\n");
        } else {
            printf("  Basic kernel verification: FAILED\n");
        }
        
        // Calculate average time
        basicTime /= numIterations;
        printf("  Basic kernel average time: %.4f ms\n", basicTime);
        
        // Tiled kernel (16x16)
        dim3 tiledBlock(16, 16);
        dim3 tiledGrid((width + tiledBlock.x - 1) / tiledBlock.x,
                       (width + tiledBlock.y - 1) / tiledBlock.y);
        
        float tiledTime = 0.0f;
        
        // Run tiled kernel for timing
        for (int iter = 0; iter < numIterations; iter++) {
            // Start timing
            CHECK_CUDA_ERROR(cudaEventRecord(start));
            
            // Launch kernel
            matrixMultiplyTiled16<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
            
            // Stop timing
            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            
            // Calculate elapsed time
            float elapsed;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
            tiledTime += elapsed;
        }
        
        // Copy result back for verification
        CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost));
        
        // Verify tiled kernel results
        if (verifyResults(h_P_CPU, h_P, size)) {
            printf("  Tiled kernel verification: PASSED\n");
        } else {
            printf("  Tiled kernel verification: FAILED\n");
        }
        
        // Calculate average time
        tiledTime /= numIterations;
        printf("  Tiled kernel average time: %.4f ms\n", tiledTime);
        printf("  Speedup: %.2fx\n", basicTime / tiledTime);
        
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
    
    printf("\nAll tests completed successfully!\n");
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CPU matrix multiplication for verification (handles rectangular matrices)
void matrixMultiplyCPU(float *P, const float *M, const float *N, int M_height, int M_width, int N_width) {
    for (int row = 0; row < M_height; row++) {
        for (int col = 0; col < N_width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < M_width; k++) {
                sum += M[row * M_width + k] * N[k * N_width + col];
            }
            P[row * N_width + col] = sum;
        }
    }
}

// Tiled matrix multiplication for rectangular matrices with 12x18 tiles
__global__ void matrixMultiplyTiled(float *P, const float *M, const float *N, int M_height, int M_width, int N_width) {
    // Fixed tile dimensions
    const int TILE_HEIGHT = 12;
    const int TILE_WIDTH = 18;
    
    // Shared memory for the tiles
    __shared__ float M_tile[12][18];
    __shared__ float N_tile[18][18];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate the row and column indices for this thread
    int row = by * TILE_HEIGHT + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    // Loop over all tiles
    int num_tiles = (M_width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load M tile into shared memory with boundary checks
        if (row < M_height && tile * TILE_WIDTH + tx < M_width) {
            M_tile[ty][tx] = M[row * M_width + tile * TILE_WIDTH + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        // Load N tile into shared memory with boundary checks
        if (tile * TILE_WIDTH + ty < M_width && col < N_width) {
            N_tile[ty][tx] = N[(tile * TILE_WIDTH + ty) * N_width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            // We only need to consider elements up to M_width for correct results
            if (tile * TILE_WIDTH + k < M_width) {
                sum += M_tile[ty][k] * N_tile[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result to global memory with boundary check
    if (row < M_height && col < N_width) {
        P[row * N_width + col] = sum;
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
    
    // Fixed tile dimensions as per assignment
    const int TILE_HEIGHT = 12;
    const int TILE_WIDTH = 18;
    
    // Test case dimensions
    int test_cases[][3] = {
        {750, 800, 850},    // M_height, M_width, N_width for case 1
        {2000, 1750, 1900}  // M_height, M_width, N_width for case 2
    };
    int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
    // Print device info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("\n");
    
    // For each test case
    for (int tc = 0; tc < num_test_cases; tc++) {
        int M_height = test_cases[tc][0];
        int M_width = test_cases[tc][1];
        int N_width = test_cases[tc][2];
        
        printf("Testing matrix dimensions: M(%d x %d) * N(%d x %d) = P(%d x %d)\n", 
               M_height, M_width, M_width, N_width, M_height, N_width);
        
        // Calculate memory sizes
        size_t M_bytes = M_height * M_width * sizeof(float);
        size_t N_bytes = M_width * N_width * sizeof(float);
        size_t P_bytes = M_height * N_width * sizeof(float);
        
        // Allocate host memory
        float *h_M = (float*)malloc(M_bytes);
        float *h_N = (float*)malloc(N_bytes);
        float *h_P = (float*)malloc(P_bytes);
        float *h_P_CPU = (float*)malloc(P_bytes);
        
        if (!h_M || !h_N || !h_P || !h_P_CPU) {
            printf("Error: Host memory allocation failed\n");
            return -1;
        }
        
        // Initialize matrices
        initializeMatrix(h_M, M_height * M_width);
        initializeMatrix(h_N, M_width * N_width);
        
        // Compute CPU result for verification
        matrixMultiplyCPU(h_P_CPU, h_M, h_N, M_height, M_width, N_width);
        
        // Allocate device memory
        float *d_M, *d_N, *d_P;
        cudaError_t err;
        
        err = cudaMalloc(&d_M, M_bytes);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMalloc d_M): %s\n", cudaGetErrorString(err));
            free(h_M); free(h_N); free(h_P); free(h_P_CPU);
            return -1;
        }
        
        err = cudaMalloc(&d_N, N_bytes);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMalloc d_N): %s\n", cudaGetErrorString(err));
            cudaFree(d_M);
            free(h_M); free(h_N); free(h_P); free(h_P_CPU);
            return -1;
        }
        
        err = cudaMalloc(&d_P, P_bytes);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMalloc d_P): %s\n", cudaGetErrorString(err));
            cudaFree(d_M); cudaFree(d_N);
            free(h_M); free(h_N); free(h_P); free(h_P_CPU);
            return -1;
        }
        
        // Copy data to device
        err = cudaMemcpy(d_M, h_M, M_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMemcpy to device): %s\n", cudaGetErrorString(err));
            cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
            free(h_M); free(h_N); free(h_P); free(h_P_CPU);
            return -1;
        }
        
        err = cudaMemcpy(d_N, h_N, N_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA Error (cudaMemcpy to device): %s\n", cudaGetErrorString(err));
            cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
            free(h_M); free(h_N); free(h_P); free(h_P_CPU);
            return -1;
        }
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Calculate block and grid dimensions
        dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
        dim3 dimGrid((N_width + TILE_WIDTH - 1) / TILE_WIDTH, 
                     (M_height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        
        // Number of iterations for timing
        const int numIterations = 10;
        float totalTime = 0.0f;
        
        // Run kernel multiple times and average the timing
        for (int iter = 0; iter < numIterations; iter++) {
            // Start timing
            cudaEventRecord(start);
            
            // Launch kernel
            matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, M_height, M_width, N_width);
            
            // Stop timing
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            // Check for kernel errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error (kernel): %s\n", cudaGetErrorString(err));
                break;
            }
            
            // Calculate elapsed time
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totalTime += milliseconds;
        }
        
        if (err == cudaSuccess) {
            // Copy result back to host for verification
            err = cudaMemcpy(h_P, d_P, P_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                printf("CUDA Error (cudaMemcpy to host): %s\n", cudaGetErrorString(err));
            } else {
                // Verify results
                if (verifyResults(h_P_CPU, h_P, M_height * N_width)) {
                    printf("  Verification: PASSED\n");
                } else {
                    printf("  Verification: FAILED\n");
                }
                
                // Print average execution time
                printf("  Average execution time: %.4f ms\n\n", totalTime / numIterations);
            }
        }
        
        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        free(h_P);
        free(h_P_CPU);
    }
    
    printf("All tests completed!\n");
    
    return 0;
}

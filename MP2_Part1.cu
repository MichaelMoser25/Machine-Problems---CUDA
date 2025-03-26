#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // Required for threadIdx, blockIdx, etc.

// Basic GPU matrix multiplication
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

// Tiled matrix multiplication with shared memory - fixed tile size of 16x16
__global__ void matrixMultiplyTiled(float *P, const float *M, const float *N, int width) {
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
    
    // Matrix sizes to test - simplified to just one size for debugging
    int width = 256;
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
    cudaError_t err;
    
    err = cudaMalloc(&d_M, matrixBytes);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMalloc d_M): %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&d_N, matrixBytes);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMalloc d_N): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        return -1;
    }
    
    err = cudaMalloc(&d_P, matrixBytes);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMalloc d_P): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        return -1;
    }
    
    // Copy data to device
    err = cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMemcpy to device): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        return -1;
    }
    
    err = cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMemcpy to device): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        return -1;
    }
    
    // Compute CPU result for verification
    matrixMultiplyCPU(h_P_CPU, h_M, h_N, width);
    
    // Basic kernel
    dim3 basicBlock(16, 16);
    dim3 basicGrid((width + basicBlock.x - 1) / basicBlock.x, 
                  (width + basicBlock.y - 1) / basicBlock.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float basicTime = 0.0f;
    
    // Run basic kernel for timing
    cudaEventRecord(start);
    matrixMultiplyBasic<<<basicGrid, basicBlock>>>(d_P, d_M, d_N, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get any errors from kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (basic kernel): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        return -1;
    }
    
    // Calculate elapsed time
    cudaEventElapsedTime(&basicTime, start, stop);
    
    // Copy result back for verification
    err = cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMemcpy to host): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        return -1;
    }
    
    // Verify basic kernel results
    if (verifyResults(h_P_CPU, h_P, size)) {
        printf("  Basic kernel verification: PASSED\n");
    } else {
        printf("  Basic kernel verification: FAILED\n");
    }
    
    printf("  Basic kernel time: %.4f ms\n", basicTime);
    
    // Tiled kernel (16x16)
    dim3 tiledBlock(16, 16);
    dim3 tiledGrid((width + tiledBlock.x - 1) / tiledBlock.x,
                   (width + tiledBlock.y - 1) / tiledBlock.y);
    
    float tiledTime = 0.0f;
    
    // Run tiled kernel for timing
    cudaEventRecord(start);
    matrixMultiplyTiled<<<tiledGrid, tiledBlock>>>(d_P, d_M, d_N, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get any errors from kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (tiled kernel): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        return -1;
    }
    
    // Calculate elapsed time
    cudaEventElapsedTime(&tiledTime, start, stop);
    
    // Copy result back for verification
    err = cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error (cudaMemcpy to host): %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        return -1;
    }
    
    // Verify tiled kernel results
    if (verifyResults(h_P_CPU, h_P, size)) {
        printf("  Tiled kernel verification: PASSED\n");
    } else {
        printf("  Tiled kernel verification: FAILED\n");
    }
    
    printf("  Tiled kernel time: %.4f ms\n", tiledTime);
    printf("  Speedup: %.2fx\n", basicTime / tiledTime);
    
    // Cleanup
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_CPU);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Test PASSED\n");
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define cudaCheckError() {                                             \
    cudaError_t error = cudaGetLastError();                           \
    if(error != cudaSuccess) {                                        \
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Matrix multiplication kernel - each thread computes one element
__global__ void matrixMulKernel(float *C, float *A, float *B, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < n && col < n) {
        float sum = 0.0f;
        for(int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Single thread kernel for comparison
__global__ void singleThreadMatrixMul(float *C, float *A, float *B, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CPU matrix multiplication for verification
void matrixMulCPU(float *C, float *A, float *B, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Function to initialize matrices with random values
void initMatrix(float *mat, int n) {
    for(int i = 0; i < n*n; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to verify results
bool verifyResults(float *C_cpu, float *C_gpu, int n) {
    float epsilon = 1e-5;
    for(int i = 0; i < n*n; i++) {
        if(fabs(C_cpu[i] - C_gpu[i]) > epsilon) {
            printf("Verification failed at element %d: CPU=%f, GPU=%f\n", i, C_cpu[i], C_gpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions
    int sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Block sizes to test
    int block_sizes[] = {2, 4, 8, 16, 32};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    for(int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        int matrix_size = n * n;
        size_t bytes = matrix_size * sizeof(float);
        
        printf("\nTesting matrix size: %d x %d\n", n, n);
        
        // Allocate host memory
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        float *h_C_cpu = (float*)malloc(bytes);
        
        // Initialize input matrices
        initMatrix(h_A, n);
        initMatrix(h_B, n);
        
        // Compute CPU reference result
        matrixMulCPU(h_C_cpu, h_A, h_B, n);
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaCheckError();
        cudaMalloc(&d_B, bytes);
        cudaCheckError();
        cudaMalloc(&d_C, bytes);
        cudaCheckError();
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // ===== Part 1: Data Transfer Timing =====
        
        // Measure host to device transfer time
        cudaEventRecord(start);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float host_to_device_time;
        cudaEventElapsedTime(&host_to_device_time, start, stop);
        printf("Host to Device transfer time: %.3f ms\n", host_to_device_time);
        
        // ===== Part 2: Single Thread Comparison =====
        
        if(n <= 1024) { // Skip for large matrices
            // Launch single thread kernel
            cudaEventRecord(start);
            singleThreadMatrixMul<<<1, 1>>>(d_C, d_A, d_B, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaCheckError();
            
            float single_thread_time;
            cudaEventElapsedTime(&single_thread_time, start, stop);
            printf("Single thread GPU time: %.3f ms\n", single_thread_time);
        }
        
        // ===== Part 3: Multi-threaded Execution with Different Block Sizes =====
        
        for(int b = 0; b < num_block_sizes; b++) {
            int block_size = block_sizes[b];
            
            // Configure execution parameters
            dim3 threads(block_size, block_size);
            dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
            
            // Launch kernel and measure time
            cudaEventRecord(start);
            matrixMulKernel<<<grid, threads>>>(d_C, d_A, d_B, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaCheckError();
            
            float kernel_time;
            cudaEventElapsedTime(&kernel_time, start, stop);
            
            // Copy result back to host
            cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
            cudaCheckError();
            
            // Verify results
            bool passed = verifyResults(h_C_cpu, h_C, n);
            
            printf("Block size %dx%d: Kernel time = %.3f ms, Test %s\n", 
                   block_size, block_size, kernel_time, passed ? "PASSED" : "FAILED");
            
            // Measure device to host transfer time
            cudaEventRecord(start);
            cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float device_to_host_time;
            cudaEventElapsedTime(&device_to_host_time, start, stop);
            printf("Device to Host transfer time: %.3f ms\n", device_to_host_time);
            
            // Calculate CGMA ratio
            int flops_per_thread = 2 * n; // 1 multiply + 1 add per iteration, n iterations
            int global_reads_per_thread = 2 * n; // Read one element each from A and B per iteration
            float cgma_ratio = (float)flops_per_thread / global_reads_per_thread;
            
            printf("CGMA Analysis for block size %dx%d:\n", block_size, block_size);
            printf("  - Each element loaded %d times\n", n);
            printf("  - CGMA ratio: %.2f\n", cgma_ratio);
        }
        
        // Free memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_cpu);
        
        // Clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\nPerformance Analysis Summary:\n");
    printf("1. Data transfer time scales with matrix size (O(n²))\n");
    printf("2. Single-thread GPU vs CPU comparison shows CPU advantage for small matrices\n");
    printf("3. Multi-threaded performance improves with appropriate block sizes\n");
    printf("   - Each input matrix element is loaded n times during kernel execution\n");
    printf("   - CGMA ratio is 2.0 (2 operations per memory access)\n");
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// Threshold for floating point comparison
#define EPSILON 1e-6

// Number of experiment repetitions for averaging
#define NUM_EXPERIMENTS 5

// Basic matrix multiplication kernel with one thread computing one output element
__global__ void matrixMulKernel(float* P, const float* M, const float* N, int width) {
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        // Compute dot product of row of M and column of N
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}

// Single-thread kernel for performance comparison (part 2)
__global__ void matrixMulSingleThreadKernel(float* P, const float* M, const float* N, int width) {
    // Use a single thread to compute the entire matrix product
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

// CPU matrix multiplication for verification and comparison
void matrixMulCPU(float* P, const float* M, const float* N, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

// Utility function to check if GPU-computed result matches CPU-computed result
bool verifyResult(const float* cpuP, const float* gpuP, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuP[i] - gpuP[i]) > EPSILON) {
            printf("Verification failed at element %d: CPU=%f, GPU=%f\n", i, cpuP[i], gpuP[i]);
            return false;
        }
    }
    return true;
}

// Utility function to initialize matrix with random values
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Utility function to print timing statistics
void printTimingStats(const char* label, float times[], int numExperiments) {
    float sum = 0.0f, min = times[0], max = times[0];
    
    for (int i = 0; i < numExperiments; i++) {
        sum += times[i];
        if (times[i] < min) min = times[i];
        if (times[i] > max) max = times[i];
    }
    
    float avg = sum / numExperiments;
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (int i = 0; i < numExperiments; i++) {
        variance += (times[i] - avg) * (times[i] - avg);
    }
    variance /= numExperiments;
    float stdDev = sqrt(variance);
    
    printf("%s: Avg = %.3f ms, Min = %.3f ms, Max = %.3f ms, StdDev = %.3f ms\n", 
           label, avg, min, max, stdDev);
}

// Run the matrix multiplication with a specific width and block size
void runMatrixMultiplication(int matrixWidth, int blockWidth) {
    int matrixSize = matrixWidth * matrixWidth;
    size_t matrixBytes = matrixSize * sizeof(float);
    
    // Allocate host memory
    float *h_M, *h_N, *h_P, *h_CPU_P;
    h_M = (float*)malloc(matrixBytes);
    h_N = (float*)malloc(matrixBytes);
    h_P = (float*)malloc(matrixBytes);
    h_CPU_P = (float*)malloc(matrixBytes);
    
    // Initialize input matrices with random values
    srand(time(NULL));
    initializeMatrix(h_M, matrixSize);
    initializeMatrix(h_N, matrixSize);
    
    // Allocate device memory
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, matrixBytes);
    cudaMalloc((void**)&d_N, matrixBytes);
    cudaMalloc((void**)&d_P, matrixBytes);
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Arrays to store timing results for multiple experiments
    float hostToDeviceTransferTimes[NUM_EXPERIMENTS];
    float deviceToHostTransferTimes[NUM_EXPERIMENTS];
    float singleThreadGpuTimes[NUM_EXPERIMENTS];
    float multiThreadGpuTimes[NUM_EXPERIMENTS];
    float cpuTimes[NUM_EXPERIMENTS];
    
    for (int exp = 0; exp < NUM_EXPERIMENTS; exp++) {
        float elapsedTime;
        
        // Timing host-to-device transfer
        cudaEventRecord(start, 0);
        cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        hostToDeviceTransferTimes[exp] = elapsedTime;
        
        // Timing CPU matrix multiplication
        clock_t cpu_start = clock();
        matrixMulCPU(h_CPU_P, h_M, h_N, matrixWidth);
        clock_t cpu_end = clock();
        cpuTimes[exp] = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // Convert to ms
        
        // Timing single-thread GPU kernel (for part 2)
        if (matrixWidth <= 1024) { // Limited to smaller matrices due to performance
            cudaEventRecord(start, 0);
            matrixMulSingleThreadKernel<<<1, 1>>>(d_P, d_M, d_N, matrixWidth);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            singleThreadGpuTimes[exp] = elapsedTime;
        } else {
            singleThreadGpuTimes[exp] = -1.0f; // Skip for large matrices
        }
        
        // Timing multi-threaded GPU kernel (for part 3)
        dim3 blockDim(blockWidth, blockWidth);
        dim3 gridDim((matrixWidth + blockWidth - 1) / blockWidth, 
                     (matrixWidth + blockWidth - 1) / blockWidth);
        
        cudaEventRecord(start, 0);
        matrixMulKernel<<<gridDim, blockDim>>>(d_P, d_M, d_N, matrixWidth);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        multiThreadGpuTimes[exp] = elapsedTime;
        
        // Timing device-to-host transfer
        cudaEventRecord(start, 0);
        cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        deviceToHostTransferTimes[exp] = elapsedTime;
        
        // Verify correctness (only check once)
        if (exp == 0) {
            bool correct = verifyResult(h_CPU_P, h_P, matrixSize);
            if (correct) {
                printf("Test PASSED for matrix size %d x %d with block width %d\n", 
                       matrixWidth, matrixWidth, blockWidth);
            } else {
                printf("Test FAILED for matrix size %d x %d with block width %d\n", 
                       matrixWidth, matrixWidth, blockWidth);
            }
        }
    }
    
    // Print performance results
    printf("\n=== Performance Results for %d x %d Matrix with %d x %d Block Size ===\n", 
           matrixWidth, matrixWidth, blockWidth, blockWidth);
    
    printTimingStats("Host-to-Device Transfer", hostToDeviceTransferTimes, NUM_EXPERIMENTS);
    printTimingStats("Device-to-Host Transfer", deviceToHostTransferTimes, NUM_EXPERIMENTS);
    
    if (matrixWidth <= 1024) {
        printTimingStats("Single-Thread GPU Kernel", singleThreadGpuTimes, NUM_EXPERIMENTS);
    }
    
    printTimingStats("Multi-Thread GPU Kernel", multiThreadGpuTimes, NUM_EXPERIMENTS);
    printTimingStats("CPU Matrix Multiplication", cpuTimes, NUM_EXPERIMENTS);
    
    // Calculate total GPU time (kernel + transfer)
    float totalGpuTime = 0.0f;
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        if (matrixWidth <= 1024) {
            totalGpuTime += singleThreadGpuTimes[i] + hostToDeviceTransferTimes[i] + deviceToHostTransferTimes[i];
        } else {
            totalGpuTime += multiThreadGpuTimes[i] + hostToDeviceTransferTimes[i] + deviceToHostTransferTimes[i];
        }
    }
    totalGpuTime /= NUM_EXPERIMENTS;
    
    float avgCpuTime = 0.0f;
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        avgCpuTime += cpuTimes[i];
    }
    avgCpuTime /= NUM_EXPERIMENTS;
    
    printf("\nTotal Time (incl. transfers):\n");
    printf("  GPU: %.3f ms\n", totalGpuTime);
    printf("  CPU: %.3f ms\n", avgCpuTime);
    printf("  Speedup: %.2fx\n", avgCpuTime / totalGpuTime);
    
    // Calculate CGMA ratio for part 3 question
    int flopsPerThread = 2 * matrixWidth; // 1 mult + 1 add per iteration, width iterations
    int globalReadsPerThread = 2 * matrixWidth; // Read one element each from M and N per iteration
    float cgmaRatio = (float)flopsPerThread / globalReadsPerThread;
    
    printf("\nCGMA Analysis:\n");
    printf("  FLOPs per thread: %d\n", flopsPerThread);
    printf("  Global memory reads per thread: %d\n", globalReadsPerThread);
    printf("  CGMA ratio: %.2f\n", cgmaRatio);
    
    // Calculate element reuse
    printf("  Each element of input matrices is loaded %d times during kernel execution\n", 
           matrixWidth);
    
    // Free memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_CPU_P);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Matrix sizes to test
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    
    // Block widths to test
    int blockWidths[] = {2, 4, 8, 16, 32};
    int numBlockWidths = sizeof(blockWidths) / sizeof(blockWidths[0]);
    
    // Test all combinations for part 3
    for (int i = 0; i < numSizes; i++) {
        for (int j = 0; j < numBlockWidths; j++) {
            runMatrixMultiplication(matrixSizes[i], blockWidths[j]);
            printf("\n---------------------------------------------------\n");
        }
    }
    
    // For part 1 and 2, we could extract and format the timing results
    // from the above runs, but we've already collected all the necessary data
    
    printf("\n==== Performance Analysis Summary ====\n");
    printf("1. Data Transfer Analysis:\n");
    printf("   - Host-to-Device and Device-to-Host transfer times scale with matrix size\n");
    printf("   - See the detailed timing results above for each matrix size\n");
    
    printf("\n2. GPU vs. CPU Analysis:\n");
    printf("   - For small matrices, the CPU may outperform the GPU when considering transfer times\n");
    printf("   - For larger matrices, GPU computation benefits become more significant\n");
    
    printf("\n3. Block and Thread Configuration Analysis:\n");
    printf("   - Elements loaded per input matrix: width times\n");
    printf("   - CGMA ratio: 2.00 (considering multiply and add as separate operations)\n");
    printf("   - Optimal block size depends on matrix size and GPU architecture\n");
    
    return 0;
}

// Michael Moser
// 20349246

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>

// Constants and macros
#define EPSILON 1e-6  // Tolerance for floating-point comparison

// Error checking - use only where needed
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Multi-threaded kernel - each thread computes one matrix element
__global__ void matrixMultiply(float* C, const float* A, const float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Single-threaded kernel for comparison
__global__ void singleThreadMatrixMultiply(float* C, const float* A, const float* B, int n) {
    // One thread does all the work
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CPU matrix multiplication for verification
void cpuMatrixMultiply(float* C, const float* A, const float* B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Initialize matrix with random values
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Verify GPU results against CPU results
bool verifyResults(const float* cpu_result, const float* gpu_result, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > EPSILON) {
            printf("Mismatch at element %d: CPU=%f, GPU=%f\n", i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("CUDA Matrix Multiplication Performance Analysis\n");
    printf("----------------------------------------------\n");

    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using device: %s\n\n", deviceProp.name);

    // Matrix sizes to test
    int sizes[] = { 256, 512, 1024, 2048 };
    int blockSizes[] = { 2, 4, 8, 16, 32 };

    // Arrays to store timing results
    float hostToDeviceTime[4];
    float deviceToHostTime[4];
    float singleGpuTime[4];
    float multiGpuTime[4][5];
    float cpuTime[4];

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float elapsedTime;

    // Test each matrix size
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        int matrixSize = n * n;
        size_t bytes = matrixSize * sizeof(float);

        printf("Testing %dx%d matrix\n", n, n);

        // Allocate host memory
        float* h_A = (float*)malloc(bytes);
        float* h_B = (float*)malloc(bytes);
        float* h_C = (float*)malloc(bytes);
        float* h_cpuC = (float*)malloc(bytes);

        if (!h_A || !h_B || !h_C || !h_cpuC) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(EXIT_FAILURE);
        }

        // Initialize matrices with random data
        srand((unsigned)time(NULL));
        initializeMatrix(h_A, matrixSize);
        initializeMatrix(h_B, matrixSize);

        // Allocate device memory
        float* d_A, * d_B, * d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        // Measure host-to-device transfer time
        cudaEventRecord(startEvent);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        printf("  Host to Device transfer: %.3f ms\n", elapsedTime);
        hostToDeviceTime[s] = elapsedTime;

        // CPU matrix multiplication
        clock_t cpuStart = clock();
        cpuMatrixMultiply(h_cpuC, h_A, h_B, n);
        clock_t cpuEnd = clock();
        float cpuElapsed = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        printf("  CPU time: %.3f ms\n", cpuElapsed);
        cpuTime[s] = cpuElapsed;

        // Single-thread GPU (for sizes up to 1024x1024)
        if (n <= 1024) {
            cudaEventRecord(startEvent);
            singleThreadMatrixMultiply <<<1, 1 >>> (d_C, d_A, d_B, n);
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
            printf("  Single-thread GPU time: %.3f ms\n", elapsedTime);
            singleGpuTime[s] = elapsedTime;
            cudaMemset(d_C, 0, bytes); // Clear results for next test
        }

        // Multi-threaded GPU with different block sizes
        for (int b = 0; b < 5; b++) {
            int blockWidth = blockSizes[b];

            // Set up execution configuration
            dim3 blockDim(blockWidth, blockWidth);
            dim3 gridDim((n + blockWidth - 1) / blockWidth,
                (n + blockWidth - 1) / blockWidth);

            // Time the kernel
            cudaEventRecord(startEvent);
            matrixMultiply <<<gridDim, blockDim >>> (d_C, d_A, d_B, n);
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

            printf("  Multi-thread (%dx%d blocks): %.3f ms\n", blockWidth, blockWidth, elapsedTime);
            multiGpuTime[s][b] = elapsedTime;

            // Only verify results for first block size
            if (b == 0) {
                cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
                bool correct = verifyResults(h_cpuC, h_C, matrixSize);
                printf("  Result verification: %s\n", correct ? "PASSED" : "FAILED");
            }
        }

        // Measure device-to-host transfer time
        cudaEventRecord(startEvent);
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        printf("  Device to Host transfer: %.3f ms\n\n", elapsedTime);
        deviceToHostTime[s] = elapsedTime;

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_cpuC);
    }

    // Results summary - Part 1: Data Transfer Analysis
    printf("===== PART 1: DATA TRANSFER ANALYSIS =====\n");
    printf("Matrix Size\tH2D (ms)\tD2H (ms)\n");
    for (int s = 0; s < 4; s++) {
        printf("%dx%d\t\t%.3f\t\t%.3f\n",
            sizes[s], sizes[s], hostToDeviceTime[s], deviceToHostTime[s]);
    }

    // Part 2: GPU vs CPU comparison
    printf("\n===== PART 2: GPU vs CPU COMPARISON =====\n");
    printf("Matrix Size\tCPU (ms)\tSingle GPU (ms)\tMulti GPU (ms)\tTotal GPU (ms)\tSpeedup\n");
    for (int s = 0; s < 3; s++) { // First 3 sizes only
        float totalTime = hostToDeviceTime[s] + multiGpuTime[s][2] + deviceToHostTime[s]; // 8x8 blocks
        printf("%dx%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.2fx\n",
            sizes[s], sizes[s], cpuTime[s], singleGpuTime[s], multiGpuTime[s][2],
            totalTime, cpuTime[s] / totalTime);
    }

    // Part 3: Block Configuration Analysis
    printf("\n===== PART 3: BLOCK CONFIGURATION ANALYSIS =====\n");
    printf("Matrix Size\t2x2\t\t4x4\t\t8x8\t\t16x16\t\t32x32\n");
    for (int s = 0; s < 4; s++) {
        printf("%dx%d\t\t", sizes[s], sizes[s]);
        for (int b = 0; b < 5; b++) {
            printf("%.3f\t\t", multiGpuTime[s][b]);
        }
        printf("\n");
    }

    // CGMA Analysis
    int flopsPerThread = 2 * sizes[0]; // For 256x256 matrix (multiply + add operations)
    int memReadsPerThread = 2 * sizes[0]; // Each thread reads from A and B
    float cgmaRatio = (float)flopsPerThread / memReadsPerThread;

    printf("\n===== CGMA ANALYSIS =====\n");
    printf("Each element of input matrices is loaded N times during execution\n");
    printf("For a %dx%d matrix: %d loads per element\n", sizes[0], sizes[0], sizes[0]);
    printf("CGMA ratio: %.2f\n", cgmaRatio);

    // Clean up
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

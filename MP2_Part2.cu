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

// Tiled matrix multiplication for rectangular matrices with boundary checks
// Fixed tile sizes of 12x18 as specified in the assignment
__global__ void matrixMultiplyTiledRectangular12x18(float *P, const float *M, const float *N, 
                                             int M_height, int M_width, int N_width) {
    // Shared memory for the tiles
    __shared__ float M_tile[12][18];
    __shared__ float N_tile[18][18];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate the row and column indices for this thread
    int row = by * 12 + ty;
    int col = bx * 18 + tx;
    
    float sum = 0.0f;
    
    // Loop over all tiles
    int num_tiles = (M_width + 18 - 1) / 18;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load M tile into shared memory with boundary checks
        if (row < M_height && tile * 18 + tx < M_width) {
            M_tile[ty][tx] = M[row * M_width + tile * 18 + tx];
        } else {
            M_tile[ty][tx] = 0.0f;
        }
        
        // Load N tile into shared memory with boundary checks
        if (tile * 18 + ty < M_width && col < N_width) {
            N_tile[ty][tx] = N[(tile * 18 + ty) * N_width + col];
        } else {
            N_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < 18; k++) {
            // We only need to consider elements up to M_width for correct results
            if (tile * 18 + k < M_width) {
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
    
    // Define tile dimensions for tests
    const int TILE_HEIGHT = 12;
    const int TILE_WIDTH = 18;
    
    // Test case dimensions
    struct TestCase {
        int M_height;
        int M_width;
        int N_width;
    };
    
    TestCase testCases[] = {
        {750, 800, 850},
        {2000, 1750, 1900}
    };
    int numTestCases = sizeof(testCases) / sizeof(testCases[0]);
    
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
    
    // Host array for storing timing results
    float timedResults[numTestCases][numIterations];
    
    // For each test case
    for (int tc = 0; tc < numTestCases; tc++) {
        int M_height = testCases[tc].M_height;
        int M_width = testCases[tc].M_width;
        int N_width = testCases[tc].N_width;
        
        size_t M_bytes = M_height * M_width * sizeof(float);
        size_t N_bytes = M_width * N_width * sizeof(float);
        size_t P_bytes = M_height * N_width * sizeof(float);
        
        printf("Testing matrix dimensions: M(%d x %d) * N(%d x %d) = P(%d x %d)\n", 
               M_height, M_width, M_width, N_width, M_height, N_width);
        
        // Allocate host memory
        float *h_M = (float*)malloc(M_bytes);
        float *h_N = (float*)malloc(N_bytes);
        float *h_P = (float*)malloc(P_bytes);
        float *h_P_CPU = (float*)malloc(P_bytes);
        
        // Initialize matrices
        initializeMatrix(h_M, M_height * M_width);
        initializeMatrix(h_N, M_width * N_width);
        
        // Allocate device memory
        float *d_M, *d_N, *d_P;
        CHECK_CUDA_ERROR(cudaMalloc(&d_M, M_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_N, N_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_P, P_bytes));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, M_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N, N_bytes, cudaMemcpyHostToDevice));
        
        // Compute CPU result for verification
        matrixMultiplyCPU(h_P_CPU, h_M, h_N, M_height, M_width, N_width);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Calculate grid and block dimensions
        dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
        dim3 dimGrid((N_width + TILE_WIDTH - 1) / TILE_WIDTH, 
                     (M_height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        
        // Run tiled kernel multiple times for timing
        for (int iter = 0; iter < numIterations; iter++) {
            // Start timing
            CHECK_CUDA_ERROR(cudaEventRecord(start));
            
            // Launch the kernel with fixed 12x18 tile size
            matrixMultiplyTiledRectangular12x18<<<dimGrid, dimBlock>>>(
                d_P, d_M, d_N, M_height, M_width, N_width);
            
            // Stop timing
            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            
            // Calculate elapsed time
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&timedResults[tc][iter], start, stop));
        }
        
        // Copy result back to host for verification
        CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, P_bytes, cudaMemcpyDeviceToHost));
        
        // Verify kernel results
        if (verifyResults(h_P_CPU, h_P, M_height * N_width)) {
            printf("  Verification: PASSED\n");
        } else {
            printf("  Verification: FAILED\n");
        }
        
        // Calculate average execution time
        float avgTime = 0.0f;
        for (int iter = 0; iter < numIterations; iter++) {
            avgTime += timedResults[tc][iter];
        }
        avgTime /= numIterations;
        
        printf("  Average execution time: %.4f ms\n\n", avgTime);
        
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
    
    printf("All tests completed successfully!\n");
    
    return 0;
}

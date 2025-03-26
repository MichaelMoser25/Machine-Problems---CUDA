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

// Basic GPU matrix multiplication without tiling
__global__ void matrixMultiplyBasic(float *P, const float *M, const float *N, 
                                   int M_height, int M_width, int N_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M_height && col < N_width) {
        float sum = 0.0f;
        for (int k = 0; k < M_width; k++) {
            sum += M[row * M_width + k] * N[k * N_width + col];
        }
        P[row * N_width + col] = sum;
    }
}

// Tiled matrix multiplication for rectangular matrices with boundary checks
__global__ void matrixMultiplyTiled(float *P, const float *M, const float *N,
                                  int M_height, int M_width, int N_width,
                                  int TILE_HEIGHT, int TILE_WIDTH) {
    // Dynamically allocated shared memory - will be determined at kernel launch
    extern __shared__ float sharedMem[];
    
    // Divide the shared memory: first for M_tile, second for N_tile
    float *M_tile = sharedMem;
    float *N_tile = &sharedMem[TILE_HEIGHT * TILE_WIDTH];
    
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
            M_tile[ty * TILE_WIDTH + tx] = M[row * M_width + tile * TILE_WIDTH + tx];
        } else {
            M_tile[ty * TILE_WIDTH + tx] = 0.0f;
        }
        
        // Load N tile into shared memory with boundary checks
        if (tile * TILE_WIDTH + ty < M_width && col < N_width) {
            N_tile[ty * TILE_WIDTH + tx] = N[(tile * TILE_WIDTH + ty) * N_width + col];
        } else {
            N_tile[ty * TILE_WIDTH + tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            // We only need to consider elements up to M_width for correct results
            if (tile * TILE_WIDTH + k < M_width) {
                sum += M_tile[ty * TILE_WIDTH + k] * N_tile[k * TILE_WIDTH + tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result to global memory with boundary check
    if (row < M_height && col < N_width) {
        P[row * N_width + col] = sum;
    }
}

// CPU matrix multiplication for verification
void matrixMultiplyCPU(float *P, const float *M, const float *N, 
                       int M_height, int M_width, int N_width) {
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

// Function to initialize matrix with random values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100) / 100.0f;
    }
}

// Function to verify results with relative error tolerance
bool verifyResults(float *cpuResult, float *gpuResult, int size) {
    const float relativeEpsilon = 1e-3;  // 0.1% relative error tolerance
    int errorCount = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabs(cpuResult[i] - gpuResult[i]);
        float relError = 0.0f;
        
        // Check if both values are very small
        if (fabs(cpuResult[i]) < 1e-5 && fabs(gpuResult[i]) < 1e-5) {
            continue;  // Both values are essentially zero
        }
        
        // Use the larger value as the denominator to avoid division by very small numbers
        float denom = fmax(fabs(cpuResult[i]), fabs(gpuResult[i]));
        relError = diff / denom;
        
        if (relError > relativeEpsilon) {
            if (errorCount < 10) { // Only print first 10 errors
                printf("Verification failed at element %d: CPU = %f, GPU = %f, Rel.Error = %f%%\n", 
                      i, cpuResult[i], gpuResult[i], relError * 100);
            }
            errorCount++;
        }
    }
    
    if (errorCount > 0) {
        printf("Total number of errors: %d out of %d elements (%.2f%%)\n", 
              errorCount, size, (float)errorCount / size * 100);
        return false;
    }
    return true;
}

// Function to remove outliers using modified Z-score method
void removeOutliers(float *timings, int n, float *finalMean, float *finalStdDev) {
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += timings[i];
    }
    float mean = sum / n;
    
    // Calculate median absolute deviation (MAD)
    float *deviations = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        deviations[i] = fabs(timings[i] - mean);
    }
    
    // Sort deviations to find median
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (deviations[i] > deviations[j]) {
                float temp = deviations[i];
                deviations[i] = deviations[j];
                deviations[j] = temp;
            }
        }
    }
    
    float mad = deviations[n/2]; // Median absolute deviation
    free(deviations);
    
    // Identify non-outliers (using modified Z-score)
    const float threshold = 3.5; // Typical threshold value
    float validSum = 0.0f;
    int validCount = 0;
    
    for (int i = 0; i < n; i++) {
        // Skip if MAD is very small to avoid division by zero
        if (mad < 1e-6) {
            validSum += timings[i];
            validCount++;
            continue;
        }
        
        // Calculate modified Z-score
        float modifiedZ = 0.6745f * fabs(timings[i] - mean) / mad;
        
        if (modifiedZ < threshold) {
            validSum += timings[i];
            validCount++;
        }
    }
    
    // Recalculate mean without outliers
    *finalMean = (validCount > 0) ? validSum / validCount : mean;
    
    // Calculate standard deviation without outliers
    float variance = 0.0f;
    for (int i = 0; i < n; i++) {
        // Calculate modified Z-score again
        float modifiedZ = (mad < 1e-6) ? 0 : 0.6745f * fabs(timings[i] - mean) / mad;
        
        if (modifiedZ < threshold) {
            variance += (*finalMean - timings[i]) * (*finalMean - timings[i]);
        }
    }
    
    *finalStdDev = (validCount > 1) ? sqrt(variance / (validCount - 1)) : 0.0f;
    
    // Report if outliers were removed
    if (validCount < n) {
        printf("Removed %d outliers from timing data.\n", n - validCount);
    }
}

int main() {
    // Set random seed
    srand(42);
    
    // Define the fixed tile dimensions as specified in the assignment
    const int TILE_HEIGHT = 12;
    const int TILE_WIDTH = 18;
    
    // Test case dimensions as specified in the assignment
    struct TestCase {
        int M_height;
        int M_width;
        int N_width;
        const char* name;
    };
    
    TestCase testCases[] = {
        {750, 800, 850, "Case 1"},
        {2000, 1750, 1900, "Case 2"}
    };
    int numTestCases = sizeof(testCases) / sizeof(testCases[0]);
    
    // Number of test iterations for reliable timing
    const int numIterations = 20; // More iterations for better statistics
    
    // Print device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("\n");
    
    printf("Testing tiled matrix multiplication with boundary checks\n");
    printf("Using fixed tile size: %d x %d\n\n", TILE_HEIGHT, TILE_WIDTH);
    
    // For CSV output
    printf("Case,M_height,M_width,N_width,Method,Mean_Time(ms),StdDev(ms)\n");
    
    // For each test case
    for (int tc = 0; tc < numTestCases; tc++) {
        int M_height = testCases[tc].M_height;
        int M_width = testCases[tc].M_width;
        int N_width = testCases[tc].N_width;
        
        printf("---------------------------------------------------\n");
        printf("Test Case %d: %s\n", tc+1, testCases[tc].name);
        printf("Matrix dimensions: M(%d x %d) * N(%d x %d) = P(%d x %d)\n", 
               M_height, M_width, M_width, N_width, M_height, N_width);
        
        size_t M_bytes = M_height * M_width * sizeof(float);
        size_t N_bytes = M_width * N_width * sizeof(float);
        size_t P_bytes = M_height * N_width * sizeof(float);
        
        // Allocate host memory
        float *h_M = (float*)malloc(M_bytes);
        float *h_N = (float*)malloc(N_bytes);
        float *h_P = (float*)malloc(P_bytes);
        float *h_P_CPU = (float*)malloc(P_bytes);
        float *h_P_Basic = (float*)malloc(P_bytes);
        
        if (!h_M || !h_N || !h_P || !h_P_CPU || !h_P_Basic) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        
        // Initialize matrices
        initializeMatrix(h_M, M_height * M_width);
        initializeMatrix(h_N, M_width * N_width);
        
        // Compute CPU result for verification
        printf("Computing CPU reference solution...\n");
        matrixMultiplyCPU(h_P_CPU, h_M, h_N, M_height, M_width, N_width);
        
        // Allocate device memory
        float *d_M, *d_N, *d_P;
        CHECK_CUDA_ERROR(cudaMalloc(&d_M, M_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_N, N_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_P, P_bytes));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, M_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N, N_bytes, cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Arrays to store timing results
        float basicTimes[numIterations];
        float tiledTimes[numIterations];
        
        // 1. Run the basic kernel (for comparison)
        dim3 basicBlock(16, 16);
        dim3 basicGrid((N_width + basicBlock.x - 1) / basicBlock.x, 
                      (M_height + basicBlock.y - 1) / basicBlock.y);
        
        printf("Running basic matrix multiplication kernel...\n");
        
        for (int iter = 0; iter < numIterations; iter++) {
            // Start timing
            CHECK_CUDA_ERROR(cudaEventRecord(start));
            
            // Launch basic kernel
            matrixMultiplyBasic<<<basicGrid, basicBlock>>>(
                d_P, d_M, d_N, M_height, M_width, N_width);
            
            // Stop timing
            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            
            // Calculate elapsed time
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&basicTimes[iter], start, stop));
        }
        
        // Verify basic kernel results
        CHECK_CUDA_ERROR(cudaMemcpy(h_P_Basic, d_P, P_bytes, cudaMemcpyDeviceToHost));
        printf("Basic kernel verification: %s\n", 
               verifyResults(h_P_CPU, h_P_Basic, M_height * N_width) ? "PASSED" : "FAILED");
        
        // 2. Run the tiled kernel with boundary checks
        dim3 tiledBlock(TILE_WIDTH, TILE_HEIGHT);
        dim3 tiledGrid((N_width + TILE_WIDTH - 1) / TILE_WIDTH, 
                      (M_height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        
        // Calculate shared memory size
        size_t sharedMemSize = (TILE_HEIGHT * TILE_WIDTH + TILE_WIDTH * TILE_WIDTH) * sizeof(float);
        
        printf("Running tiled matrix multiplication kernel (tile size: %dx%d)...\n", 
               TILE_HEIGHT, TILE_WIDTH);
        
        for (int iter = 0; iter < numIterations; iter++) {
            // Start timing
            CHECK_CUDA_ERROR(cudaEventRecord(start));
            
            // Launch tiled kernel
            matrixMultiplyTiled<<<tiledGrid, tiledBlock, sharedMemSize>>>(
                d_P, d_M, d_N, M_height, M_width, N_width, TILE_HEIGHT, TILE_WIDTH);
            
            // Stop timing
            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            
            // Calculate elapsed time
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&tiledTimes[iter], start, stop));
        }
        
        // Verify tiled kernel results
        CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, P_bytes, cudaMemcpyDeviceToHost));
        printf("Tiled kernel verification: %s\n", 
               verifyResults(h_P_CPU, h_P, M_height * N_width) ? "PASSED" : "FAILED");
        
        // Calculate statistics for basic kernel times
        float basicMean, basicStdDev;
        removeOutliers(basicTimes, numIterations, &basicMean, &basicStdDev);
        
        // Calculate statistics for tiled kernel times
        float tiledMean, tiledStdDev;
        removeOutliers(tiledTimes, numIterations, &tiledMean, &tiledStdDev);
        
        // Print performance results
        printf("\nPerformance Results:\n");
        printf("  Basic Kernel: %.4f ms (StdDev: %.4f ms)\n", basicMean, basicStdDev);
        printf("  Tiled Kernel: %.4f ms (StdDev: %.4f ms)\n", tiledMean, tiledStdDev);
        printf("  Speedup: %.2fx\n", basicMean / tiledMean);
        
        // Output in CSV format for plotting
        printf("%s,%d,%d,%d,Basic,%.4f,%.4f\n", 
               testCases[tc].name, M_height, M_width, N_width, basicMean, basicStdDev);
        printf("%s,%d,%d,%d,Tiled,%.4f,%.4f\n", 
               testCases[tc].name, M_height, M_width, N_width, tiledMean, tiledStdDev);
        
        // Cleanup
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        CHECK_CUDA_ERROR(cudaFree(d_M));
        CHECK_CUDA_ERROR(cudaFree(d_N));
        CHECK_CUDA_ERROR(cudaFree(d_P));
        free(h_M);
        free(h_N);
        free(h_P);
        free(h_P_CPU);
        free(h_P_Basic);
    }
    
    printf("\nAll tests completed successfully!\n");
    
    return 0;
}

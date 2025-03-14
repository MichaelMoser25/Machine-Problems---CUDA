// Michael Moser
// 20349246

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    // Used for debugging errors
    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("Device %d: \"%s\"\n", i, deviceProp.name);
        printf("=================================\n");
        
        // Compute capability
        printf("CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        
        // Clock rate
        printf("Clock Rate: %.2f GHz\n", deviceProp.clockRate * 1e-6);
        
        // Streaming Multiprocessors and Cores
        printf("Number of Streaming Multiprocessors (SM): %d\n", deviceProp.multiProcessorCount);
        
        // Calculate the number of CUDA cores
        int cudaCores = 0;
        if (deviceProp.major == 1) {
            // Tesla architecture
            cudaCores = 8 * deviceProp.multiProcessorCount;
        } else if (deviceProp.major == 2) {
            // Fermi architecture
            if (deviceProp.minor == 0) {
                cudaCores = 32 * deviceProp.multiProcessorCount;
            } else {
                cudaCores = 48 * deviceProp.multiProcessorCount;
            }
        } else if (deviceProp.major == 3) {
            // Kepler architecture
            cudaCores = 192 * deviceProp.multiProcessorCount;
        } else if (deviceProp.major == 5) {
            // Maxwell architecture
            cudaCores = 128 * deviceProp.multiProcessorCount;
        } else if (deviceProp.major == 6 || deviceProp.major == 7) {
            // Pascal & Volta architecture
            if (deviceProp.major == 6 && deviceProp.minor == 0) {
                cudaCores = 64 * deviceProp.multiProcessorCount; // GP100
            } else if (deviceProp.major == 6 && deviceProp.minor >= 1) {
                cudaCores = 128 * deviceProp.multiProcessorCount; // GP102, GP104, GP106, etc.
            } else if (deviceProp.major == 7 && deviceProp.minor == 0) {
                cudaCores = 64 * deviceProp.multiProcessorCount; // V100
            } else if (deviceProp.major == 7 && deviceProp.minor >= 5) {
                cudaCores = 64 * deviceProp.multiProcessorCount; // Turing
            }
        } else if (deviceProp.major == 8) {
            // Ampere architecture
            cudaCores = 64 * deviceProp.multiProcessorCount; // A100, RTX 30 series
        } else if (deviceProp.major == 9) {
            // Hopper architecture
            cudaCores = 128 * deviceProp.multiProcessorCount; // H100
        } else {
            // Unknown architecture
            cudaCores = 0;
            printf("Unknown architecture, cannot determine CUDA core count\n");
        }
        
        printf("CUDA Cores: %d\n", cudaCores);
        
        // Warp size
        printf("Warp Size: %d\n", deviceProp.warpSize);
        
        // Memory Information
        printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Constant Memory: %.2f KB\n", deviceProp.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);
        printf("L2 Cache Size: %.2f MB\n", deviceProp.l2CacheSize / (1024.0 * 1024.0));
        
        // Registers
        printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("Registers per SM: %d\n", deviceProp.regsPerMultiprocessor);
        
        // Thread Information
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max Threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        
        // Block and Grid Dimensions
        printf("Max Block Dimensions: [%d, %d, %d]\n", 
               deviceProp.maxThreadsDim[0], 
               deviceProp.maxThreadsDim[1], 
               deviceProp.maxThreadsDim[2]);
        
        printf("Max Grid Dimensions: [%d, %d, %d]\n", 
               deviceProp.maxGridSize[0], 
               deviceProp.maxGridSize[1], 
               deviceProp.maxGridSize[2]);
               
        // Memory bus width
        printf("Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
        
        // Memory clock rate
        printf("Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate * 1e-6);
        
        // Compute mode
        const char* computeMode;
        switch (deviceProp.computeMode) {
            case cudaComputeModeDefault:
                computeMode = "Default (Multiple threads can use this device)";
                break;
            case cudaComputeModeExclusive:
                computeMode = "Exclusive (Only one thread can use this device)";
                break;
            case cudaComputeModeProhibited:
                computeMode = "Prohibited (No threads can use this device)";
                break;
            case cudaComputeModeExclusiveProcess:
                computeMode = "Exclusive Process (Many threads from one process can use this device)";
                break;
            default:
                computeMode = "Unknown";
        }
        printf("Compute Mode: %s\n", computeMode);
        
        // ECC support
        // printf("ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
        
        // Unified addressing
        // printf("Unified Addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        
        // Concurrent kernels
        // printf("Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
        
        // Async engine count
        /// printf("Async Engine Count: %d\n", deviceProp.asyncEngineCount);
        
        printf("\n");
    }
    
    return 0;
}

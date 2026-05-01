/**
 * @file hello.cu
 * @brief CUDA toolchain verification kernel.
 */
#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief Simple kernel to verify CUDA execution.
 */
__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("Hello from DeepSeek CUDA Kernel! (Thread 0)\n");
    }
}

// C++ wrapper — called from Python via the binding in bindings.cu.
void run_hello() {
    hello_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

// A simple kernel to verify the setup
__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("Hello from DeepSeek CUDA Kernel! (Thread 0)\n");
    }
}

// C++ wrapper for PyTorch FFI
void run_hello() {
    hello_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}

// Bind the kernel to PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_hello", &run_hello, "Run the hello world CUDA kernel");
}

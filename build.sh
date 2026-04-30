#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Run CMake
# Use the current python's path to find the correct LibTorch
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') \
         -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10

# Build
make -j$(nproc)

echo "Build complete! Library is at build/libds_kernels.so"

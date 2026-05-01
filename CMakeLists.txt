cmake_minimum_required(VERSION 3.18)
project(DeepSeekOpt LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU Architectures: 86 (RTX 30 series), 89 (RTX 40 series)
# If the target GPU is different, CMake will attempt auto-detection if this is not set,
# but being explicit is better for performance.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89)
endif()

# Find PyTorch (LibTorch)
# The Python interpreter is used to locate the correct LibTorch prefix path.
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)


# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src/kernels/include
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Shared library for custom kernels.
# bindings.cu owns the single PYBIND11_MODULE entry point.
# Each kernel .cu file contains only the kernel + C++ wrapper (no module def).
add_library(ds_kernels SHARED
    src/kernels/hello.cu
    src/kernels/moe_routing.cu
    src/kernels/moe_scan.cu
    src/kernels/moe_dispatch.cu
    src/kernels/bindings.cu
)

# Target properties
set_target_properties(ds_kernels PROPERTIES
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)

# Define the extension name so PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) expands
# to PyInit_ds_kernels — the symbol Python's import system searches for.
target_compile_definitions(ds_kernels PRIVATE TORCH_EXTENSION_NAME=ds_kernels)

# Link against Torch
target_link_libraries(ds_kernels "${TORCH_LIBRARIES}")

# Target installation configuration
install(TARGETS ds_kernels DESTINATION lib)

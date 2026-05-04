/**
 * @file swiglu.h
 * @brief Header for the in-place SwiGLU activation kernel.
 *
 * Sits between GEMM1 and GEMM2 in the expert MLP fused pipeline.
 * Transforms a [total_tokens, 2*intermediate_size] tensor into a
 * [total_tokens, intermediate_size] tensor via:
 *     output[i, j] = silu(input[i, j]) * input[i, j + intermediate_size]
 * where silu(x) = x * sigmoid(x).
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

/**
 * @brief Launches the SwiGLU activation kernel.
 *
 * @param input             [in]  BF16 pointer [total_tokens, 2*intermediate_size]
 * @param output            [out] BF16 pointer [total_tokens, intermediate_size]
 * @param total_tokens      Total number of dispatched (token, expert) slots.
 * @param intermediate_size Half the inner dimension (1408 for DeepSeek-V2-Lite).
 * @param stream            CUDA stream for async execution.
 */
void launch_swiglu(
    const __nv_bfloat16* input,
    __nv_bfloat16*       output,
    int                  total_tokens,
    int                  intermediate_size,
    cudaStream_t         stream
);

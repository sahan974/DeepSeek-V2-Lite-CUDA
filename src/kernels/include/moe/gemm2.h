/**
 * @file gemm2.h
 * @brief Header for GEMM2: down_proj projection for all active experts.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/**
 * @brief Launches the GEMM2 down_proj kernel.
 *
 * For each expert e with tokens_e = offsets[e+1] - offsets[e] > 0:
 *   gemm2_out[offsets[e]:offsets[e+1]] = swiglu_out[offsets[e]:offsets[e+1]] @ w2[e]^T
 *
 * @param swiglu_out         BF16 [total_slots, intermediate_size]
 * @param w2                 BF16 [num_experts, hidden_size, intermediate_size]
 * @param gemm2_out          BF16 [total_slots, hidden_size]  (output)
 * @param expert_offsets_host int32 host pointer [num_experts + 1]
 * @param num_experts        Number of experts (64).
 * @param intermediate_size  Input feature dimension (1408).
 * @param hidden_size        Output feature dimension (2048).
 * @param stream             CUDA stream.
 */
void launch_moe_gemm2(
    const __nv_bfloat16* swiglu_out,
    const __nv_bfloat16* w2,
    __nv_bfloat16*       gemm2_out,
    const int*           expert_offsets_host,
    int                  num_experts,
    int                  intermediate_size,
    int                  hidden_size,
    cudaStream_t         stream
);

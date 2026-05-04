/**
 * @file gemm1.h
 * @brief Header for GEMM1: fused gate_proj + up_proj batched over all experts.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

/**
 * @brief Launches the GEMM1 kernel.
 *
 * For each expert e with tokens_e = offsets[e+1] - offsets[e] > 0:
 *   gemm1_out[offsets[e]:offsets[e+1]] = dispatched[offsets[e]:offsets[e+1]] @ packed_w1[e]^T
 *
 * @param dispatched        BF16 [total_slots, hidden_size]
 * @param packed_w1         BF16 [num_experts, 2*intermediate_size, hidden_size]
 * @param gemm1_out         BF16 [total_slots, 2*intermediate_size]  (output)
 * @param expert_offsets    int32 host pointer [num_experts + 1]
 * @param num_experts       Number of experts (64).
 * @param hidden_size       Input feature dimension (2048).
 * @param double_intermed   Output feature dimension (2 * 1408 = 2816).
 * @param stream            CUDA stream.
 */
void launch_moe_gemm1(
    const __nv_bfloat16* dispatched,
    const __nv_bfloat16* packed_w1,
    __nv_bfloat16*       gemm1_out,
    const int*           expert_offsets_host,
    int                  num_experts,
    int                  hidden_size,
    int                  double_intermed,
    cudaStream_t         stream
);

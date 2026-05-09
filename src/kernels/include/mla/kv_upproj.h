/**
 * @file kv_upproj.h
 * @brief KV up-projection kernel declaration for MLA attention.
 *
 * Applies kv_b_proj: normed_kv [rows, 512] -> kv_up [rows, 4096]
 * using a single cublasGemmEx call with Tensor Core acceleration.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

/**
 * @brief Host launcher: single cuBLAS GEMM for the full KV up-projection.
 *
 * Computes: kv_up[M, N] = normed_kv[M, K] @ weight[N, K]^T
 *   M = rows  (bsz * seq_len)
 *   K = 512   (kv_lora_rank)
 *   N = 4096  (num_heads * (qk_nope_head_dim + v_head_dim))
 *
 * @param normed_kv  BF16 pointer [M, K]  — normalized KV latent
 * @param weight     BF16 pointer [N, K]  — kv_b_proj.weight (nn.Linear layout)
 * @param kv_up      BF16 pointer [M, N]  — output buffer (pre-allocated)
 * @param rows        M dimension
 * @param kv_lora_rank  K dimension (512)
 * @param out_dim    N dimension (4096)
 * @param stream     CUDA stream
 */
void launch_kv_upproj(
    const __nv_bfloat16* normed_kv,
    const __nv_bfloat16* weight,
    __nv_bfloat16*       kv_up,
    int                  rows,
    int                  kv_lora_rank,
    int                  out_dim,
    cudaStream_t         stream
);

/**
 * @brief PyTorch extension entry point.
 *
 * @param normed_kv  BF16 CUDA tensor [bsz, seq, kv_lora_rank]  or [rows, kv_lora_rank]
 * @param weight     BF16 CUDA tensor [out_dim, kv_lora_rank]    (kv_b_proj.weight)
 * @return           BF16 CUDA tensor [..., out_dim]             (same leading dims as input)
 */
torch::Tensor kv_upproj_forward(
    torch::Tensor normed_kv,
    torch::Tensor weight
);

/**
 * @file rms_norm.h
 * @brief RMSNorm CUDA kernel declaration for MLA attention pipeline.
 *
 * Applied as kv_a_layernorm over the compressed KV latent [bsz*seq, 512]
 * before the kv_b_proj up-projection step.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

/**
 * @brief Host launcher: applies RMSNorm over the last dimension.
 *
 * @param input      BF16 pointer [rows, hidden_size]
 * @param weight     Float32 pointer [hidden_size] — learned scale parameter
 * @param output     BF16 pointer [rows, hidden_size]
 * @param rows        Number of rows (bsz * seq_len)
 * @param hidden_size Normalisation dimension (512 for kv_a_layernorm)
 * @param eps         Variance epsilon for numerical stability
 * @param stream      CUDA stream
 */
void launch_rms_norm(
    const __nv_bfloat16* input,
    const float*          weight,
    __nv_bfloat16*        output,
    int                   rows,
    int                   hidden_size,
    float                 eps,
    cudaStream_t          stream
);

/**
 * @brief PyTorch extension entry point.
 *
 * @param input  BF16 CUDA tensor  [..., hidden_size]  (any leading dims, contiguous)
 * @param weight Float32 CUDA tensor [hidden_size]
 * @param eps    Variance epsilon (default 1e-6)
 * @return       BF16 CUDA tensor [..., hidden_size]
 */
torch::Tensor rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double        eps
);

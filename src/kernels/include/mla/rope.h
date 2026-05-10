/**
 * @file rope.h
 * @brief YaRN RoPE CUDA kernel declaration for MLA attention.
 *
 * Applies rotary position embeddings to q_pe and k_pe (the 64-dim rope
 * parts of Q and K). Matches the reference apply_rotary_pos_emb() exactly,
 * including the interleave-to-non-interleave transform applied before rotation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

/**
 * @brief Host launcher: applies RoPE to a single 2D BF16 tensor.
 *
 * The caller flattens [bsz, heads, seq, rope_dim] → [total_rows, rope_dim]
 * and pre-indexes + broadcasts cos/sin to the same shape before calling.
 *
 * @param input      BF16 pointer [total_rows, rope_dim] — in interleaved format
 * @param cos_table  BF16 pointer [total_rows, rope_dim] — pre-indexed, contiguous
 * @param sin_table  BF16 pointer [total_rows, rope_dim] — pre-indexed, contiguous
 * @param output     BF16 pointer [total_rows, rope_dim] — output (may alias input)
 * @param total_rows number of token × head rows
 * @param rope_dim   size of last dimension (64 for DeepSeek-V2-Lite)
 * @param stream     CUDA stream
 */
void launch_rope(
    const __nv_bfloat16* input,
    const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table,
    __nv_bfloat16*       output,
    int                  total_rows,
    int                  rope_dim,
    cudaStream_t         stream
);

/**
 * @brief PyTorch extension entry point.
 *
 * Applies YaRN RoPE to q_pe [bsz, heads, seq, rope_dim] and
 * k_pe [bsz, 1, seq, rope_dim] using pre-indexed cos/sin tables.
 *
 * The Python caller must index the cos/sin tables by position_ids before
 * calling:
 *   cos = cos_cached[position_ids].unsqueeze(1)  # [bsz, 1, seq, rope_dim]
 *   sin = sin_cached[position_ids].unsqueeze(1)  # [bsz, 1, seq, rope_dim]
 *
 * @param q_pe  BF16 CUDA tensor [bsz, heads, seq, rope_dim]
 * @param k_pe  BF16 CUDA tensor [bsz, 1,     seq, rope_dim]
 * @param cos   BF16 CUDA tensor [bsz, 1,     seq, rope_dim]
 * @param sin   BF16 CUDA tensor [bsz, 1,     seq, rope_dim]
 * @return pair: (q_pe_rotated, k_pe_rotated), same shapes as inputs
 */
std::vector<torch::Tensor> rope_forward(
    torch::Tensor q_pe,
    torch::Tensor k_pe,
    torch::Tensor cos,
    torch::Tensor sin
);

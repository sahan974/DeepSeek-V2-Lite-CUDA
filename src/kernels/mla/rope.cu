/**
 * @file rope.cu
 * @brief YaRN RoPE CUDA kernel for MLA attention.
 *
 * Position in the MLA pipeline:
 *   q_pe [bsz, 16, seq, 64]   k_pe [bsz, 1, seq, 64]
 *       │                           │
 *       └─────── RoPE (this file) ──┘
 *                       │
 *       q_pe_rot [bsz, 16, seq, 64]   k_pe_rot [bsz, 1, seq, 64]
 *
 * Reference implementation (apply_rotary_pos_emb in mla.py):
 *
 *   Step 1 — interleave-to-non-interleave transform:
 *     q = q.view(b, h, s, d//2, 2).transpose(4, 3).reshape(b, h, s, d)
 *     This maps q[..., 2k] → deinterleaved[..., k]
 *              q[..., 2k+1] → deinterleaved[..., k + d//2]
 *
 *   Step 2 — rotate:
 *     q_embed = q_deinterleaved * cos + rotate_half(q_deinterleaved) * sin
 *     rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])
 *
 * Combining steps 1 and 2 per output element:
 *   For k in [0, d//2):
 *     out[k]       = q[2k]   * cos[k]       - q[2k+1] * sin[k]
 *     out[k + d//2] = q[2k+1] * cos[k+d//2] + q[2k]   * sin[k+d//2]
 *
 * Kernel design:
 *   - Input/output are 2D [total_rows, rope_dim] (caller flattens leading dims).
 *   - cos/sin are [total_rows, rope_dim] (caller pre-indexes by position_ids
 *     and broadcasts across heads).
 *   - One block per row, rope_dim/2 threads per block (32 for rope_dim=64).
 *   - Thread k writes output[k] and output[k + rope_dim/2] from pair (input[2k], input[2k+1]).
 *   - All arithmetic is in float32; inputs/outputs are BF16.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "mla/rope.h"

// One block per row, rope_dim/2 threads per block.
// For rope_dim=64: 32 threads, each writing 2 output elements.
static constexpr int ROPE_HALF = 32;  // rope_dim / 2

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/**
 * @brief Applies RoPE to one row of the input tensor.
 *
 * Grid:  total_rows blocks
 * Block: rope_half threads  (= rope_dim / 2)
 *
 * Thread k computes output[row, k] and output[row, k + rope_half] using the
 * pair (input[row, 2k], input[row, 2k+1]) and cos/sin at positions k and k+rope_half.
 */
__global__ void rope_kernel(
    const __nv_bfloat16* __restrict__ input,      // [total_rows, rope_dim]
    const __nv_bfloat16* __restrict__ cos_table,  // [total_rows, rope_dim]
    const __nv_bfloat16* __restrict__ sin_table,  // [total_rows, rope_dim]
    __nv_bfloat16*       __restrict__ output,     // [total_rows, rope_dim]
    int rope_dim                                   // 64 for DeepSeek-V2-Lite
) {
    const int row  = blockIdx.x;
    const int k    = threadIdx.x;           // k in [0, rope_half)
    const int half = rope_dim / 2;

    if (k >= half) return;  // Guard — should never trigger for rope_dim=64, ROPE_HALF=32.

    // Row-base pointer offsets.
    const long long base = (long long)row * rope_dim;

    // Load interleaved pair for this thread.
    float x0 = __bfloat162float(input[base + 2 * k]);
    float x1 = __bfloat162float(input[base + 2 * k + 1]);

    // Load cos/sin for the two output positions k and k+half.
    float c_lo = __bfloat162float(cos_table[base + k]);
    float s_lo = __bfloat162float(sin_table[base + k]);
    float c_hi = __bfloat162float(cos_table[base + k + half]);
    float s_hi = __bfloat162float(sin_table[base + k + half]);

    // Apply rotation.
    // out[k]      = x0 * cos[k]      - x1 * sin[k]
    // out[k+half] = x1 * cos[k+half] + x0 * sin[k+half]
    float out_lo = x0 * c_lo - x1 * s_lo;
    float out_hi = x1 * c_hi + x0 * s_hi;

    output[base + k]        = __float2bfloat16_rn(out_lo);
    output[base + k + half] = __float2bfloat16_rn(out_hi);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void launch_rope(
    const __nv_bfloat16* input,
    const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table,
    __nv_bfloat16*       output,
    int                  total_rows,
    int                  rope_dim,
    cudaStream_t         stream
) {
    if (rope_dim % 2 != 0) {
        throw std::runtime_error("rope: rope_dim must be even.");
    }
    if (total_rows <= 0) return;

    const int threads = rope_dim / 2;
    rope_kernel<<<total_rows, threads, 0, stream>>>(
        input, cos_table, sin_table, output, rope_dim
    );
}

// ---------------------------------------------------------------------------
// PyTorch extension entry point
// ---------------------------------------------------------------------------

/**
 * @brief Applies YaRN RoPE to q_pe and k_pe using pre-indexed cos/sin.
 *
 * The caller must prepare cos and sin as [bsz, 1, seq, rope_dim] BF16 tensors
 * by indexing the cached tables with position_ids:
 *   cos = cos_cached[position_ids].unsqueeze(1).to(torch.bfloat16)
 *   sin = sin_cached[position_ids].unsqueeze(1).to(torch.bfloat16)
 *
 * @param q_pe  BF16 CUDA tensor [bsz, heads, seq, rope_dim]  contiguous
 * @param k_pe  BF16 CUDA tensor [bsz, 1,     seq, rope_dim]  contiguous
 * @param cos   BF16 CUDA tensor [bsz, 1,     seq, rope_dim]  contiguous
 * @param sin   BF16 CUDA tensor [bsz, 1,     seq, rope_dim]  contiguous
 * @return      {q_pe_rotated, k_pe_rotated}  same shapes as inputs, BF16
 */
std::vector<torch::Tensor> rope_forward(
    torch::Tensor q_pe,
    torch::Tensor k_pe,
    torch::Tensor cos,
    torch::Tensor sin
) {
    // ── Validation ──────────────────────────────────────────────────────────
    for (const auto& t : {q_pe, k_pe, cos, sin}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(),
                    "rope: all inputs must be contiguous CUDA tensors.");
        TORCH_CHECK(t.dtype() == torch::kBFloat16,
                    "rope: all inputs must be bfloat16.");
    }
    TORCH_CHECK(q_pe.dim() == 4, "rope: q_pe must be 4-D [bsz, heads, seq, rope_dim].");
    TORCH_CHECK(k_pe.dim() == 4, "rope: k_pe must be 4-D [bsz, 1, seq, rope_dim].");
    TORCH_CHECK(cos.dim()  == 4, "rope: cos must be 4-D [bsz, 1, seq, rope_dim].");
    TORCH_CHECK(sin.dim()  == 4, "rope: sin must be 4-D [bsz, 1, seq, rope_dim].");

    const int bsz      = q_pe.size(0);
    const int heads    = q_pe.size(1);
    const int seq      = q_pe.size(2);
    const int rope_dim = q_pe.size(3);

    TORCH_CHECK(k_pe.size(0) == bsz && k_pe.size(2) == seq && k_pe.size(3) == rope_dim,
                "rope: k_pe shape [", k_pe.size(0), ",", k_pe.size(1), ",",
                k_pe.size(2), ",", k_pe.size(3), "] incompatible with q_pe.");
    TORCH_CHECK(cos.size(0) == bsz && cos.size(2) == seq && cos.size(3) == rope_dim,
                "rope: cos shape incompatible.");
    TORCH_CHECK(sin.size(0) == bsz && sin.size(2) == seq && sin.size(3) == rope_dim,
                "rope: sin shape incompatible.");
    TORCH_CHECK(rope_dim % 2 == 0, "rope: rope_dim must be even.");

    // ── Allocate outputs ────────────────────────────────────────────────────
    auto q_rot = torch::empty_like(q_pe);
    auto k_rot = torch::empty_like(k_pe);

    // ── Apply RoPE to q_pe ──────────────────────────────────────────────────
    // Flatten: [bsz, heads, seq, rope_dim] → [bsz*heads*seq, rope_dim]
    // Broadcast cos/sin: [bsz, 1, seq, rope_dim] → [bsz, heads, seq, rope_dim]
    // then flatten.
    auto cos_q = cos.expand({bsz, heads, seq, rope_dim}).contiguous();
    auto sin_q = sin.expand({bsz, heads, seq, rope_dim}).contiguous();

    const int q_rows = bsz * heads * seq;
    launch_rope(
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cos_q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(sin_q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_rot.data_ptr()),
        q_rows,
        rope_dim,
        /*stream=*/0
    );

    // ── Apply RoPE to k_pe ──────────────────────────────────────────────────
    // k_pe has 1 head, cos/sin are [bsz, 1, seq, rope_dim] — already the right shape.
    const int k_rows = bsz * 1 * seq;
    launch_rope(
        reinterpret_cast<const __nv_bfloat16*>(k_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_rot.data_ptr()),
        k_rows,
        rope_dim,
        /*stream=*/0
    );

    return {q_rot, k_rot};
}

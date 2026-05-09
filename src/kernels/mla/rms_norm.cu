/**
 * @file rms_norm.cu
 * @brief RMSNorm CUDA kernel for the MLA kv_a_layernorm step.
 *
 * Position in the MLA pipeline:
 *   compressed_kv [bsz, seq, 512]    (raw output of kv_a_proj)
 *       │
 *       └── kv_a_layernorm ──► normed_kv [bsz, seq, 512]
 *                                   │
 *                                   └── kv_b_proj ──► [bsz, seq, 4096]
 *
 * Operation (matches reference RMSNorm.forward exactly):
 *   variance = mean(x^2, dim=-1, keepdim=True)
 *   x_norm   = x * rsqrt(variance + eps)
 *   output   = weight * x_norm
 *
 * Implementation details:
 *   - Grid: one block per row (rows = bsz * seq_len).
 *   - Block: RMS_NORM_THREADS threads (256).
 *   - Vectorized BF16 I/O via __nv_bfloat162 (2 elements per load/store).
 *   - Variance computed in float32 via warp shuffles + shared memory reduction.
 *   - hidden_size=512 → 256 bfloat162 pairs per row, one pair per thread
 *     at the default block size — zero-overhead stride loop.
 *   - Weight is float32 (nn.Parameter default) and applied in float before
 *     the final BF16 downcast.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <stdexcept>
#include "mla/rms_norm.h"

static constexpr int RMS_NORM_THREADS = 256;
static constexpr int MAX_WARPS        = (RMS_NORM_THREADS + 31) / 32;  // 8

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/**
 * @brief RMSNorm kernel: one block per token row.
 *
 * Grid:  rows  blocks   (bsz * seq_len)
 * Block: RMS_NORM_THREADS threads
 */
__global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,   // [rows, hidden_size]
    const float*          __restrict__ weight,  // [hidden_size]   float32
    __nv_bfloat16*        __restrict__ output,  // [rows, hidden_size]
    int hidden_size,
    float eps
) {
    const int row      = blockIdx.x;
    const int tid      = threadIdx.x;
    const int vec_width = hidden_size / 2;   // hidden_size must be even

    // Vectorised row pointers.
    const __nv_bfloat162* in_row  =
        reinterpret_cast<const __nv_bfloat162*>(input  + (long long)row * hidden_size);
    __nv_bfloat162*       out_row =
        reinterpret_cast<__nv_bfloat162*>(output + (long long)row * hidden_size);

    // ── Step 1: accumulate local sum-of-squares ──────────────────────────────
    float local_ss = 0.0f;

    for (int v = tid; v < vec_width; v += blockDim.x) {
        __nv_bfloat162 val2 = in_row[v];
        float x0 = __bfloat162float(val2.x);
        float x1 = __bfloat162float(val2.y);
        local_ss += x0 * x0 + x1 * x1;
    }

    // ── Step 2: warp-level reduction via shuffle ──────────────────────────────
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_ss += __shfl_down_sync(0xFFFFFFFF, local_ss, offset);
    }

    // ── Step 3: cross-warp reduction via shared memory ────────────────────────
    __shared__ float warp_sums[MAX_WARPS];
    __shared__ float inv_rms;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = local_ss;
    }
    __syncthreads();

    // Thread 0 reduces all warp partial sums.
    if (tid == 0) {
        float total_ss  = 0.0f;
        const int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            total_ss += warp_sums[w];
        }
        inv_rms = rsqrtf(total_ss / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    // ── Step 4: normalize and apply weight ────────────────────────────────────
    const float scale = inv_rms;

    for (int v = tid; v < vec_width; v += blockDim.x) {
        __nv_bfloat162 val2 = in_row[v];

        float x0 = __bfloat162float(val2.x) * scale;
        float x1 = __bfloat162float(val2.y) * scale;

        // Weight is float32 [hidden_size]. Scalar index into it.
        float w0 = weight[v * 2];
        float w1 = weight[v * 2 + 1];

        __nv_bfloat162 out2;
        out2.x = __float2bfloat16_rn(x0 * w0);
        out2.y = __float2bfloat16_rn(x1 * w1);
        out_row[v] = out2;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void launch_rms_norm(
    const __nv_bfloat16* input,
    const float*          weight,
    __nv_bfloat16*        output,
    int                   rows,
    int                   hidden_size,
    float                 eps,
    cudaStream_t          stream
) {
    if (hidden_size % 2 != 0) {
        throw std::runtime_error(
            "rms_norm: hidden_size must be even for bfloat162 vectorisation.");
    }
    if (rows <= 0) return;

    rms_norm_kernel<<<rows, RMS_NORM_THREADS, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

// ---------------------------------------------------------------------------
// PyTorch extension entry point
// ---------------------------------------------------------------------------

/**
 * @brief Applies RMSNorm over the last dimension of a BF16 tensor.
 *
 * @param input  BF16 CUDA tensor  [..., hidden_size]   (any leading dims, contiguous)
 * @param weight Float32 CUDA tensor [hidden_size]       (kv_a_layernorm.weight)
 * @param eps    Variance epsilon (default 1e-6)
 * @return       BF16 CUDA tensor  [..., hidden_size]
 */
torch::Tensor rms_norm_forward(torch::Tensor input, torch::Tensor weight, double eps) {
    TORCH_CHECK(input.is_cuda(),       "rms_norm: input must be a CUDA tensor.");
    TORCH_CHECK(weight.is_cuda(),      "rms_norm: weight must be a CUDA tensor.");
    TORCH_CHECK(input.is_contiguous(), "rms_norm: input must be contiguous.");
    TORCH_CHECK(input.dtype()  == torch::kBFloat16,
                "rms_norm: input must be bfloat16.");
    TORCH_CHECK(weight.dtype() == torch::kFloat32,
                "rms_norm: weight must be float32 (default nn.Parameter dtype).");
    TORCH_CHECK(input.dim() >= 2,
                "rms_norm: input must have at least 2 dimensions.");

    const int hidden_size = static_cast<int>(input.size(-1));
    TORCH_CHECK(weight.numel() == hidden_size,
                "rms_norm: weight size ", weight.numel(),
                " does not match input hidden_size ", hidden_size, ".");
    TORCH_CHECK(hidden_size % 2 == 0,
                "rms_norm: hidden_size must be even for bfloat162 vectorisation.");

    // Flatten leading dims so the kernel sees [rows, hidden_size].
    const int rows = static_cast<int>(input.numel() / hidden_size);

    auto output = torch::empty_like(input);

    launch_rms_norm(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows,
        hidden_size,
        static_cast<float>(eps),
        /*stream=*/0
    );

    return output;
}

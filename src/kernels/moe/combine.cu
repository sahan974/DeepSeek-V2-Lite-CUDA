/**
 * @file moe_combine.cu
 * @brief Token Combine (Gather) Kernel for Mixture-of-Experts routing.
 *
 * Reverses the expert-contiguous ordering produced by the dispatch kernel.
 * For each dispatched slot, the expert output is scaled by its gating weight
 * and atomically accumulated into the corresponding position in the final
 * output tensor, indexed by the original token index stored in token_map.
 *
 * The operation performed per slot s:
 *   final_output[token_map[s], :] += expert_output[s, :] * dispatched_weights[s]
 *
 * Implementation Details:
 * - Accumulation in FP32: Rather than accumulating in BF16 (which causes
 *   cascading rounding errors with TOP_K=6 additions), use a temporary
 *   FP32 accumulation buffer with native atomicAdd (supported on all GPUs).
 *   A second lightweight kernel converts the FP32 result to BF16.
 * - Thread Mapping: One thread block per dispatched slot. All threads in the
 *   block cooperate to cover hidden_size elements.
 * - Zeroed Output: The FP32 accumulation buffer is zero-initialized internally.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include "moe/combine.h"

static constexpr int COMBINE_THREADS = 256;

/**
 * @brief Accumulates weighted expert outputs into a FP32 buffer using native atomicAdd.
 *
 * Each block processes one dispatched slot. All threads cooperate to cover
 * hidden_size elements. Accumulation is done in FP32 to avoid BF16 precision loss.
 */
__global__ void moe_combine_accumulate_kernel(
    const __nv_bfloat16* __restrict__ expert_output,     // [total_slots, hidden_size]
    float*                            accum_output,      // [num_tokens, hidden_size], zeroed, FP32
    const int*           __restrict__ token_map,         // [total_slots]
    const float*         __restrict__ dispatched_weights,// [total_slots]
    int total_slots,
    int hidden_size
) {
    const int slot_id = blockIdx.x;
    if (slot_id >= total_slots) return;

    const int   token_id = token_map[slot_id];
    const float weight   = dispatched_weights[slot_id];

    const __nv_bfloat16* src = expert_output + static_cast<ptrdiff_t>(slot_id)  * hidden_size;
    float*               dst = accum_output  + static_cast<ptrdiff_t>(token_id) * hidden_size;

    // Each thread covers a strided subset of hidden_size elements.
    // Accumulate in FP32: atomicAdd is natively supported for float on all GPUs.
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float scaled = __bfloat162float(src[i]) * weight;
        atomicAdd(&dst[i], scaled);
    }
}

/**
 * @brief Casts a FP32 buffer to BF16 element-wise.
 */
__global__ void cast_fp32_to_bf16_kernel(
    const float*    __restrict__ src,  // [N]
    __nv_bfloat16*               dst,  // [N]
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

/**
 * @brief Internal host-side launcher — kept for header compatibility only.
 */
void launch_moe_combine(
    const __nv_bfloat16* expert_output,
    __nv_bfloat16*       final_output,
    const int*           token_map,
    const float*         dispatched_weights,
    int                  total_slots,
    int                  hidden_size,
    cudaStream_t         stream
) {
    (void)expert_output; (void)final_output; (void)token_map;
    (void)dispatched_weights; (void)total_slots; (void)hidden_size; (void)stream;
    throw std::runtime_error(
        "launch_moe_combine: use moe_combine_forward via PyTorch extension instead.");
}

/**
 * @brief PyTorch extension entry point for the token combine kernel.
 *
 * Accumulates weighted expert outputs in FP32 (via native atomicAdd), then
 * casts the result back to BF16. This eliminates cascading BF16 rounding errors.
 *
 * @param expert_output      BF16 CUDA tensor [total_slots, hidden_size].
 * @param token_map          int32 CUDA tensor [total_slots].
 * @param dispatched_weights float32 CUDA tensor [total_slots].
 * @param num_tokens         Number of original tokens (determines output height).
 * @return                   BF16 CUDA tensor [num_tokens, hidden_size].
 */
torch::Tensor moe_combine_forward(
    torch::Tensor expert_output,
    torch::Tensor token_map,
    torch::Tensor dispatched_weights,
    int           num_tokens
) {
    TORCH_CHECK(expert_output.is_cuda(),
                "expert_output must reside on a CUDA device.");
    TORCH_CHECK(expert_output.is_contiguous(),
                "expert_output must be contiguous in memory.");
    TORCH_CHECK(expert_output.dtype() == torch::kBFloat16,
                "expert_output must be of type bfloat16.");
    TORCH_CHECK(expert_output.dim() == 2,
                "expert_output must be a 2-D tensor [total_slots, hidden_size].");

    TORCH_CHECK(token_map.is_cuda(),
                "token_map must reside on a CUDA device.");
    TORCH_CHECK(token_map.is_contiguous(),
                "token_map must be contiguous in memory.");
    TORCH_CHECK(token_map.dtype() == torch::kInt32,
                "token_map must be of type int32.");
    TORCH_CHECK(token_map.dim() == 1,
                "token_map must be a 1-D tensor [total_slots].");

    TORCH_CHECK(dispatched_weights.is_cuda(),
                "dispatched_weights must reside on a CUDA device.");
    TORCH_CHECK(dispatched_weights.is_contiguous(),
                "dispatched_weights must be contiguous in memory.");
    TORCH_CHECK(dispatched_weights.dtype() == torch::kFloat32,
                "dispatched_weights must be of type float32.");
    TORCH_CHECK(dispatched_weights.dim() == 1,
                "dispatched_weights must be a 1-D tensor [total_slots].");

    const int total_slots = expert_output.size(0);
    const int hidden_size = expert_output.size(1);

    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive.");

    // FP32 accumulation buffer — eliminates BF16 cascading rounding errors.
    auto fp32_opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(expert_output.device());
    auto accum = torch::zeros({num_tokens, hidden_size}, fp32_opts);

    // Step 1: Accumulate weighted expert outputs in FP32.
    moe_combine_accumulate_kernel<<<total_slots, COMBINE_THREADS, 0, 0>>>(
        reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
        accum.data_ptr<float>(),
        token_map.data_ptr<int>(),
        dispatched_weights.data_ptr<float>(),
        total_slots,
        hidden_size
    );

    // Step 2: Cast FP32 → BF16.
    auto bf16_opts = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(expert_output.device());
    auto final_output = torch::empty({num_tokens, hidden_size}, bf16_opts);

    const int N           = num_tokens * hidden_size;
    const int cast_blocks = (N + COMBINE_THREADS - 1) / COMBINE_THREADS;
    cast_fp32_to_bf16_kernel<<<cast_blocks, COMBINE_THREADS, 0, 0>>>(
        accum.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(final_output.data_ptr()),
        N
    );

    return final_output;
}

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
 * - Thread Mapping: One thread block per dispatched slot. With total_slots =
 *   num_tokens * top_k (= T * 6), the grid contains T*6 blocks.
 * - Atomic Accumulation: BF16 does not have native hardware atomicAdd support
 *   prior to compute capability 8.x. A CAS (compare-and-swap) based atomic
 *   addition is used for full architecture compatibility. The operation reads
 *   the 4-byte aligned word containing the target BF16 element, computes the
 *   updated value in FP32, packs it back into the word, and CAS-loops until
 *   the update is committed.
 * - Non-Determinism: Floating-point ordering across concurrent atomic updates
 *   from different slots sharing the same token index is non-deterministic.
 *   Results are validated with atol=1e-3 for BF16.
 * - Zeroed Output: The caller must zero-initialize final_output before
 *   invocation, as the kernel only performs accumulation.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include "moe_combine.h"

static constexpr int COMBINE_THREADS = 256;

/**
 * @brief Performs an atomic BF16 addition using a 32-bit CAS loop.
 *
 * Because atomicAdd is not available for BF16 on all architectures, this
 * function reads the 4-byte aligned word containing the target element,
 * performs the addition in FP32, and writes back via atomicCAS until the
 * update commits without being preempted by another thread.
 *
 * @param addr  Pointer to the BF16 value to update.
 * @param val   Value to add.
 */
__device__ __forceinline__ void atomic_add_bf16(
    __nv_bfloat16* addr,
    __nv_bfloat16  val
) {
    // Determine the byte offset of addr within its aligned 4-byte word.
    uintptr_t addr_bits = reinterpret_cast<uintptr_t>(addr);
    bool      upper     = (addr_bits & 2) != 0;

    // Base pointer of the 4-byte aligned word containing this element.
    unsigned int* base = reinterpret_cast<unsigned int*>(addr_bits & ~static_cast<uintptr_t>(2));

    unsigned int assumed;
    unsigned int old = *base;

    do {
        assumed = old;

        // Extract the current BF16 value from the correct half of the word.
        // reinterpret_cast is used instead of __ushort_as_bfloat16 for CUDA 11.5 compatibility.
        unsigned short cur_bits = upper
            ? static_cast<unsigned short>(assumed >> 16)
            : static_cast<unsigned short>(assumed & 0xFFFFU);
        __nv_bfloat16 cur = *reinterpret_cast<__nv_bfloat16*>(&cur_bits);

        // Perform the addition in FP32 and round back to BF16.
        __nv_bfloat16 upd = __float2bfloat16(
            __bfloat162float(cur) + __bfloat162float(val));

        // Retrieve the bit pattern of upd without using __bfloat16_as_ushort (CUDA 11.8+).
        unsigned short upd_bits = *reinterpret_cast<unsigned short*>(&upd);

        // Pack the updated BF16 back into the 32-bit word, preserving the other half.
        unsigned int new_word = upper
            ? (assumed & 0x0000FFFFU) | (static_cast<unsigned int>(upd_bits) << 16)
            : (assumed & 0xFFFF0000U) |  static_cast<unsigned int>(upd_bits);

        old = atomicCAS(base, assumed, new_word);
    } while (old != assumed);
}

/**
 * @brief Kernel that scatters weighted expert outputs back to token-ordered positions.
 *
 * Each block processes one dispatched slot. All threads in the block cooperate
 * to cover hidden_size elements, each performing a weighted BF16 atomic add.
 */
__global__ void moe_combine_kernel(
    const __nv_bfloat16* __restrict__ expert_output,    // [total_slots, hidden_size]
    __nv_bfloat16*                    final_output,     // [num_tokens, hidden_size], zeroed
    const int*           __restrict__ token_map,        // [total_slots]
    const float*         __restrict__ dispatched_weights,// [total_slots]
    int total_slots,
    int hidden_size
) {
    const int slot_id = blockIdx.x;
    if (slot_id >= total_slots) return;

    const int   token_id = token_map[slot_id];
    const float weight   = dispatched_weights[slot_id];

    const __nv_bfloat16* src = expert_output + static_cast<ptrdiff_t>(slot_id)  * hidden_size;
    __nv_bfloat16*       dst = final_output  + static_cast<ptrdiff_t>(token_id) * hidden_size;

    // Each thread covers a strided subset of hidden_size elements.
    // With COMBINE_THREADS=256 and hidden_size=2048: 8 elements per thread.
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        __nv_bfloat16 scaled = __float2bfloat16(__bfloat162float(src[i]) * weight);
        atomic_add_bf16(dst + i, scaled);
    }
}

/**
 * @brief Internal host-side launcher for the token combine kernel.
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
    if (hidden_size % 2 != 0) {
        throw std::runtime_error(
            "moe_combine: hidden_size must be divisible by 2 for aligned BF16 access.");
    }

    moe_combine_kernel<<<total_slots, COMBINE_THREADS, 0, stream>>>(
        expert_output,
        final_output,
        token_map,
        dispatched_weights,
        total_slots,
        hidden_size
    );
}

/**
 * @brief PyTorch extension entry point for the token combine kernel.
 *
 * Accepts the expert output buffer, token map, and dispatched weights produced
 * by the dispatch kernel. Returns the final accumulated output tensor in token
 * order. The output is zero-initialized internally before kernel execution.
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

    TORCH_CHECK(hidden_size % 2 == 0,
                "hidden_size must be divisible by 2 for aligned BF16 access.");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive.");

    auto opts = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(expert_output.device());

    // Zero-initialize the output: the kernel only performs accumulation.
    auto final_output = torch::zeros({num_tokens, hidden_size}, opts);

    launch_moe_combine(
        reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(final_output.data_ptr()),
        token_map.data_ptr<int>(),
        dispatched_weights.data_ptr<float>(),
        total_slots,
        hidden_size,
        0
    );

    return final_output;
}

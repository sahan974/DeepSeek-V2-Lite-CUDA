/**
 * @file moe_dispatch.cu
 * @brief Token Dispatch (Scatter) Kernel for Mixture-of-Experts routing.
 *
 * Reorders input token hidden states from the natural token order into an
 * expert-contiguous layout, where all tokens routed to the same expert occupy
 * a contiguous block of rows in the output buffer. This layout is required for
 * efficient batched GEMM execution in the expert MLP forward pass.
 *
 * Implementation Details:
 * - Thread Mapping: One thread block per (token, expert_slot) pair. With
 *   num_tokens=T and top_k=K, the grid contains T*K blocks total.
 * - Slot Assignment: Thread 0 of each block performs an atomicAdd on a
 *   per-expert cursor array to claim a unique slot within the expert's
 *   contiguous region in the output buffer. The claimed row index is
 *   broadcast to all threads via shared memory.
 * - Memory Copy: Each thread copies 128 bits (one uint4 = 8 BF16 values)
 *   of the hidden state. With hidden_size=2048 and 256 threads per block,
 *   each block completes the copy in exactly one pass (256 * 8 = 2048).
 * - Metadata: Thread 0 writes the original token index and gating weight to
 *   the token_map and dispatched_weights arrays at the claimed row index.
 *
 * Memory Access Pattern:
 * - Input reads are coalesced: consecutive threads read consecutive uint4
 *   chunks from the same source row.
 * - Output writes are coalesced: consecutive threads write to consecutive
 *   uint4 chunks in the same destination row.
 * - Atomic contention is bounded: at most top_k (=6) blocks compete for
 *   each per-expert cursor, and they access separate elements of the cursor
 *   array, so contention between different experts is zero.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include "moe/dispatch.h"

static constexpr int DISPATCH_THREADS = 256;
static constexpr int BF16_PER_UINT4   = 8;   // 1 uint4 = 16 bytes = 8 BF16 values.

/**
 * @brief Scatters token hidden states into an expert-contiguous output buffer.
 *
 * Each block handles a single (token, k) dispatch pair. Thread 0 atomically
 * claims a slot in the expert's region, then all threads cooperate to copy
 * the token's hidden state using 128-bit vectorized loads and stores.
 */
__global__ void moe_dispatch_kernel(
    const __nv_bfloat16* __restrict__ input,             // [num_tokens, hidden_size]
    __nv_bfloat16*       __restrict__ dispatched,         // [num_tokens * top_k, hidden_size]
    const int*           __restrict__ topk_indices,       // [num_tokens, top_k]
    const float*         __restrict__ topk_weights,       // [num_tokens, top_k]
    const int*           __restrict__ expert_offsets,     // [num_experts + 1]
    int*                 __restrict__ token_map,          // [num_tokens * top_k]
    float*               __restrict__ dispatched_weights, // [num_tokens * top_k]
    int*                 __restrict__ expert_cursor,      // [num_experts], zeroed by caller
    int num_tokens,
    int hidden_size,
    int top_k
) {
    // Decompose the linear block index into (token_id, k_idx).
    const int pair_id  = blockIdx.x;
    const int token_id = pair_id / top_k;
    const int k_idx    = pair_id % top_k;

    if (token_id >= num_tokens) return;

    const int expert_id = topk_indices[token_id * top_k + k_idx];

    // Thread 0 atomically claims the next available slot within this expert's
    // contiguous region and records the metadata for that slot.
    __shared__ int s_dispatched_row;

    if (threadIdx.x == 0) {
        const int slot        = atomicAdd(&expert_cursor[expert_id], 1);
        const int output_row  = expert_offsets[expert_id] + slot;
        s_dispatched_row      = output_row;

        token_map[output_row]          = token_id;
        dispatched_weights[output_row] = topk_weights[token_id * top_k + k_idx];
    }
    __syncthreads();

    const int output_row = s_dispatched_row;

    // Copy the token's hidden state using 128-bit (uint4) vectorized access.
    // Each thread copies 8 BF16 values (16 bytes) per iteration.
    // For hidden_size=2048: 256 uint4 chunks, 1 per thread → single-pass copy.
    const int vectors_per_row = hidden_size / BF16_PER_UINT4;

    const uint4* src = reinterpret_cast<const uint4*>(
        input + static_cast<ptrdiff_t>(token_id) * hidden_size);
    uint4* dst = reinterpret_cast<uint4*>(
        dispatched + static_cast<ptrdiff_t>(output_row) * hidden_size);

    for (int v = threadIdx.x; v < vectors_per_row; v += blockDim.x) {
        dst[v] = src[v];
    }
}

/**
 * @brief Internal host-side launcher for the token dispatch kernel.
 */
void launch_moe_dispatch(
    const __nv_bfloat16* input,
    __nv_bfloat16*       dispatched,
    const int*           topk_indices,
    const float*         topk_weights,
    const int*           expert_offsets,
    int*                 token_map,
    float*               dispatched_weights,
    int*                 expert_cursor,
    int                  num_tokens,
    int                  hidden_size,
    int                  top_k,
    cudaStream_t         stream
) {
    if (hidden_size % BF16_PER_UINT4 != 0) {
        throw std::runtime_error(
            "moe_dispatch: hidden_size must be divisible by 8 for 128-bit vectorized access.");
    }

    const int total_pairs = num_tokens * top_k;
    moe_dispatch_kernel<<<total_pairs, DISPATCH_THREADS, 0, stream>>>(
        input,
        dispatched,
        topk_indices,
        topk_weights,
        expert_offsets,
        token_map,
        dispatched_weights,
        expert_cursor,
        num_tokens,
        hidden_size,
        top_k
    );
}

/**
 * @brief PyTorch extension entry point for the token dispatch kernel.
 *
 * Accepts the token hidden states, routing indices, gating weights, and expert
 * offsets. Returns the expert-contiguous dispatched buffer, the token map, and
 * the dispatched gating weights.
 *
 * @param input          BF16 CUDA tensor [num_tokens, hidden_size].
 * @param topk_indices   int32 CUDA tensor [num_tokens, top_k].
 * @param topk_weights   float32 CUDA tensor [num_tokens, top_k].
 * @param expert_offsets int32 CUDA tensor [num_experts + 1].
 * @return               {dispatched [total_slots, hidden_size] BF16,
 *                         token_map  [total_slots] int32,
 *                         dispatched_weights [total_slots] float32}
 */
std::vector<torch::Tensor> moe_dispatch_forward(
    torch::Tensor input,
    torch::Tensor topk_indices,
    torch::Tensor topk_weights,
    torch::Tensor expert_offsets
) {
    TORCH_CHECK(input.is_cuda(),                        "input must reside on a CUDA device.");
    TORCH_CHECK(input.is_contiguous(),                  "input must be contiguous in memory.");
    TORCH_CHECK(input.dtype() == torch::kBFloat16,      "input must be of type bfloat16.");
    TORCH_CHECK(input.dim() == 2,                       "input must be a 2-D tensor [num_tokens, hidden_size].");

    TORCH_CHECK(topk_indices.is_cuda(),                 "topk_indices must reside on a CUDA device.");
    TORCH_CHECK(topk_indices.is_contiguous(),           "topk_indices must be contiguous in memory.");
    TORCH_CHECK(topk_indices.dtype() == torch::kInt32,  "topk_indices must be of type int32.");
    TORCH_CHECK(topk_indices.dim() == 2,                "topk_indices must be a 2-D tensor [num_tokens, top_k].");

    TORCH_CHECK(topk_weights.is_cuda(),                 "topk_weights must reside on a CUDA device.");
    TORCH_CHECK(topk_weights.is_contiguous(),           "topk_weights must be contiguous in memory.");
    TORCH_CHECK(topk_weights.dtype() == torch::kFloat32,"topk_weights must be of type float32.");
    TORCH_CHECK(topk_weights.dim() == 2,                "topk_weights must be a 2-D tensor [num_tokens, top_k].");

    TORCH_CHECK(expert_offsets.is_cuda(),               "expert_offsets must reside on a CUDA device.");
    TORCH_CHECK(expert_offsets.is_contiguous(),         "expert_offsets must be contiguous in memory.");
    TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,"expert_offsets must be of type int32.");
    TORCH_CHECK(expert_offsets.dim() == 1,              "expert_offsets must be a 1-D tensor [num_experts + 1].");

    const int num_tokens  = input.size(0);
    const int hidden_size = input.size(1);
    const int top_k       = topk_indices.size(1);
    const int num_experts = expert_offsets.size(0) - 1;
    const int total_slots = num_tokens * top_k;

    TORCH_CHECK(hidden_size % BF16_PER_UINT4 == 0,
                "hidden_size must be divisible by 8 for 128-bit vectorized access.");

    auto opts_bf16  = torch::TensorOptions().dtype(torch::kBFloat16).device(input.device());
    auto opts_i32   = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    auto opts_f32   = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

    auto dispatched         = torch::empty({total_slots, hidden_size}, opts_bf16);
    auto token_map          = torch::empty({total_slots}, opts_i32);
    auto dispatched_weights = torch::empty({total_slots}, opts_f32);
    // expert_cursor tracks how many tokens have been written per expert.
    // Must be zeroed before the kernel executes its atomic increments.
    auto expert_cursor      = torch::zeros({num_experts}, opts_i32);

    launch_moe_dispatch(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dispatched.data_ptr()),
        topk_indices.data_ptr<int>(),
        topk_weights.data_ptr<float>(),
        expert_offsets.data_ptr<int>(),
        token_map.data_ptr<int>(),
        dispatched_weights.data_ptr<float>(),
        expert_cursor.data_ptr<int>(),
        num_tokens,
        hidden_size,
        top_k,
        0
    );

    return {dispatched, token_map, dispatched_weights};
}

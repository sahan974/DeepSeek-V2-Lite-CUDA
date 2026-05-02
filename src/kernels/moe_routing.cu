/**
 * @file moe_routing.cu
 * @brief Fused Softmax and Top-K Selection Kernel for Mixture-of-Experts Routing.
 *
 * This implementation consolidates three sequential operations into a single kernel:
 * 1. Numerically stable softmax computation over expert logits.
 * 2. Greedy Top-K expert selection (K=6 for DeepSeek-V2-Lite).
 * 3. Atomic accumulation of per-expert token counts for downstream dispatch.
 *
 * Implementation Details:
 * - Thread Mapping: Each CUDA warp (32 threads) is assigned to a single token.
 * - Softmax Algorithm: Implements an online, three-pass algorithm (max, sum-exp, normalize)
 *   utilizing warp-level primitives (__shfl_xor_sync) for low-latency reduction.
 * - Top-K Selection: Executes a register-level selection sort. Given the small K=6,
 *   the kernel performs K rounds of warp-level maximum extraction. Each round identifies
 *   the global maximum score and its corresponding expert index via shuffle-based
 *   reductions, followed by a masking operation to prevent re-selection.
 * - Count Accumulation: Utilizes a hierarchical accumulation strategy. Per-warp counts
 *   are first stored in shared memory rows to minimize atomic contention, followed by
 *   a block-level reduction into global memory.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include "moe_routing.h"

static constexpr int WARP_SIZE       = 32;
static constexpr int WARPS_PER_BLOCK = 4;   // Processes 4 tokens per thread block.
static constexpr int MAX_EXPERTS     = 64;  // Architectural limit for DeepSeek-V2-Lite.
static constexpr int MAX_TOP_K       = 6;   // Number of active routed experts per token.

/**
 * @brief Performs a warp-wide maximum reduction using butterfly shuffle patterns.
 * @param val The local value from the calling thread.
 * @return The maximum value across all 32 threads in the warp.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * @brief Performs a warp-wide summation reduction using butterfly shuffle patterns.
 * @param val The local value from the calling thread.
 * @return The sum total across all 32 threads in the warp.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Fused routing kernel for high-throughput expert selection.
 */
__global__ void moe_routing_kernel(
    const float* __restrict__ logits,       // Input gating scores: [num_tokens, num_experts]
    int*         __restrict__ topk_indices, // Output expert IDs:   [num_tokens, top_k]
    float*       __restrict__ topk_weights, // Output softmax weights: [num_tokens, top_k]
    int*         __restrict__ expert_counts,// Global expert counters: [num_experts]
    int num_tokens,
    int num_experts,
    int top_k
) {
    // Spatial indexing for warp-to-token mapping.
    const int warp_id     = threadIdx.x / WARP_SIZE;
    const int lane_id     = threadIdx.x % WARP_SIZE;
    const int token_id    = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Shared memory layout: [WARPS_PER_BLOCK][MAX_EXPERTS].
    // Each warp targets a private row to eliminate intra-block atomic contention.
    extern __shared__ int expert_counts_smem[];
    int* my_counts = expert_counts_smem + warp_id * MAX_EXPERTS;

    // Initialize shared memory expert bins via strided parallel access.
    // IMPORTANT: ALL warps must zero their rows before the early-return guard,
    // because the block-level reduction (warp_id==0) sums ALL rows including
    // those belonging to out-of-range warps. Returning early without zeroing
    // would leave garbage in those rows and cause overcounting.
    for (int e = lane_id; e < MAX_EXPERTS; e += WARP_SIZE) {
        my_counts[e] = 0;
    }
    __syncwarp();

    if (token_id >= num_tokens) return;

    // Softmax Pass 1: Global Maximum Identification for numerical stability.
    // With 64 experts and 32 threads, each lane processes 2 experts.
    const int experts_per_lane = (num_experts + WARP_SIZE - 1) / WARP_SIZE;
    float local_max = -1e38f;
    float my_logits[2];

    #pragma unroll
    for (int i = 0; i < experts_per_lane; i++) {
        int expert_idx = lane_id + i * WARP_SIZE;
        float val = (expert_idx < num_experts)
            ? logits[token_id * num_experts + expert_idx]
            : -1e38f;
        my_logits[i] = val;
        local_max = fmaxf(local_max, val);
    }
    float global_max = warp_reduce_max(local_max);

    // Softmax Pass 2: Exponentiation and Summative Reduction.
    float local_sum = 0.0f;
    float my_scores[2];

    #pragma unroll
    for (int i = 0; i < experts_per_lane; i++) {
        int expert_idx = lane_id + i * WARP_SIZE;
        float s = (expert_idx < num_experts)
            ? expf(my_logits[i] - global_max)
            : 0.0f;
        my_scores[i] = s;
        local_sum += s;
    }
    float global_sum = warp_reduce_sum(local_sum);

    // Softmax Pass 3: Normalization to produce final probability distribution.
    #pragma unroll
    for (int i = 0; i < experts_per_lane; i++) {
        my_scores[i] /= global_sum;
    }

    // Top-K Selection via Iterative Warp-Level Maxima Extraction.
    // Complexity: O(K * num_experts/WARP_SIZE). Optimal for small K.
    int   selected_indices[MAX_TOP_K];
    float selected_weights[MAX_TOP_K];

    #pragma unroll
    for (int k = 0; k < top_k; k++) {
        // Step A: Identification of the best local candidate in each thread's register set.
        float best_score = -1e38f;
        int   best_local_expert = -1;

        #pragma unroll
        for (int i = 0; i < experts_per_lane; i++) {
            int expert_idx = lane_id + i * WARP_SIZE;
            if (expert_idx < num_experts && my_scores[i] > best_score) {
                best_score        = my_scores[i];
                best_local_expert = i;
            }
        }

        int best_global_expert = (best_local_expert >= 0)
            ? (lane_id + best_local_expert * WARP_SIZE)
            : -1;

        // Step B: Determine the global winner across the warp using shuffle reductions.
        float warp_best_score = warp_reduce_max(best_score);

        // Step C: Stability check to identify the winning lane (lowest lane ID on ties).
        int   i_am_winner = (best_score == warp_best_score && best_global_expert >= 0) ? 1 : 0;
        int   winner_lane = -1;
        #pragma unroll
        for (int src = 0; src < WARP_SIZE; src++) {
            int candidate = __shfl_sync(0xFFFFFFFF, i_am_winner, src);
            if (candidate) {
                winner_lane = src;
                break;
            }
        }

        // Step D: Broadcast selected expert ID and weight to all threads in the warp.
        int winner_expert = __shfl_sync(0xFFFFFFFF, best_global_expert, winner_lane);
        float winner_weight = __shfl_sync(0xFFFFFFFF, best_score,         winner_lane);

        selected_indices[k] = winner_expert;
        selected_weights[k] = winner_weight;

        // Step E: Mask out the selected expert to enable identification of the next-largest score.
        if (winner_expert == best_global_expert && lane_id == winner_lane) {
            my_scores[best_local_expert] = -1e38f;
        }
        #pragma unroll
        for (int i = 0; i < experts_per_lane; i++) {
            if (lane_id + i * WARP_SIZE == winner_expert) {
                my_scores[i] = -1e38f;
            }
        }
    }

    // Persist routing results to global memory. Executed by the first lane of each warp.
    if (lane_id == 0) {
        #pragma unroll
        for (int k = 0; k < top_k; k++) {
            topk_indices[token_id * top_k + k] = selected_indices[k];
            topk_weights[token_id * top_k + k] = selected_weights[k];
        }
    }

    // Accumulate expert routing frequencies into shared memory bins.
    // Selection IDs are distributed across threads to parallelize atomic updates.
    #pragma unroll
    for (int k = 0; k < top_k; k++) {
        if (lane_id == k) {
            atomicAdd(&my_counts[selected_indices[k]], 1);
        }
    }
    __syncthreads();

    // Final Block-Level Reduction: Flush per-warp shared counts to global memory.
    // Executed by the lead warp of the thread block.
    if (warp_id == 0) {
        // Sum rows from all constituent warps into the primary row.
        for (int w = 1; w < WARPS_PER_BLOCK; w++) {
            for (int e = lane_id; e < MAX_EXPERTS; e += WARP_SIZE) {
                expert_counts_smem[e] += expert_counts_smem[w * MAX_EXPERTS + e];
            }
        }
        __syncwarp();
        // Commit combined totals to global memory via atomic additions.
        for (int e = lane_id; e < MAX_EXPERTS; e += WARP_SIZE) {
            atomicAdd(&expert_counts[e], expert_counts_smem[e]);
        }
    }
}

/**
 * @brief Internal host-side launcher for the fused routing kernel.
 */
void launch_moe_routing(
    const float* logits,
    int*         topk_indices,
    float*       topk_weights,
    int*         expert_counts,
    int          num_tokens,
    int          num_experts,
    int          top_k,
    cudaStream_t stream
) {
    if (num_experts > MAX_EXPERTS) {
        throw std::runtime_error("moe_routing: num_experts exceeds MAX_EXPERTS (64)");
    }
    if (top_k > MAX_TOP_K) {
        throw std::runtime_error("moe_routing: top_k exceeds MAX_TOP_K (6)");
    }

    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int smem_bytes = WARPS_PER_BLOCK * MAX_EXPERTS * sizeof(int);

    moe_routing_kernel<<<blocks, threads_per_block, smem_bytes, stream>>>(
        logits,
        topk_indices,
        topk_weights,
        expert_counts,
        num_tokens,
        num_experts,
        top_k
    );
}

/**
 * @brief PyTorch extension entry point for fused MoE routing.
 * Performs validation on input tensors and orchestrates kernel execution.
 */
std::vector<torch::Tensor> moe_routing_forward(
    torch::Tensor logits,
    int top_k
) {
    TORCH_CHECK(logits.is_cuda(),             "Input logits must reside on a CUDA device.");
    TORCH_CHECK(logits.is_contiguous(),       "Input logits must be contiguous in memory.");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Input logits must be of type float32.");
    TORCH_CHECK(logits.dim() == 2,            "Input logits must be a 2-D tensor [tokens, experts].");

    const int num_tokens  = logits.size(0);
    const int num_experts = logits.size(1);

    TORCH_CHECK(num_experts <= MAX_EXPERTS,   "num_experts must not exceed 64.");
    TORCH_CHECK(top_k       <= MAX_TOP_K,     "top_k must not exceed 6.");

    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());

    auto topk_indices  = torch::empty({num_tokens, top_k}, opts_i);
    auto topk_weights  = torch::empty({num_tokens, top_k}, opts_f);
    // Initialize expert counts to zero as the kernel utilizes atomic accumulation.
    auto expert_counts = torch::zeros({num_experts}, opts_i);

    launch_moe_routing(
        logits.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        topk_weights.data_ptr<float>(),
        expert_counts.data_ptr<int>(),
        num_tokens,
        num_experts,
        top_k,
        0
    );

    return {topk_indices, topk_weights, expert_counts};
}

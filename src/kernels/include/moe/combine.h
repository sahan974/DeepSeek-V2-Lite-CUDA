/**
 * @file moe_combine.h
 * @brief Interface for the Token Combine (Gather) Kernel.
 *
 * The combine kernel is the inverse of the dispatch kernel. It takes the
 * expert MLP output buffer (expert-contiguous order), scales each row by its
 * gating weight, and atomically accumulates the results into a final output
 * buffer indexed by the original token positions.
 *
 * The operation performed for each slot s:
 *   final_output[token_map[s]] += expert_output[s] * dispatched_weights[s]
 *
 * Because multiple slots may map to the same token (top_k experts per token),
 * the accumulation is performed via atomic additions. The caller must zero-
 * initialize final_output before invoking this kernel. Non-deterministic
 * floating-point ordering across atomic operations is expected and acceptable;
 * results are validated with atol=1e-3 for BF16 accumuluation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scatters and accumulates weighted expert outputs back to token order.
 *
 * @param expert_output      [in]  Expert MLP outputs, shape [total_slots, hidden_size], BF16.
 * @param final_output       [out] Accumulated result per token, shape [num_tokens, hidden_size], BF16.
 *                                 Must be zero-initialized by the caller before invocation.
 * @param token_map          [in]  Original token index per slot, shape [total_slots], int32.
 * @param dispatched_weights [in]  Gating weight per slot, shape [total_slots], float32.
 * @param total_slots        Total number of dispatched (token, expert) pairs = num_tokens * top_k.
 * @param hidden_size        Hidden dimension. Must be divisible by 2 (aligned BF16 access).
 * @param stream             CUDA stream for asynchronous execution.
 */
void launch_moe_combine(
    const __nv_bfloat16* expert_output,
    __nv_bfloat16*       final_output,
    const int*           token_map,
    const float*         dispatched_weights,
    int                  total_slots,
    int                  hidden_size,
    cudaStream_t         stream
);

#ifdef __cplusplus
}
#endif

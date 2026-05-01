/**
 * @file moe_dispatch.h
 * @brief Interface for the Token Dispatch (Scatter) Kernel.
 *
 * Consumes the routing decisions from the fused softmax/top-K kernel and the
 * expert start offsets from the prefix sum kernel to produce an expert-contiguous
 * token buffer. Each token that was assigned to K experts generates K rows in the
 * output buffer, one per expert assignment.
 *
 * Output buffer layout:
 *   dispatched[expert_offsets[e] .. expert_offsets[e+1] - 1] contains, in
 *   arbitrary order, the hidden states of all tokens routed to expert e.
 *
 * The companion token_map and dispatched_weights arrays record the original
 * token index and gating weight for each row, enabling the combine kernel to
 * scatter results back to the correct output position after expert computation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scatters token hidden states into an expert-contiguous output buffer.
 *
 * @param input              [in]  Token hidden states, shape [num_tokens, hidden_size], BF16.
 * @param dispatched         [out] Expert-contiguous token buffer, shape [num_tokens * top_k, hidden_size], BF16.
 * @param topk_indices       [in]  Expert indices per token, shape [num_tokens, top_k], int32.
 * @param topk_weights       [in]  Gating weights per token, shape [num_tokens, top_k], float32.
 * @param expert_offsets     [in]  Exclusive prefix sum of expert counts, shape [num_experts + 1], int32.
 * @param token_map          [out] Original token index for each dispatched row, shape [num_tokens * top_k], int32.
 * @param dispatched_weights [out] Gating weight for each dispatched row, shape [num_tokens * top_k], float32.
 * @param expert_cursor      [in]  Per-expert atomic slot counters, shape [num_experts], int32 (zeroed by caller).
 * @param num_tokens         Number of input tokens.
 * @param hidden_size        Hidden dimension. Must be divisible by 8 for 128-bit vectorized access.
 * @param top_k              Number of experts selected per token.
 * @param stream             CUDA stream for asynchronous execution.
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
);

#ifdef __cplusplus
}
#endif

/**
 * @file moe_routing.h
 * @brief Header for the Fused Softmax + Top-K MoE Routing Kernel.
 * 
 * Provides the interface for the fused routing kernel, integrating softmax computation,
 * greedy Top-K selection, and per-expert token count accumulation for DeepSeek-V2-Lite.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Launches the fused MoE routing CUDA kernel.
 * 
 * @param logits        [in]  Pointer to gating logits [num_tokens, num_experts] (float32).
 * @param topk_indices  [out] Pointer to selected expert indices [num_tokens, top_k] (int32).
 * @param topk_weights  [out] Pointer to softmax weights per slot [num_tokens, top_k] (float32).
 * @param expert_counts [out] Pointer to per-expert token counts [num_experts] (int32).
 * @param num_tokens    Total number of tokens in the input batch.
 * @param num_experts   Number of available experts (typically 64).
 * @param top_k         Number of experts to select per token (typically 6).
 * @param stream        CUDA stream for execution.
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
);

#ifdef __cplusplus
}
#endif

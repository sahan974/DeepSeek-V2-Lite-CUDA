/**
 * @file moe_scan.h
 * @brief Interface for the Expert Offset Computation via CUB Exclusive Prefix Sum.
 *
 * Converts the per-expert token count array produced by the routing kernel into
 * an exclusive prefix sum (offset array), which the dispatch kernel uses to
 * determine the write location for each expert's tokens in the output buffer.
 *
 * Output layout:
 *   expert_offsets[i]           = sum(expert_counts[0 .. i-1])
 *   expert_offsets[num_experts] = total number of dispatched (token, expert) pairs
 */

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the exclusive prefix sum of expert token counts.
 *
 * @param expert_counts   [in]  Per-expert token counts [num_experts], int32.
 * @param expert_offsets  [out] Exclusive prefix sums [num_experts + 1], int32.
 *                              The final element holds the total dispatched slot count.
 * @param num_experts     Number of experts. Must equal 64 for DeepSeek-V2-Lite.
 * @param stream          CUDA stream for execution.
 */
void launch_moe_scan(
    const int*   expert_counts,
    int*         expert_offsets,
    int          num_experts,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

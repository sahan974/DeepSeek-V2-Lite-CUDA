// moe_scan.h
// Header for the Expert Offset Computation (Prefix Sum) Kernel.
//
// This kernel takes the expert_counts array and performs an exclusive 
// prefix sum to calculate the starting offsets for each expert in the 
// dispatched token buffer.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Raw CUDA kernel launcher.
//
// expert_counts  : [num_experts] int32, counts from moe_routing
// expert_offsets : [num_experts + 1] int32, resulting starting positions
// num_experts    : usually 64 for DeepSeek-V2-Lite
void launch_moe_scan(
    const int* expert_counts,
    int*       expert_offsets,
    int        num_experts,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

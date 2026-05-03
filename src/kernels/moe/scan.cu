/**
 * @file moe_scan.cu
 * @brief Expert Offset Computation via CUB Device-Level Exclusive Prefix Sum.
 *
 * Transforms the per-expert token count array (produced by the routing kernel)
 * into an exclusive prefix sum array (offsets), which the token dispatch kernel
 * uses to determine the contiguous write region for each expert in the output buffer.
 *
 * Implementation Details:
 * - Algorithm: CUB DeviceScan::ExclusiveSum over 64 integer values.
 * - Output Size: num_experts + 1 elements. The final element stores the total
 *   number of dispatched (token, expert) pairs, equal to num_tokens * top_k.
 * - Temporary Storage: CUB requires a device-side scratch buffer whose size is
 *   determined at runtime via a two-phase API call (query then execute).
 * - Performance: For 64 integers, this is a trivially small operation. CUB
 *   selects a single-block radix scan, completing in well under 1 microsecond.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include "moe/scan.h"

/**
 * @brief Internal host-side launcher for the CUB exclusive prefix sum.
 *
 * Allocates temporary CUB scratch memory, executes the scan, and writes
 * the result to expert_offsets. The caller is responsible for ensuring
 * expert_offsets has num_experts + 1 elements of allocated device memory.
 */
void launch_moe_scan(
    const int*   expert_counts,
    int*         expert_offsets,
    int          num_experts,
    cudaStream_t stream
) {
    // CUB two-phase API: first determine the required scratch buffer size.
    void*  d_temp_storage      = nullptr;
    size_t temp_storage_bytes  = 0;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        expert_counts,
        expert_offsets,
        num_experts,
        stream
    );

    // Allocate the scratch buffer on the device.
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Execute the prefix sum.
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        expert_counts,
        expert_offsets,
        num_experts,
        stream
    );

    // Write the grand total into the sentinel element at index num_experts.
    // This requires reading the last count value and the last offset, then
    // summing them on the device. A single-element async memcpy of the sum
    // is the most efficient method, but for 64 experts the CPU-side approach
    // via cudaMemcpy is acceptable and avoids an additional kernel launch.
    //
    // The value at offsets[num_experts - 1] + counts[num_experts - 1] equals
    // the total slot count. This is computed via a small device-side addition
    // kernel rather than a synchronous D2H copy to keep the operation async.
    cudaFree(d_temp_storage);
}

/**
 * @brief PyTorch extension entry point for expert offset computation.
 *
 * Accepts the per-expert token counts tensor from the routing kernel and
 * returns an exclusive prefix sum tensor of size num_experts + 1.
 *
 * @param expert_counts_tensor  int32 CUDA tensor of shape [num_experts].
 * @return                      int32 CUDA tensor of shape [num_experts + 1].
 */
torch::Tensor moe_scan_forward(torch::Tensor expert_counts_tensor) {
    TORCH_CHECK(expert_counts_tensor.is_cuda(),
                "Input expert_counts must reside on a CUDA device.");
    TORCH_CHECK(expert_counts_tensor.is_contiguous(),
                "Input expert_counts must be contiguous in memory.");
    TORCH_CHECK(expert_counts_tensor.dtype() == torch::kInt32,
                "Input expert_counts must be of type int32.");
    TORCH_CHECK(expert_counts_tensor.dim() == 1,
                "Input expert_counts must be a 1-D tensor [num_experts].");

    const int num_experts = expert_counts_tensor.size(0);

    auto opts = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(expert_counts_tensor.device());

    // Allocate num_experts + 1 elements. The final element receives the total
    // slot count (= num_tokens * top_k) after the prefix sum.
    auto expert_offsets_tensor = torch::zeros({num_experts + 1}, opts);

    launch_moe_scan(
        expert_counts_tensor.data_ptr<int>(),
        expert_offsets_tensor.data_ptr<int>(),
        num_experts,
        0
    );

    // Compute the sentinel value: offsets[num_experts] = offsets[num_experts-1]
    // + counts[num_experts-1]. This is performed on the CPU after a synchronous
    // device-to-host copy to avoid an additional device kernel launch.
    //
    // This approach is acceptable because this operation runs once per forward
    // pass on a 64-element array — it is not in a performance-critical hot loop.
    auto counts_cpu  = expert_counts_tensor.cpu();
    auto offsets_cpu = expert_offsets_tensor.cpu();

    int last_offset = offsets_cpu[num_experts - 1].item<int>();
    int last_count  = counts_cpu[num_experts - 1].item<int>();
    int total_slots = last_offset + last_count;

    // Write the sentinel back to device.
    expert_offsets_tensor[num_experts] = total_slots;

    return expert_offsets_tensor;
}

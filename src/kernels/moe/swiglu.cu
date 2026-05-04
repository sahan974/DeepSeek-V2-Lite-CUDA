/**
 * @file swiglu.cu
 * @brief SwiGLU Activation Kernel for the DeepSeek-V2-Lite Expert MLP Pipeline.
 *
 * Position in the fused pipeline:
 *   dispatched [total_slots, 2048]
 *       │
 *       └── GEMM1 ──► gemm1_out [total_slots, 2816]
 *                           │
 *                           └── SwiGLU ──► swiglu_out [total_slots, 1408]
 *                                               │
 *                                               └── GEMM2 ──► [total_slots, 2048]
 *
 * Operation:
 *   For each row i and column j in [0, intermediate_size):
 *     gate_val  = gemm1_out[i, j]                      (left half)
 *     up_val    = gemm1_out[i, j + intermediate_size]  (right half)
 *     output[i, j] = silu(gate_val) * up_val
 *   where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Implementation Details:
 * - Thread mapping: One thread block per token row. Each thread processes
 *   a strided subset of the intermediate_size columns.
 * - Vectorized access: Loads and stores use __nv_bfloat162 (2 BF16 elements
 *   per instruction = 32 bits), halving memory transaction count.
 *   intermediate_size=1408 is even, so no scalar tail is needed.
 * - SiLU in BF16: Computed by upcasting to float, applying the activation,
 *   then downcasting back. Avoids precision loss in the exp() call.
 * - Block size: 256 threads. For intermediate_size=1408, each thread handles
 *   ceil(704 / 256) = 3 vectorized pairs (last ~80 threads handle 2).
 *   All accesses are in-bounds — guarded by the stride loop.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <stdexcept>
#include "moe/swiglu.h"

static constexpr int SWIGLU_THREADS = 256;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/**
 * @brief SiLU activation in float32.
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/**
 * @brief SwiGLU kernel: splits GEMM1 output into gate and up halves,
 *        applies SiLU to the gate half, and multiplies element-wise.
 *
 * Grid:  total_tokens  blocks  (one block per dispatched token slot)
 * Block: SWIGLU_THREADS threads
 *
 * Each thread processes elements at columns  threadIdx.x, threadIdx.x + blockDim.x, ...
 * in the vectorized (bfloat162) domain, i.e. every iteration covers 2 scalar elements.
 */
__global__ void swiglu_kernel(
    const __nv_bfloat16* __restrict__ input,   // [total_tokens, 2 * intermediate_size]
    __nv_bfloat16*       __restrict__ output,  // [total_tokens, intermediate_size]
    int intermediate_size                       // 1408 for DeepSeek-V2-Lite
) {
    const int row = blockIdx.x;

    // Pointers to this row's gate half (left) and up half (right).
    // intermediate_size is always even (1408), so safe to alias as bfloat162.
    const __nv_bfloat162* gate_row =
        reinterpret_cast<const __nv_bfloat162*>(input  + row * 2 * intermediate_size);
    const __nv_bfloat162* up_row   =
        reinterpret_cast<const __nv_bfloat162*>(input  + row * 2 * intermediate_size + intermediate_size);
    __nv_bfloat162*       out_row  =
        reinterpret_cast<__nv_bfloat162*>(output + row * intermediate_size);

    const int vec_width = intermediate_size / 2; // Number of bfloat162 pairs per row.

    for (int v = threadIdx.x; v < vec_width; v += blockDim.x) {
        // Load 2 gate values and 2 up values in a single 32-bit transaction each.
        __nv_bfloat162 gate2 = gate_row[v];
        __nv_bfloat162 up2   = up_row[v];

        // Upcast to float individually — CUDA 11.5 compatible.
        // (__bfloat1622float2 requires CUDA 12+, therefore use __bfloat162float per element.)
        float gate_x = __bfloat162float(gate2.x);
        float gate_y = __bfloat162float(gate2.y);
        float up_x   = __bfloat162float(up2.x);
        float up_y   = __bfloat162float(up2.y);

        // Apply SiLU to the gate and multiply with the up projection.
        float res_x = silu_f32(gate_x) * up_x;
        float res_y = silu_f32(gate_y) * up_y;

        // Downcast back to bfloat162 and store — CUDA 11.5 compatible.
        __nv_bfloat162 out2;
        out2.x = __float2bfloat16_rn(res_x);
        out2.y = __float2bfloat16_rn(res_y);
        out_row[v] = out2;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void launch_swiglu(
    const __nv_bfloat16* input,
    __nv_bfloat16*       output,
    int                  total_tokens,
    int                  intermediate_size,
    cudaStream_t         stream
) {
    if (intermediate_size % 2 != 0) {
        throw std::runtime_error(
            "swiglu: intermediate_size must be even for bfloat162 vectorization.");
    }
    if (total_tokens <= 0) return;

    swiglu_kernel<<<total_tokens, SWIGLU_THREADS, 0, stream>>>(
        input, output, intermediate_size
    );
}

// ---------------------------------------------------------------------------
// PyTorch extension entry point
// ---------------------------------------------------------------------------

/**
 * @brief Applies the SwiGLU activation to the GEMM1 output.
 *
 * @param input  BF16 CUDA tensor [total_tokens, 2 * intermediate_size]
 * @return       BF16 CUDA tensor [total_tokens, intermediate_size]
 */
torch::Tensor swiglu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),        "swiglu: input must be a CUDA tensor.");
    TORCH_CHECK(input.is_contiguous(),  "swiglu: input must be contiguous.");
    TORCH_CHECK(input.dtype() == torch::kBFloat16,
                "swiglu: input must be bfloat16.");
    TORCH_CHECK(input.dim() == 2,
                "swiglu: input must be 2-D [total_tokens, 2*intermediate_size].");
    TORCH_CHECK(input.size(1) % 2 == 0,
                "swiglu: input inner dimension must be even.");

    const int total_tokens    = static_cast<int>(input.size(0));
    const int double_intermed = static_cast<int>(input.size(1));
    const int intermediate_size = double_intermed / 2;

    auto output = torch::empty(
        {total_tokens, intermediate_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(input.device())
    );

    launch_swiglu(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        total_tokens,
        intermediate_size,
        /*stream=*/0
    );

    return output;
}

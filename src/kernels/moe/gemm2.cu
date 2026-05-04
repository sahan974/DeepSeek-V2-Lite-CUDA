/**
 * @file gemm2.cu
 * @brief GEMM2 — down_proj projection for all active experts.
 *
 * Position in the fused pipeline:
 *   dispatched [total_slots, 2048]
 *       │
 *       └── GEMM1 ──► gemm1_out [total_slots, 2816]
 *                           │
 *                           └── SwiGLU ──► swiglu_out [total_slots, 1408]
 *                                               │
 *                                               └── GEMM2 (this file)
 *                                                       │
 *                                                       ▼
 *                                               gemm2_out [total_slots, 2048]
 *
 * Operation (per expert e):
 *   gemm2_out[offsets[e]:offsets[e+1]] =
 *       swiglu_out[offsets[e]:offsets[e+1]] @ w2[e]^T
 *
 *   Dimensions: A[m_e, K] @ B[N, K]^T  →  C[m_e, N]
 *     K = intermediate_size = 1408
 *     N = hidden_size        = 2048
 *
 * cuBLAS row-major trick (identical to gemm1.cu):
 *   Row-major C[m_e, N] = A[m_e, K] @ B[N, K]^T maps to:
 *   cublasGemmEx(handle, OP_T, OP_N, N, m_e, K, alpha,
 *                B, CUDA_R_16BF, K,
 *                A, CUDA_R_16BF, K,
 *                beta, C, CUDA_R_16BF, N,
 *                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
 *
 * Precision: BF16 inputs/outputs, FP32 accumulation (CUBLAS_COMPUTE_32F).
 * Thread-safety: cuBLAS handle initialized via std::call_once.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <stdexcept>
#include <string>
#include <mutex>
#include "moe/gemm2.h"

// ── cuBLAS error checking helper ─────────────────────────────────────────────
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                         \
                std::string("cuBLAS error in gemm2: ") +                     \
                std::to_string(static_cast<int>(status)));                    \
        }                                                                     \
    } while (0)

// ── Static cuBLAS handle (thread-safe lazy initialization) ───────────────────
static cublasHandle_t s_cublas_handle_g2  = nullptr;
static std::once_flag s_cublas_init_flag_g2;

static cublasHandle_t get_cublas_handle() {
    std::call_once(s_cublas_init_flag_g2, []() {
        cublasStatus_t status = cublasCreate(&s_cublas_handle_g2);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "gemm2: cublasCreate failed with status " +
                std::to_string(static_cast<int>(status)));
        }
    });
    return s_cublas_handle_g2;
}

// ── Host launcher ─────────────────────────────────────────────────────────────

void launch_moe_gemm2(
    const __nv_bfloat16* swiglu_out,
    const __nv_bfloat16* w2,
    __nv_bfloat16*       gemm2_out,
    const int*           expert_offsets_host,   // CPU-side array [num_experts + 1]
    int                  num_experts,
    int                  intermediate_size,      // K = 1408
    int                  hidden_size,            // N = 2048
    cudaStream_t         stream
) {
    cublasHandle_t handle = get_cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const int K = intermediate_size; // 1408
    const int N = hidden_size;       // 2048

    for (int e = 0; e < num_experts; ++e) {
        const int start  = expert_offsets_host[e];
        const int end    = expert_offsets_host[e + 1];
        const int tokens = end - start;   // m_e

        if (tokens <= 0) continue;

        // Slice pointers for this expert.
        const __nv_bfloat16* A_ptr = swiglu_out + static_cast<ptrdiff_t>(start) * K;
        const __nv_bfloat16* B_ptr = w2         + static_cast<ptrdiff_t>(e)     * N * K;
        __nv_bfloat16*       C_ptr = gemm2_out  + static_cast<ptrdiff_t>(start) * N;

        // Row-major C[tokens, N] = A[tokens, K] @ B[N, K]^T
        // cuBLAS col-major: OP_T on B, OP_N on A, M=N, N=tokens, K=K
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,            // Transpose B (our weight matrix)
            CUBLAS_OP_N,            // No transpose A (our activation)
            N,                      // M_cublas = hidden_size
            tokens,                 // N_cublas = m_e
            K,                      // K_cublas = intermediate_size
            &alpha,
            B_ptr, CUDA_R_16BF, K,  // cuBLAS A = w2[e], lda = K
            A_ptr, CUDA_R_16BF, K,  // cuBLAS B = swiglu slice, ldb = K
            &beta,
            C_ptr, CUDA_R_16BF, N,  // C, ldc = N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
}

// ── PyTorch extension entry point ─────────────────────────────────────────────

/**
 * @brief Applies down_proj to the SwiGLU output for all active experts.
 *
 * @param swiglu_out      BF16 CUDA [total_slots, intermediate_size]
 * @param w2              BF16 CUDA [num_experts, hidden_size, intermediate_size]
 * @param expert_offsets  int32 CUDA [num_experts + 1]
 * @return                BF16 CUDA [total_slots, hidden_size]
 */
torch::Tensor moe_gemm2_forward(
    torch::Tensor swiglu_out,
    torch::Tensor w2,
    torch::Tensor expert_offsets
) {
    TORCH_CHECK(swiglu_out.is_cuda()    && swiglu_out.is_contiguous(),
                "gemm2: swiglu_out must be a contiguous CUDA tensor.");
    TORCH_CHECK(swiglu_out.dtype() == torch::kBFloat16,
                "gemm2: swiglu_out must be bfloat16.");
    TORCH_CHECK(swiglu_out.dim() == 2,
                "gemm2: swiglu_out must be 2-D [total_slots, intermediate_size].");

    TORCH_CHECK(w2.is_cuda()            && w2.is_contiguous(),
                "gemm2: w2 must be a contiguous CUDA tensor.");
    TORCH_CHECK(w2.dtype() == torch::kBFloat16,
                "gemm2: w2 must be bfloat16.");
    TORCH_CHECK(w2.dim() == 3,
                "gemm2: w2 must be 3-D [num_experts, hidden_size, intermediate_size].");

    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.is_contiguous(),
                "gemm2: expert_offsets must be a contiguous CUDA tensor.");
    TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,
                "gemm2: expert_offsets must be int32.");
    TORCH_CHECK(expert_offsets.dim() == 1,
                "gemm2: expert_offsets must be 1-D [num_experts + 1].");

    const int total_slots     = static_cast<int>(swiglu_out.size(0));
    const int intermediate_size = static_cast<int>(swiglu_out.size(1));
    const int num_experts     = static_cast<int>(w2.size(0));
    const int hidden_size     = static_cast<int>(w2.size(1));

    TORCH_CHECK(w2.size(2) == intermediate_size,
                "gemm2: w2 intermediate_size dim does not match swiglu_out.");
    TORCH_CHECK(expert_offsets.size(0) == num_experts + 1,
                "gemm2: expert_offsets size must be num_experts + 1.");

    auto gemm2_out = torch::zeros(
        {total_slots, hidden_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(swiglu_out.device())
    );

    if (total_slots == 0) return gemm2_out;

    // Copy offsets to CPU for the loop (65 ints = 260 bytes).
    auto h_offsets_tensor = expert_offsets.cpu();
    const int* h_offsets  = h_offsets_tensor.data_ptr<int>();

    launch_moe_gemm2(
        reinterpret_cast<const __nv_bfloat16*>(swiglu_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(gemm2_out.data_ptr()),
        h_offsets,
        num_experts,
        intermediate_size,
        hidden_size,
        /*stream=*/0
    );

    return gemm2_out;
}

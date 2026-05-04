/**
 * @file gemm1.cu
 * @brief GEMM1 — Fused gate_proj + up_proj for all active experts.
 *
 * Position in the fused pipeline:
 *   dispatched [total_slots, 2048]
 *       │
 *       └── GEMM1 (this file) ──► gemm1_out [total_slots, 2816]
 *                                       │
 *                                       └── SwiGLU ──► [total_slots, 1408]
 *                                                           │
 *                                                           └── GEMM2 ──► [total_slots, 2048]
 *
 * Strategy — cuBLAS loop over experts (CUDA 11.5 compatible):
 * ─────────────────────────────────────────────────────────────
 * CUTLASS GroupedGemm requires CUTLASS 3.x (CUDA 12+). Since GCP runs
 * CUDA 11.5, we use a tight C++ loop calling cublasGemmEx once per expert.
 * This eliminates all Python overhead (GIL, object creation, dispatch) and
 * lets cuBLAS exploit Tensor Cores on Ampere (sm_86) for each call.
 *
 * For each expert e with m_e = offsets[e+1] - offsets[e] tokens:
 *   Computes: gemm1_out[offsets[e]:offsets[e+1]] =
 *             dispatched[offsets[e]:offsets[e+1]] @ packed_w1[e]^T
 *
 * Dimensions:  A[m_e, K] @ B[N, K]^T  →  C[m_e, N]
 *   K = hidden_size      = 2048
 *   N = double_intermed  = 2816  (= 2 × 1408)
 *
 * cuBLAS row-major trick:
 *   For row-major C = A @ B^T, call cuBLAS (col-major) as:
 *   cublasGemmEx(handle, OP_N, OP_T, N, m_e, K, &alpha,
 *                B_ptr, K,   ← B[N,K] row-major, treat as [K,N] col-major
 *                A_ptr, K,   ← A[m_e,K] row-major, treat as [K,m_e] col-major + OP_T
 *                &beta, C_ptr, N)
 *   Wait — the correct formula for row-major C[M×N] = A[M×K] @ B[N×K]^T is:
 *   cublasGemmEx(handle, OP_T, OP_N, N, M, K, &alpha,
 *                B_ptr, K, A_ptr, K, &beta, C_ptr, N, ...)
 *   See: https://docs.nvidia.com/cuda/cublas/#cublasgemmex
 *
 * Precision: BF16 inputs/outputs, FP32 accumulation (CUBLAS_COMPUTE_32F).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <stdexcept>
#include <string>
#include "moe/gemm1.h"

// ── Static cuBLAS handle (thread-safe lazy initialization) ───────────────────
// std::call_once guarantees the handle is created exactly once even if multiple
// threads race on the first call — the standard pattern for singleton resources
// in production CUDA libraries.
#include <mutex>

static cublasHandle_t  s_cublas_handle    = nullptr;
static std::once_flag  s_cublas_init_flag;

static cublasHandle_t get_cublas_handle() {
    std::call_once(s_cublas_init_flag, []() {
        cublasStatus_t status = cublasCreate(&s_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "gemm1: cublasCreate failed with status " +
                std::to_string(static_cast<int>(status)));
        }
    });
    return s_cublas_handle;
}
// ── cuBLAS error checking helper ─────────────────────────────────────────────
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                         \
                std::string("cuBLAS error in gemm1: ") +                     \
                std::to_string(static_cast<int>(status)));                    \
        }                                                                     \
    } while (0)

// ── Host launcher ─────────────────────────────────────────────────────────────

void launch_moe_gemm1(
    const __nv_bfloat16* dispatched,
    const __nv_bfloat16* packed_w1,
    __nv_bfloat16*       gemm1_out,
    const int*           expert_offsets_host,   // CPU-side array [num_experts + 1]
    int                  num_experts,
    int                  hidden_size,            // K = 2048
    int                  double_intermed,        // N = 2816
    cudaStream_t         stream
) {
    // Obtain the cuBLAS handle (lazily created static handle).
    cublasHandle_t handle = get_cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Byte strides for indexing into contiguous BF16 buffers.
    const int K = hidden_size;     // 2048
    const int N = double_intermed; // 2816

    for (int e = 0; e < num_experts; ++e) {
        const int start  = expert_offsets_host[e];
        const int end    = expert_offsets_host[e + 1];
        const int tokens = end - start;          // m_e

        if (tokens <= 0) continue;  // No tokens routed to this expert — skip.

        // Pointers to this expert's slice of the buffers.
        const __nv_bfloat16* A_ptr = dispatched + static_cast<ptrdiff_t>(start) * K;
        const __nv_bfloat16* B_ptr = packed_w1  + static_cast<ptrdiff_t>(e)     * N * K;
        __nv_bfloat16*       C_ptr = gemm1_out  + static_cast<ptrdiff_t>(start) * N;

        // Row-major C[tokens × N] = A[tokens × K] @ B[N × K]^T
        //
        // cuBLAS column-major equivalent (swap A↔B, flip transposes):
        //   op(B_cm)[N × K] @ op(A_cm)[K × tokens]  →  C_cm[N × tokens]
        //   = B_cm^N @ A_cm^T   with M=N, N=tokens, K=K
        //   → cublasGemmEx(OP_N, OP_T, N, tokens, K, B, K, A, K, C, N)
        //
        // Explanation:
        //   B is [N, K] row-major → cuBLAS sees [K, N] col-major → OP_N gives [K, N] → we want [N, K], so OP_T
        //   Wait — let me be precise:
        //
        //   Standard trick: for row-major C = op(A) * op(B), call cuBLAS as:
        //     cublasGemmEx(handle, op_B, op_A, cols_C, rows_C, K, alpha,
        //                  B, ldb, A, lda, beta, C, ldc)
        //
        //   We want C_rm[m×N] = A_rm[m×K] @ B_rm[N×K]^T
        //     op_A = OP_N, op_B = OP_T
        //     cuBLAS call (col-major, swapping A↔B and m↔N):
        //       cublasGemmEx(handle, OP_N, OP_T, N, m, K, &alpha,
        //                    B_ptr, K,     ← B[N,K] row-maj, ldb=K
        //                    A_ptr, K,     ← A[m,K] row-maj, lda=K
        //                    &beta, C_ptr, N, ...)
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,          // op on "A" in cuBLAS = B in our notation
            CUBLAS_OP_N,          // op on "B" in cuBLAS = A in our notation
            N,                    // M_cublas = N
            tokens,               // N_cublas = m_e
            K,                    // K_cublas = hidden_size
            &alpha,
            B_ptr, CUDA_R_16BF, K, // A_ptr, Atype, lda
            A_ptr, CUDA_R_16BF, K, // B_ptr, Btype, ldb
            &beta,
            C_ptr, CUDA_R_16BF, N, // C_ptr, Ctype, ldc
            CUBLAS_COMPUTE_32F,    // computeType
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
}

// ── PyTorch extension entry point ─────────────────────────────────────────────

/**
 * @brief Computes the fused gate+up projection for all active experts.
 *
 * @param dispatched      BF16 CUDA [total_slots, hidden_size]
 * @param packed_w1       BF16 CUDA [num_experts, 2*intermediate_size, hidden_size]
 * @param expert_offsets  int32 CUDA [num_experts + 1]
 * @return                BF16 CUDA [total_slots, 2*intermediate_size]
 */
torch::Tensor moe_gemm1_forward(
    torch::Tensor dispatched,
    torch::Tensor packed_w1,
    torch::Tensor expert_offsets
) {
    TORCH_CHECK(dispatched.is_cuda()     && dispatched.is_contiguous(),
                "gemm1: dispatched must be a contiguous CUDA tensor.");
    TORCH_CHECK(dispatched.dtype() == torch::kBFloat16,
                "gemm1: dispatched must be bfloat16.");
    TORCH_CHECK(dispatched.dim() == 2,
                "gemm1: dispatched must be 2-D [total_slots, hidden_size].");

    TORCH_CHECK(packed_w1.is_cuda()     && packed_w1.is_contiguous(),
                "gemm1: packed_w1 must be a contiguous CUDA tensor.");
    TORCH_CHECK(packed_w1.dtype() == torch::kBFloat16,
                "gemm1: packed_w1 must be bfloat16.");
    TORCH_CHECK(packed_w1.dim() == 3,
                "gemm1: packed_w1 must be 3-D [num_experts, 2*intermediate, hidden].");

    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.is_contiguous(),
                "gemm1: expert_offsets must be a contiguous CUDA tensor.");
    TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,
                "gemm1: expert_offsets must be int32.");
    TORCH_CHECK(expert_offsets.dim() == 1,
                "gemm1: expert_offsets must be 1-D [num_experts + 1].");

    const int total_slots    = static_cast<int>(dispatched.size(0));
    const int hidden_size    = static_cast<int>(dispatched.size(1));
    const int num_experts    = static_cast<int>(packed_w1.size(0));
    const int double_intermed = static_cast<int>(packed_w1.size(1));

    TORCH_CHECK(packed_w1.size(2) == hidden_size,
                "gemm1: packed_w1 hidden_size dim does not match dispatched.");
    TORCH_CHECK(expert_offsets.size(0) == num_experts + 1,
                "gemm1: expert_offsets size must be num_experts + 1.");

    // Allocate output.
    auto gemm1_out = torch::zeros(
        {total_slots, double_intermed},
        torch::TensorOptions().dtype(torch::kBFloat16).device(dispatched.device())
    );

    if (total_slots == 0) return gemm1_out;

    // Copy offsets to CPU for the loop — only 65 ints (260 bytes).
    auto h_offsets_tensor = expert_offsets.cpu();
    const int* h_offsets = h_offsets_tensor.data_ptr<int>();

    launch_moe_gemm1(
        reinterpret_cast<const __nv_bfloat16*>(dispatched.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(packed_w1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(gemm1_out.data_ptr()),
        h_offsets,
        num_experts,
        hidden_size,
        double_intermed,
        /*stream=*/0
    );

    return gemm1_out;
}

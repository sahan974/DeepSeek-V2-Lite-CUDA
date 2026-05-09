/**
 * @file kv_upproj.cu
 * @brief KV up-projection for MLA attention via cuBLAS.
 *
 * Position in the MLA pipeline:
 *   compressed_kv [bsz, seq, 512]
 *       │
 *       └── kv_a_layernorm ──► normed_kv [bsz, seq, 512]
 *                                   │
 *                                   └── kv_b_proj (this file) ──► kv_up [bsz, seq, 4096]
 *                                                                       │
 *                                               ┌─────────────────────┘
 *                                               │
 *                                    reshape + split
 *                                               │
 *                                  k_nope [bsz, 16, seq, 128]
 *                                  value  [bsz, 16, seq, 128]
 *
 * Strategy — single cublasGemmEx:
 * ──────────────────────────────
 * kv_b_proj is an nn.Linear(512, 4096, bias=False).
 * Its weight is stored row-major as [4096, 512].
 *
 * We compute: kv_up[M, N] = normed_kv[M, K] @ weight[N, K]^T
 *   M = bsz * seq      (number of token rows)
 *   K = 512            (kv_lora_rank)
 *   N = 4096           (num_heads * (qk_nope_head_dim + v_head_dim))
 *
 * cuBLAS column-major trick for row-major inputs:
 *   C_rm[M×N] = A_rm[M×K] @ B_rm[N×K]^T
 *   → cublasGemmEx(OP_N, OP_T, N, M, K, &alpha,
 *                  B_ptr, K,    (weight, ldb=K)
 *                  A_ptr, K,    (normed_kv, lda=K)
 *                  &beta, C_ptr, N)
 *
 * Precision: BF16 inputs/outputs, FP32 accumulation (CUBLAS_COMPUTE_32F).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <stdexcept>
#include <string>
#include <mutex>
#include "mla/kv_upproj.h"

// ── Static cuBLAS handle ─────────────────────────────────────────────────────

static cublasHandle_t s_kv_cublas_handle    = nullptr;
static std::once_flag  s_kv_cublas_init_flag;

static cublasHandle_t get_kv_cublas_handle() {
    std::call_once(s_kv_cublas_init_flag, []() {
        cublasStatus_t status = cublasCreate(&s_kv_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "kv_upproj: cublasCreate failed with status " +
                std::to_string(static_cast<int>(status)));
        }
    });
    return s_kv_cublas_handle;
}

#define KV_CUBLAS_CHECK(call)                                                \
    do {                                                                     \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            throw std::runtime_error(                                        \
                std::string("cuBLAS error in kv_upproj: ") +               \
                std::to_string(static_cast<int>(status)));                   \
        }                                                                    \
    } while (0)

// ── Host launcher ─────────────────────────────────────────────────────────────

void launch_kv_upproj(
    const __nv_bfloat16* normed_kv,
    const __nv_bfloat16* weight,
    __nv_bfloat16*       kv_up,
    int                  rows,
    int                  kv_lora_rank,
    int                  out_dim,
    cudaStream_t         stream
) {
    if (rows <= 0) return;

    cublasHandle_t handle = get_kv_cublas_handle();
    KV_CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const int M = rows;           // token rows  (bsz * seq_len)
    const int K = kv_lora_rank;   // 512
    const int N = out_dim;        // 4096

    // C_rm[M×N] = A_rm[M×K] @ B_rm[N×K]^T
    // cuBLAS col-major call (swap M↔N, flip transa/transb):
    //   cublasGemmEx(OP_N, OP_T, N, M, K, &alpha,
    //                weight[N×K] row-major, ldb=K,
    //                normed_kv[M×K] row-major, lda=K,
    //                &beta, kv_up[M×N] row-major, ldc=N)
    KV_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,          // transpose weight (OP applied to "A" in cuBLAS = B in ours)
        CUBLAS_OP_N,          // no transpose on normed_kv
        N,                    // M_cublas = N (output columns = 4096)
        M,                    // N_cublas = M (output rows = tokens)
        K,                    // K_cublas = 512
        &alpha,
        weight,   CUDA_R_16BF, K,  // A_cublas = weight  [N, K] row-major, lda=K
        normed_kv, CUDA_R_16BF, K, // B_cublas = normed_kv [M, K] row-major, ldb=K
        &beta,
        kv_up,    CUDA_R_16BF, N,  // C_cublas = kv_up [M, N] row-major, ldc=N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// ── PyTorch extension entry point ─────────────────────────────────────────────

/**
 * @brief Applies the kv_b_proj linear to the normalized KV latent.
 *
 * Equivalent to F.linear(normed_kv, weight) but executed via cuBLAS with
 * FP32 accumulation for full numerical accuracy on BF16 inputs.
 *
 * @param normed_kv  BF16 CUDA tensor [..., kv_lora_rank]  (contiguous)
 * @param weight     BF16 CUDA tensor [out_dim, kv_lora_rank]
 * @return           BF16 CUDA tensor [..., out_dim]
 */
torch::Tensor kv_upproj_forward(torch::Tensor normed_kv, torch::Tensor weight) {
    TORCH_CHECK(normed_kv.is_cuda() && normed_kv.is_contiguous(),
                "kv_upproj: normed_kv must be a contiguous CUDA tensor.");
    TORCH_CHECK(normed_kv.dtype() == torch::kBFloat16,
                "kv_upproj: normed_kv must be bfloat16.");
    TORCH_CHECK(normed_kv.dim() >= 2,
                "kv_upproj: normed_kv must have at least 2 dimensions.");

    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(),
                "kv_upproj: weight must be a contiguous CUDA tensor.");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16,
                "kv_upproj: weight must be bfloat16.");
    TORCH_CHECK(weight.dim() == 2,
                "kv_upproj: weight must be 2-D [out_dim, kv_lora_rank].");

    const int kv_lora_rank = static_cast<int>(normed_kv.size(-1));
    const int out_dim      = static_cast<int>(weight.size(0));

    TORCH_CHECK(weight.size(1) == kv_lora_rank,
                "kv_upproj: weight inner dim ", weight.size(1),
                " must match normed_kv last dim ", kv_lora_rank, ".");

    // Flatten leading dims → [rows, kv_lora_rank]
    const int rows = static_cast<int>(normed_kv.numel() / kv_lora_rank);
    auto normed_2d = normed_kv.view({rows, kv_lora_rank});

    // Allocate output preserving original leading shape.
    auto out_shape = normed_kv.sizes().vec();
    out_shape.back() = out_dim;
    auto kv_up = torch::empty(
        out_shape,
        torch::TensorOptions().dtype(torch::kBFloat16).device(normed_kv.device())
    );

    launch_kv_upproj(
        reinterpret_cast<const __nv_bfloat16*>(normed_2d.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(kv_up.data_ptr()),
        rows,
        kv_lora_rank,
        out_dim,
        /*stream=*/0
    );

    return kv_up;
}

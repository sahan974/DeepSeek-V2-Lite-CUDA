/**
 * @file bindings.cu
 * @brief PyBind11 entry point for the ds_kernels library.
 *
 * Consolidates all CUDA kernel bindings into a single translation unit to ensure
 * unified module registration and prevent multiple definition errors at link time.
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations for all kernel wrappers.

// hello.cu
void run_hello();

// moe_routing.cu
std::vector<torch::Tensor> moe_routing_forward(torch::Tensor logits, int top_k);

// moe_scan.cu
torch::Tensor moe_scan_forward(torch::Tensor expert_counts);

// moe_dispatch.cu
std::vector<torch::Tensor> moe_dispatch_forward(
    torch::Tensor input,
    torch::Tensor topk_indices,
    torch::Tensor topk_weights,
    torch::Tensor expert_offsets
);

// moe/combine.cu
torch::Tensor moe_combine_forward(
    torch::Tensor expert_output,
    torch::Tensor token_map,
    torch::Tensor dispatched_weights,
    int           num_tokens
);

// moe/swiglu.cu
torch::Tensor swiglu_forward(torch::Tensor input);

// moe/gemm1.cu
torch::Tensor moe_gemm1_forward(
    torch::Tensor dispatched,
    torch::Tensor packed_w1,
    torch::Tensor expert_offsets
);

// moe/gemm2.cu
torch::Tensor moe_gemm2_forward(
    torch::Tensor swiglu_out,
    torch::Tensor w2,
    torch::Tensor expert_offsets
);

// mla/rms_norm.cu
torch::Tensor rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double        eps
);

// mla/kv_upproj.cu
torch::Tensor kv_upproj_forward(
    torch::Tensor normed_kv,
    torch::Tensor weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_hello",
          &run_hello,
          "Executes the hello world CUDA kernel (toolchain smoke test).");

    m.def("moe_routing",
          &moe_routing_forward,
          "Fused softmax and top-K MoE routing kernel.\n"
          "Args:\n"
          "  logits (Tensor): float32 CUDA tensor [num_tokens, num_experts]\n"
          "  top_k  (int):    number of experts to select per token (default 6)\n"
          "Returns:\n"
          "  topk_indices  (Tensor): int32   [num_tokens, top_k]\n"
          "  topk_weights  (Tensor): float32 [num_tokens, top_k]\n"
          "  expert_counts (Tensor): int32   [num_experts]",
          py::arg("logits"),
          py::arg("top_k") = 6);

    m.def("moe_scan",
          &moe_scan_forward,
          "Exclusive prefix sum of expert token counts to produce dispatch offsets.\n"
          "Args:\n"
          "  expert_counts (Tensor): int32 CUDA tensor [num_experts]\n"
          "Returns:\n"
          "  expert_offsets (Tensor): int32 CUDA tensor [num_experts + 1]");

    m.def("moe_dispatch",
          &moe_dispatch_forward,
          "Scatters token hidden states into an expert-contiguous output buffer.\n"
          "Args:\n"
          "  input          (Tensor): bfloat16 CUDA tensor [num_tokens, hidden_size]\n"
          "  topk_indices   (Tensor): int32    CUDA tensor [num_tokens, top_k]\n"
          "  topk_weights   (Tensor): float32  CUDA tensor [num_tokens, top_k]\n"
          "  expert_offsets (Tensor): int32    CUDA tensor [num_experts + 1]\n"
          "Returns:\n"
          "  dispatched         (Tensor): bfloat16 [num_tokens * top_k, hidden_size]\n"
          "  token_map          (Tensor): int32    [num_tokens * top_k]\n"
          "  dispatched_weights (Tensor): float32  [num_tokens * top_k]");

    m.def("moe_combine",
          &moe_combine_forward,
          "Accumulates weighted expert outputs back to token-ordered positions.\n"
          "Args:\n"
          "  expert_output      (Tensor): bfloat16 CUDA tensor [total_slots, hidden_size]\n"
          "  token_map          (Tensor): int32    CUDA tensor [total_slots]\n"
          "  dispatched_weights (Tensor): float32  CUDA tensor [total_slots]\n"
          "  num_tokens         (int):    number of original tokens\n"
          "Returns:\n"
          "  final_output (Tensor): bfloat16 [num_tokens, hidden_size]",
          py::arg("expert_output"),
          py::arg("token_map"),
          py::arg("dispatched_weights"),
          py::arg("num_tokens"));

    m.def("swiglu",
          &swiglu_forward,
          "SwiGLU activation: silu(gate_half) * up_half.\n"
          "Args:\n"
          "  input (Tensor): bfloat16 CUDA tensor [total_tokens, 2*intermediate_size]\n"
          "Returns:\n"
          "  output (Tensor): bfloat16 CUDA tensor [total_tokens, intermediate_size]");

    m.def("moe_gemm1",
          &moe_gemm1_forward,
          "Batched gate+up projection for all active experts via cuBLAS.\n"
          "Args:\n"
          "  dispatched     (Tensor): bfloat16 CUDA [total_slots, hidden_size]\n"
          "  packed_w1      (Tensor): bfloat16 CUDA [num_experts, 2*intermediate, hidden]\n"
          "  expert_offsets (Tensor): int32    CUDA [num_experts + 1]\n"
          "Returns:\n"
          "  gemm1_out (Tensor): bfloat16 CUDA [total_slots, 2*intermediate_size]");

    m.def("moe_gemm2",
          &moe_gemm2_forward,
          "Batched down_proj for all active experts via cuBLAS.\n"
          "Args:\n"
          "  swiglu_out     (Tensor): bfloat16 CUDA [total_slots, intermediate_size]\n"
          "  w2             (Tensor): bfloat16 CUDA [num_experts, hidden_size, intermediate_size]\n"
          "  expert_offsets (Tensor): int32    CUDA [num_experts + 1]\n"
          "Returns:\n"
          "  gemm2_out (Tensor): bfloat16 CUDA [total_slots, hidden_size]");

    m.def("rms_norm",
          &rms_norm_forward,
          "RMSNorm over the last dimension of a BF16 tensor.\n"
          "Args:\n"
          "  input  (Tensor): bfloat16 CUDA tensor [..., hidden_size]\n"
          "  weight (Tensor): float32  CUDA tensor [hidden_size]\n"
          "  eps    (float):  variance epsilon (default 1e-6)\n"
          "Returns:\n"
          "  output (Tensor): bfloat16 CUDA tensor [..., hidden_size]",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps") = 1e-6);

    m.def("kv_upproj",
          &kv_upproj_forward,
          "KV up-projection: normed_kv @ kv_b_proj.weight.T via cuBLAS.\n"
          "Args:\n"
          "  normed_kv (Tensor): bfloat16 CUDA tensor [..., kv_lora_rank]\n"
          "  weight    (Tensor): bfloat16 CUDA tensor [out_dim, kv_lora_rank]\n"
          "Returns:\n"
          "  kv_up (Tensor): bfloat16 CUDA tensor [..., out_dim]");
}

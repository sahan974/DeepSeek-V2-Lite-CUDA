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
}

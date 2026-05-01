/**
 * @file bindings.cu
 * @brief PyBind11 entry point for the ds_kernels library.
 *
 * Consolidates all CUDA kernel bindings into a single translation unit to ensure
 * unified module registration and prevent multiple definition errors.
 */

#include <torch/extension.h>
#include <vector>

// ── Forward declarations ───────────────────────────────────────────────────
// hello.cu
void run_hello();

// moe_routing.cu
std::vector<torch::Tensor> moe_routing_forward(torch::Tensor logits, int top_k);

// moe_scan.cu
torch::Tensor moe_scan_forward(torch::Tensor expert_counts);

// ── Module registration ────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ── Toolchain verification ──────────────────────────────────────────────
    m.def("run_hello",
          &run_hello,
          "Run the hello world CUDA kernel (toolchain smoke test)");

    // ── MoE Routing ─────────────────────────────────────────────────────────
    m.def("moe_routing",
          &moe_routing_forward,
          "Fused softmax + top-K MoE routing kernel.\n"
          "Args:\n"
          "  logits (Tensor): float32 CUDA tensor of shape [num_tokens, num_experts]\n"
          "  top_k  (int):    number of experts to select per token (default 6)\n"
          "Returns:\n"
          "  topk_indices  (Tensor): int32   [num_tokens, top_k]\n"
          "  topk_weights  (Tensor): float32 [num_tokens, top_k]\n"
          "  expert_counts (Tensor): int32   [num_experts]",
          py::arg("logits"),
          py::arg("top_k") = 6);

    // ── MoE Scan (Prefix Sum) ───────────────────────────────────────────────
    m.def("moe_scan",
          &moe_scan_forward,
          "Exclusive prefix sum of expert counts to produce offsets.\n"
          "Args:\n"
          "  expert_counts (Tensor): int32 CUDA tensor [num_experts]\n"
          "Returns:\n"
          "  expert_offsets (Tensor): int32 CUDA tensor [num_experts + 1]");
}

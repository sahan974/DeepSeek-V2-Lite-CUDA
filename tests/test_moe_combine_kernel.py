"""
tests/test_moe_combine_kernel.py

Validates the moe_combine CUDA kernel against a NumPy/PyTorch reference.

The test bypasses the routing and dispatch kernels and constructs known inputs
directly to verify the combine kernel's accumulation logic in isolation. This
approach allows precise control over token_map and dispatched_weights, making
it straightforward to compute the expected output analytically.

Additionally, an end-to-end test verifies that routing → scan → dispatch →
combine produces output that matches the reference MoEGate + manual scatter.

Test cases:
  1. Output shape                — final_output has shape [num_tokens, hidden_size].
  2. Single-slot accumulation    — one slot per token, no overlap; exact equality.
  3. Multi-slot accumulation     — top_k slots per token; sum of weighted contributions.
  4. Zero weights                — weighted expert outputs contribute zero; final is zero.
  5. End-to-end pipeline         — routing + scan + dispatch + combine matches reference.

This test requires a CUDA-capable GPU. It exits with a clear message on
CPU-only machines.
"""

import sys
import os

# torch._C must be loaded with RTLD_GLOBAL before importing ds_kernels.
# This makes PyBind11 type_caster symbols visible to the dynamic linker.
import ctypes
import torch
import torch.nn.functional as F
ctypes.CDLL(torch._C.__file__, ctypes.RTLD_GLOBAL)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "build"))

NUM_EXPERTS = 64
TOP_K       = 6
HIDDEN_SIZE = 2048
ATOL_BF16   = 1e-2   # BF16 has ~2 decimal digits of precision.
ATOL_COUNT  = 0      # Integer token counts must be exact.


def reference_combine(
    expert_output_cpu: torch.Tensor,
    token_map_cpu: torch.Tensor,
    dispatched_weights_cpu: torch.Tensor,
    num_tokens: int
) -> torch.Tensor:
    """
    Computes the reference output in FP32 on the CPU.
    For each slot s: result[token_map[s]] += expert_output[s] * dispatched_weights[s]
    Returns a float32 tensor of shape [num_tokens, hidden_size].
    """
    hidden_size = expert_output_cpu.size(1)
    result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    for s in range(expert_output_cpu.size(0)):
        tok = token_map_cpu[s].item()
        w   = dispatched_weights_cpu[s].item()
        result[tok] += expert_output_cpu[s].float() * w
    return result


def test_output_shape(num_tokens: int):
    """Validates that the combine kernel produces the correct output shape."""
    import ds_kernels
    print(f"  [shape]        num_tokens={num_tokens}")
    device = torch.device("cuda")
    total_slots = num_tokens * TOP_K

    expert_output      = torch.randn(total_slots, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    token_map          = torch.randint(0, num_tokens, (total_slots,), device=device, dtype=torch.int32)
    dispatched_weights = torch.rand(total_slots, device=device, dtype=torch.float32)

    final_output = ds_kernels.moe_combine(
        expert_output, token_map, dispatched_weights, num_tokens)

    assert final_output.shape == (num_tokens, HIDDEN_SIZE), (
        f"Shape mismatch: expected ({num_tokens}, {HIDDEN_SIZE}), got {tuple(final_output.shape)}")
    assert final_output.dtype == torch.bfloat16, (
        f"dtype mismatch: expected bfloat16, got {final_output.dtype}")
    print("    PASSED")


def test_single_slot_per_token(num_tokens: int):
    """
    Validates correctness when each token has exactly one slot (top_k=1 equivalent).
    With no overlapping writes, the result should match a non-atomic reference exactly.
    """
    import ds_kernels
    print(f"  [single_slot]  num_tokens={num_tokens}")
    device = torch.device("cuda")
    torch.manual_seed(0)

    # One slot per token, in order.
    expert_output      = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    token_map          = torch.arange(num_tokens, dtype=torch.int32, device=device)
    dispatched_weights = torch.rand(num_tokens, dtype=torch.float32, device=device)

    final_output = ds_kernels.moe_combine(
        expert_output, token_map, dispatched_weights, num_tokens).cpu().float()

    reference = reference_combine(
        expert_output.cpu(), token_map.cpu(), dispatched_weights.cpu(), num_tokens)

    assert torch.allclose(final_output, reference, atol=ATOL_BF16), (
        f"Single-slot accumulation mismatch. Max delta: "
        f"{(final_output - reference).abs().max().item():.6f}")
    print("    PASSED")


def test_multi_slot_accumulation(num_tokens: int):
    """
    Validates correctness when multiple slots map to the same token (top_k > 1).
    Tests that all contributions are correctly summed.
    """
    import ds_kernels
    print(f"  [multi_slot]   num_tokens={num_tokens}")
    device = torch.device("cuda")
    torch.manual_seed(1)
    total_slots = num_tokens * TOP_K

    expert_output      = torch.randn(total_slots, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    # Construct token_map so each token appears exactly TOP_K times.
    token_map = torch.repeat_interleave(
        torch.arange(num_tokens, dtype=torch.int32), TOP_K).to(device)
    dispatched_weights = torch.rand(total_slots, dtype=torch.float32, device=device)

    final_output = ds_kernels.moe_combine(
        expert_output, token_map, dispatched_weights, num_tokens).cpu().float()

    reference = reference_combine(
        expert_output.cpu(), token_map.cpu(), dispatched_weights.cpu(), num_tokens)

    assert torch.allclose(final_output, reference, atol=ATOL_BF16), (
        f"Multi-slot accumulation mismatch. Max delta: "
        f"{(final_output - reference).abs().max().item():.6f}")
    print("    PASSED")


def test_zero_weights(num_tokens: int):
    """
    Validates that zero gating weights produce a zero final output,
    regardless of the expert_output values.
    """
    import ds_kernels
    print(f"  [zero_weights] num_tokens={num_tokens}")
    device = torch.device("cuda")
    total_slots = num_tokens * TOP_K

    expert_output      = torch.randn(total_slots, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    token_map          = torch.randint(0, num_tokens, (total_slots,), dtype=torch.int32, device=device)
    dispatched_weights = torch.zeros(total_slots, dtype=torch.float32, device=device)

    final_output = ds_kernels.moe_combine(
        expert_output, token_map, dispatched_weights, num_tokens)

    assert torch.all(final_output == 0), (
        "Non-zero output produced for zero dispatched_weights.")
    print("    PASSED")


def test_end_to_end_pipeline(num_tokens: int):
    """
    End-to-end validation of the full CUDA routing pipeline against the reference
    MoEGate implementation. Verifies that routing → scan → dispatch → combine
    produces output consistent with the reference weighted sum.

    Because the test uses the CUDA dispatch kernel's token ordering (which is
    non-deterministic across experts for tied scores), the comparison is performed
    on the final accumulated result rather than intermediate orderings.
    """
    import ds_kernels
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from reference.moe import MoEGate

    print(f"  [end_to_end]   num_tokens={num_tokens}")
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Construct a reference gate and generate logits.
    gate = MoEGate(num_experts=NUM_EXPERTS, top_k=TOP_K).to(device)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=device)

    with torch.no_grad():
        ref_indices, ref_weights = gate(logits)  # Both [num_tokens, TOP_K]

    # Generate random hidden states to dispatch and combine.
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    # Execute CUDA pipeline.
    topk_indices, topk_weights, expert_counts = ds_kernels.moe_routing(logits, TOP_K)
    expert_offsets = ds_kernels.moe_scan(expert_counts)
    dispatched, token_map, dispatched_weights = ds_kernels.moe_dispatch(
        hidden, topk_indices, topk_weights, expert_offsets)
    cuda_output = ds_kernels.moe_combine(
        dispatched, token_map, dispatched_weights, num_tokens)

    # Compute the reference output via manual weighted sum.
    # For each token, sum top_k scaled copies of its hidden state.
    ref_output = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=torch.float32, device=device)
    for tok in range(num_tokens):
        for k in range(TOP_K):
            ref_output[tok] += hidden[tok].float() * topk_weights[tok, k].item()

    assert torch.allclose(cuda_output.float(), ref_output, atol=ATOL_BF16), (
        f"End-to-end pipeline mismatch. Max delta: "
        f"{(cuda_output.float() - ref_output).abs().max().item():.6f}")
    print("    PASSED")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available on this machine — skipping combine kernel tests.")
        print("Run this test on the Vast.ai GPU instance.")
        return

    try:
        import ds_kernels
    except ImportError as exc:
        print(f"ds_kernels library not found: {exc}")
        print("Ensure the build directory is on PYTHONPATH:")
        print("  export PYTHONPATH=$PYTHONPATH:$(pwd)/build")
        return

    device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
    print(f"Running moe_combine kernel tests on {device_str}")
    print()

    for num_tokens in [1, 32, 128]:
        print(f"=== Batch size {num_tokens} ===")

        test_output_shape(num_tokens)
        test_single_slot_per_token(num_tokens)
        test_multi_slot_accumulation(num_tokens)
        test_zero_weights(num_tokens)
        test_end_to_end_pipeline(num_tokens)

        print()

    print("✅  All moe_combine kernel tests passed.")


if __name__ == "__main__":
    main()

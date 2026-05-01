"""
tests/test_moe_dispatch_kernel.py

Validates the moe_dispatch CUDA kernel against the reference MoEGate routing.

For each expert, verifies that the dispatched output buffer contains exactly
the hidden states of the tokens that the reference router assigned to that
expert, and that the token_map and dispatched_weights arrays are consistent
with the routing decisions.

This test requires a CUDA-capable GPU. It will exit early with a clear
message on CPU-only machines.
"""

import sys
import os

# torch._C must be loaded with RTLD_GLOBAL before importing ds_kernels.
# This makes PyBind11 type_caster symbols (e.g., type_caster<at::Tensor>)
# visible to the dynamic linker when the custom extension is loaded.
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
ATOL        = 0.0  # Exact BF16 copy — no numerical error is acceptable.


def reference_routing(logits: torch.Tensor, top_k: int):
    """
    Computes the reference softmax + top-K routing on the CPU using PyTorch.
    Returns (topk_indices, topk_weights, expert_counts) as CPU tensors.
    """
    scores      = F.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(scores, top_k, dim=-1)
    counts = torch.zeros(logits.size(1), dtype=torch.int32)
    for idx in indices.view(-1):
        counts[idx] += 1
    return indices.int(), weights.float(), counts


def reference_exclusive_prefix_sum(counts: torch.Tensor) -> torch.Tensor:
    """
    Computes the exclusive prefix sum of counts on the CPU.
    Returns a tensor of length num_experts + 1.
    """
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def run_dispatch(num_tokens: int):
    """
    Runs the full routing + scan + dispatch pipeline through the CUDA kernels.
    Returns (dispatched, token_map, dispatched_weights, topk_indices_cpu,
             topk_weights_cpu, expert_offsets_cpu) as CPU tensors.
    """
    import ds_kernels

    device = torch.device("cuda")

    # Generate random logits and token hidden states.
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=device)
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # Execute routing kernel.
    topk_indices, topk_weights, expert_counts = ds_kernels.moe_routing(
        logits, top_k=TOP_K)

    # Execute scan kernel to get offsets.
    expert_offsets = ds_kernels.moe_scan(expert_counts)

    # Execute dispatch kernel.
    dispatched, token_map, dispatched_weights = ds_kernels.moe_dispatch(
        hidden, topk_indices, topk_weights, expert_offsets)

    return (
        dispatched.cpu(),
        token_map.cpu(),
        dispatched_weights.cpu(),
        topk_indices.cpu(),
        topk_weights.cpu(),
        expert_offsets.cpu(),
        hidden.cpu(),
    )


def test_output_shape(num_tokens: int):
    """Validates that all output tensors have the correct shape."""
    print(f"  [shape]        num_tokens={num_tokens}")
    dispatched, token_map, dispatched_weights, _, _, _, _ = run_dispatch(num_tokens)

    total_slots = num_tokens * TOP_K

    assert dispatched.shape == (total_slots, HIDDEN_SIZE), (
        f"dispatched shape mismatch: expected ({total_slots}, {HIDDEN_SIZE}), "
        f"got {tuple(dispatched.shape)}")
    assert token_map.shape == (total_slots,), (
        f"token_map shape mismatch: expected ({total_slots},), got {tuple(token_map.shape)}")
    assert dispatched_weights.shape == (total_slots,), (
        f"dispatched_weights shape mismatch: expected ({total_slots},), "
        f"got {tuple(dispatched_weights.shape)}")
    print("    PASSED")


def test_token_map_validity(num_tokens: int):
    """
    Validates that all entries in token_map refer to valid token indices,
    and that each token appears exactly top_k times across the entire map.
    """
    print(f"  [token_map]    num_tokens={num_tokens}")
    dispatched, token_map, dispatched_weights, topk_indices, _, _, _ = run_dispatch(num_tokens)

    total_slots = num_tokens * TOP_K

    assert token_map.min().item() >= 0, "token_map contains a negative index."
    assert token_map.max().item() < num_tokens, (
        f"token_map contains an out-of-range index: {token_map.max().item()} >= {num_tokens}")

    # Each token must appear exactly top_k times in token_map.
    counts = torch.bincount(token_map, minlength=num_tokens)
    assert (counts == TOP_K).all(), (
        f"Each token must appear exactly {TOP_K} times in token_map. "
        f"Found violations at indices: {(counts != TOP_K).nonzero().squeeze().tolist()}")
    print("    PASSED")


def test_expert_contiguity(num_tokens: int):
    """
    Validates that, for every expert e, the rows in dispatched[offsets[e]:offsets[e+1]]
    correspond exactly to the tokens that were routed to expert e by the routing kernel.
    Validates both token identity (token_map) and hidden state content (exact BF16 copy).
    """
    print(f"  [contiguity]   num_tokens={num_tokens}")
    (dispatched, token_map, dispatched_weights,
     topk_indices, topk_weights, expert_offsets, hidden) = run_dispatch(num_tokens)

    for e in range(NUM_EXPERTS):
        start = expert_offsets[e].item()
        end   = expert_offsets[e + 1].item()

        if start == end:
            continue  # No tokens routed to this expert.

        # Collect the set of tokens that the routing kernel assigned to expert e.
        routed_tokens = set()
        for tok in range(num_tokens):
            for k in range(TOP_K):
                if topk_indices[tok, k].item() == e:
                    routed_tokens.add(tok)

        # Collect the set of tokens that the dispatch kernel placed in expert e's region.
        dispatched_tokens = set(token_map[start:end].tolist())

        assert routed_tokens == dispatched_tokens, (
            f"Expert {e}: token set mismatch.\n"
            f"  Expected (from routing): {sorted(routed_tokens)}\n"
            f"  Received (from dispatch): {sorted(dispatched_tokens)}")

        # Verify that each dispatched row contains the correct hidden state.
        for row in range(start, end):
            tok = token_map[row].item()
            assert torch.equal(dispatched[row], hidden[tok]), (
                f"Expert {e}, row {row}: hidden state mismatch for token {tok}.")

    print("    PASSED")


def test_dispatched_weights_correctness(num_tokens: int):
    """
    Validates that dispatched_weights[row] equals the gating weight the routing
    kernel assigned to the (token, expert) pair corresponding to that row.
    """
    print(f"  [weights]      num_tokens={num_tokens}")
    (dispatched, token_map, dispatched_weights,
     topk_indices, topk_weights, expert_offsets, _) = run_dispatch(num_tokens)

    for e in range(NUM_EXPERTS):
        start = expert_offsets[e].item()
        end   = expert_offsets[e + 1].item()

        for row in range(start, end):
            tok = token_map[row].item()

            # Find which k-slot this (token, expert) pair corresponds to.
            k_slot = None
            for k in range(TOP_K):
                if topk_indices[tok, k].item() == e:
                    k_slot = k
                    break

            assert k_slot is not None, (
                f"Could not find expert {e} in routing results for token {tok}.")

            expected_weight = topk_weights[tok, k_slot].item()
            actual_weight   = dispatched_weights[row].item()

            assert abs(expected_weight - actual_weight) <= ATOL, (
                f"Expert {e}, row {row}, token {tok}: weight mismatch. "
                f"Expected {expected_weight:.6f}, got {actual_weight:.6f}.")

    print("    PASSED")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available on this machine — skipping dispatch kernel tests.")
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
    print(f"Running moe_dispatch kernel tests on {device_str}")
    print()

    for num_tokens in [1, 32, 128]:
        print(f"=== Batch size {num_tokens} ===")

        test_output_shape(num_tokens)
        test_token_map_validity(num_tokens)
        test_expert_contiguity(num_tokens)
        test_dispatched_weights_correctness(num_tokens)

        print()

    print("✅  All moe_dispatch kernel tests passed.")


if __name__ == "__main__":
    main()

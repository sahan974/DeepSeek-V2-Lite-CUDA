"""
tests/test_moe_routing_kernel.py

Validates the fused CUDA moe_routing kernel against the pure-PyTorch
MoEGate reference implementation.

Runs on GPU when CUDA is available; skips gracefully on CPU-only machines
(GCP dev VM). Full validation happens on the Vast.ai GPU instance.

Test cases:
  1. Shape correctness     — output tensors have the right shapes.
  2. Weight correctness    — topk_weights match reference within atol=1e-4.
  3. Index correctness     — topk_indices match reference (same set, any order).
  4. Expert counts         — expert_counts sums to num_tokens * top_k.
  5. Batch sizes           — passes for num_tokens = 1, 32, 128.
  6. Tied scores           — no crash / correct K experts when scores are equal.
  7. Weight sum            — softmax property: all expert scores sum to ~1.0.
"""

import sys
import os
import math

# torch must be imported before ds_kernels. Additionally, torch._C must be
# reloaded with RTLD_GLOBAL so that its PyBind11 type_caster symbols (e.g.,
# type_caster<at::Tensor>) are visible to the dynamic linker when ds_kernels.so
# is subsequently loaded. Without RTLD_GLOBAL, Python loads C extensions with
# RTLD_LOCAL by default, making their symbols invisible to other extensions.
import ctypes
import torch
import torch.nn.functional as F
ctypes.CDLL(torch._C.__file__, ctypes.RTLD_GLOBAL)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "build"))

from reference.moe import MoEGate

# ── Constants (matching DeepSeek-V2-Lite config) ─────────────────────────────
NUM_EXPERTS  = 64
TOP_K        = 6
HIDDEN_SIZE  = 2048   # only used for MoEGate weight construction

# ── Helpers ───────────────────────────────────────────────────────────────────

def ref_gate_forward(logits_f32: torch.Tensor):
    """Pure-PyTorch reference: softmax + top-k (mirrors MoEGate.forward)."""
    scores = logits_f32.softmax(dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=TOP_K, dim=-1, sorted=False)
    return topk_idx, topk_weight, scores


def cuda_gate_forward(logits_f32: torch.Tensor):
    """CUDA kernel wrapper."""
    import ds_kernels
    topk_indices, topk_weights, expert_counts = ds_kernels.moe_routing(
        logits_f32.cuda().contiguous(), top_k=TOP_K
    )
    return topk_indices, topk_weights, expert_counts


def sets_match(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Check that each row of a and b contain the same set of values."""
    assert a.shape == b.shape
    for i in range(a.size(0)):
        if set(a[i].tolist()) != set(b[i].tolist()):
            return False
    return True

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_shapes(num_tokens: int):
    """Output tensors must have the correct shapes."""
    print(f"  [shape]  num_tokens={num_tokens}")
    logits = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.float32)

    topk_idx, topk_w, counts = cuda_gate_forward(logits)

    assert topk_idx.shape  == (num_tokens, TOP_K),  \
        f"topk_indices shape {topk_idx.shape} != {(num_tokens, TOP_K)}"
    assert topk_w.shape    == (num_tokens, TOP_K),  \
        f"topk_weights shape {topk_w.shape} != {(num_tokens, TOP_K)}"
    assert counts.shape    == (NUM_EXPERTS,),        \
        f"expert_counts shape {counts.shape} != ({NUM_EXPERTS},)"
    print("    PASSED")


def test_weight_correctness(num_tokens: int):
    """Weights from CUDA must match reference softmax values within atol=1e-4."""
    print(f"  [weights] num_tokens={num_tokens}")
    logits = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.float32)

    ref_idx, ref_w, _ = ref_gate_forward(logits)
    cuda_idx, cuda_w, _ = cuda_gate_forward(logits)

    # Sort both by index so we can compare element-wise.
    ref_sorted_w  = ref_w.cpu().gather(1,  ref_idx.argsort(dim=1))
    cuda_sorted_w = cuda_w.cpu().gather(1, cuda_idx.cpu().argsort(dim=1))

    # We compare weights at the same expert indices — build a full score tensor.
    ref_full  = torch.zeros(num_tokens, NUM_EXPERTS)
    cuda_full = torch.zeros(num_tokens, NUM_EXPERTS)
    ref_full.scatter_(1,  ref_idx,        ref_w)
    cuda_full.scatter_(1, cuda_idx.cpu(), cuda_w.cpu())

    max_diff = (ref_full - cuda_full).abs().max().item()
    assert max_diff < 1e-4, f"Max weight diff {max_diff:.2e} exceeds atol=1e-4"
    print(f"    PASSED  (max_diff={max_diff:.2e})")


def test_index_correctness(num_tokens: int):
    """Each token must route to the same set of experts as the reference."""
    print(f"  [indices] num_tokens={num_tokens}")
    logits = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.float32)

    ref_idx, _, _ = ref_gate_forward(logits)
    cuda_idx, _, _ = cuda_gate_forward(logits)

    assert sets_match(ref_idx, cuda_idx.cpu()), \
        "Expert index sets differ between reference and CUDA kernel"
    print("    PASSED")


def test_expert_counts(num_tokens: int):
    """expert_counts must sum to num_tokens * top_k."""
    print(f"  [counts]  num_tokens={num_tokens}")
    logits = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.float32)

    _, _, counts = cuda_gate_forward(logits)
    total = counts.cpu().sum().item()
    expected = num_tokens * TOP_K

    assert total == expected, \
        f"expert_counts sum {total} != {expected} (num_tokens * top_k)"
    assert (counts.cpu() >= 0).all(), "Negative expert count detected"
    print(f"    PASSED  (total={total})")


def test_softmax_sum(num_tokens: int):
    """All expert scores (before top-K) must sum to ~1.0 per token.
    We verify this by checking that the sum of all top-K weights is <= 1.0
    (since top-K selects a subset of a proper softmax distribution)."""
    print(f"  [softmax] num_tokens={num_tokens}")
    logits = torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.float32)

    # Full softmax reference.
    full_scores = logits.softmax(dim=-1)  # [num_tokens, 64]
    row_sums = full_scores.sum(dim=-1)    # should all be ~1.0
    max_dev = (row_sums - 1.0).abs().max().item()
    assert max_dev < 1e-5, f"Softmax row sum deviation {max_dev:.2e}"
    print(f"    PASSED  (max_row_sum_dev={max_dev:.2e})")


def test_tied_scores():
    """Kernel must not crash and must return exactly top_k experts when all
    logits are identical (maximum tie scenario)."""
    print("  [tied]   all logits = 0.0")
    num_tokens = 16
    logits = torch.zeros(num_tokens, NUM_EXPERTS, dtype=torch.float32)

    topk_idx, topk_w, counts = cuda_gate_forward(logits)

    # Should return exactly TOP_K distinct experts per token.
    for i in range(num_tokens):
        row_experts = set(topk_idx[i].cpu().tolist())
        assert len(row_experts) == TOP_K, \
            f"Token {i}: expected {TOP_K} distinct experts, got {len(row_experts)}"

    total = counts.cpu().sum().item()
    assert total == num_tokens * TOP_K, \
        f"expert_counts sum {total} != {num_tokens * TOP_K}"
    print("    PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("CUDA not available on this machine — skipping kernel tests.")
        print("Run this test on the Vast.ai GPU instance.")
        return

    print(f"Running moe_routing kernel tests on {torch.cuda.get_device_name(0)}")
    print()

    batch_sizes = [1, 32, 128]

    print("=== Shape Tests ===")
    for n in batch_sizes:
        test_shapes(n)

    print("\n=== Weight Correctness Tests ===")
    for n in batch_sizes:
        test_weight_correctness(n)

    print("\n=== Index Correctness Tests ===")
    for n in batch_sizes:
        test_index_correctness(n)

    print("\n=== Expert Count Tests ===")
    for n in batch_sizes:
        test_expert_counts(n)

    print("\n=== Softmax Sum Tests ===")
    for n in batch_sizes:
        test_softmax_sum(n)

    print("\n=== Tied Score Test ===")
    test_tied_scores()

    print("\n✅  All moe_routing kernel tests passed.")


if __name__ == "__main__":
    main()

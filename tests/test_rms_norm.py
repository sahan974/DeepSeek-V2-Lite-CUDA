"""
tests/test_rms_norm.py
======================
Unit test for the CUDA RMSNorm kernel (ds_kernels.rms_norm).

Tests:
  1. Shape test     — output has same shape as input.
  2. Dtype test     — output is bfloat16.
  3. Accuracy test  — matches reference RMSNorm.forward() within atol=1e-2.
  4. 3-D input      — kernel flattens leading dims correctly.
  5. Stability test — no NaN or Inf in output.

Run:
    cd /workspace
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/test_rms_norm.py
"""

import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import ds_kernels
from reference.mla import RMSNorm

DEVICE = "cuda"
ATOL   = 1e-2


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm matching the Python implementation exactly."""
    model = RMSNorm(x.shape[-1], eps=eps)
    model.weight = torch.nn.Parameter(weight.clone())
    model = model.to(x.device).to(torch.float32)
    with torch.no_grad():
        x_f32 = x.float()
        out   = model(x_f32)
    return out.to(x.dtype)


def make_inputs(rows, hidden, seed=0):
    """Creates a random BF16 input and float32 weight."""
    torch.manual_seed(seed)
    x      = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.5
    weight = torch.ones(hidden, device=DEVICE, dtype=torch.float32)
    return x, weight


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_shape():
    print("  [1] Shape test ...", end=" ")
    x, w = make_inputs(rows=4, hidden=512)
    out  = ds_kernels.rms_norm(x, w, 1e-6)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("PASS")


def test_dtype():
    print("  [2] Dtype test ...", end=" ")
    x, w = make_inputs(rows=4, hidden=512)
    out  = ds_kernels.rms_norm(x, w, 1e-6)
    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"
    print("PASS")


def test_accuracy_identity_weight():
    """With weight=ones, CUDA output should match reference within atol."""
    print("  [3] Accuracy test (identity weight, hidden=512) ...", end=" ")
    rows, hidden = 8, 512
    x, w = make_inputs(rows=rows, hidden=hidden, seed=42)

    cuda_out = ds_kernels.rms_norm(x, w, 1e-6)
    ref_out  = ref_rms_norm(x, w, eps=1e-6)

    max_diff = (cuda_out.float() - ref_out.float()).abs().max().item()
    assert max_diff < ATOL, f"Max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_accuracy_random_weight():
    """With random weights, CUDA output should match reference."""
    print("  [4] Accuracy test (random weight, hidden=512) ...", end=" ")
    rows, hidden = 16, 512
    torch.manual_seed(7)
    x      = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.3
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.float32).abs() + 0.5

    cuda_out = ds_kernels.rms_norm(x, weight, 1e-6)
    ref_out  = ref_rms_norm(x, weight, eps=1e-6)

    max_diff = (cuda_out.float() - ref_out.float()).abs().max().item()
    assert max_diff < ATOL, f"Max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_3d_input():
    """Kernel should handle [bsz, seq, hidden] input by flattening."""
    print("  [5] 3-D input test (bsz=2, seq=32, hidden=512) ...", end=" ")
    bsz, seq, hidden = 2, 32, 512
    torch.manual_seed(3)
    x      = torch.randn(bsz, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(hidden, device=DEVICE, dtype=torch.float32)

    cuda_out = ds_kernels.rms_norm(x, weight, 1e-6)
    assert cuda_out.shape == (bsz, seq, hidden), \
        f"Shape mismatch: {cuda_out.shape}"

    # Compare against reference applied on flattened view.
    x_2d   = x.view(-1, hidden)
    ref_2d = ref_rms_norm(x_2d, weight, eps=1e-6).view(bsz, seq, hidden)
    max_diff = (cuda_out.float() - ref_2d.float()).abs().max().item()
    assert max_diff < ATOL, f"Max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_stability():
    """No NaN or Inf in output for typical BF16 inputs."""
    print("  [6] Stability test (no NaN/Inf) ...", end=" ")
    x, w = make_inputs(rows=64, hidden=512, seed=99)
    out  = ds_kernels.rms_norm(x, w, 1e-6)
    assert not torch.isnan(out).any(), "Found NaN in output"
    assert not torch.isinf(out).any(), "Found Inf in output"
    print("PASS")


def test_against_reference_harness():
    """
    Cross-check against the intermediate tensors saved by the reference harness.
    Skipped if the harness output file is not present.
    """
    harness_path = os.path.join(ROOT, "benchmarks", "results", "mla_reference_outputs.pt")
    if not os.path.exists(harness_path):
        print("  [7] Harness cross-check ... SKIPPED (run test_mla_reference_harness.py first)")
        return

    print("  [7] Harness cross-check ...", end=" ")
    all_data = torch.load(harness_path)

    for config_key, tensors in all_data.items():
        compressed_kv = tensors["compressed_kv"].to(DEVICE, torch.bfloat16)
        normed_ref    = tensors["normed_kv"].to(DEVICE, torch.bfloat16)

        # We need the layernorm weight — use ones (harness uses default weights).
        hidden = compressed_kv.shape[-1]  # 512
        weight = torch.ones(hidden, device=DEVICE, dtype=torch.float32)

        x_2d = compressed_kv.view(-1, hidden)
        cuda_out = ds_kernels.rms_norm(x_2d, weight, 1e-6).view_as(compressed_kv)

        max_diff = (cuda_out.float() - normed_ref.float()).abs().max().item()
        assert max_diff < ATOL, \
            f"Harness mismatch for config {config_key}: max_diff={max_diff:.4e}"
        print(f"PASS  config={config_key}  (max_delta={max_diff:.2e})", end="  ")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RMSNorm CUDA Kernel Tests")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    test_shape()
    test_dtype()
    test_accuracy_identity_weight()
    test_accuracy_random_weight()
    test_3d_input()
    test_stability()
    test_against_reference_harness()

    print("=" * 60)
    print("  All RMSNorm tests PASSED")
    print("=" * 60)

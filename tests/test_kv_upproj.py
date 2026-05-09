"""
tests/test_kv_upproj.py
========================
Unit test for the CUDA KV up-projection kernel (ds_kernels.kv_upproj).

Tests:
  1. Shape test     — output shape is [..., 4096].
  2. Dtype test     — output is bfloat16.
  3. Accuracy test  — matches F.linear(normed_kv, weight) within atol=5e-2.
  4. 3-D input      — [bsz, seq, 512] input is handled correctly.
  5. Stability test — no NaN or Inf in output.
  6. Harness check  — validates against saved reference tensors (if available).

Run:
    cd /workspace
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/test_kv_upproj.py
"""

import sys
import os
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import ds_kernels

DEVICE = "cuda"
ATOL   = 5e-2

# Architecture constants matching DeepSeek-V2-Lite config
KV_LORA_RANK = 512
OUT_DIM      = 4096   # num_heads * (qk_nope_head_dim + v_head_dim) = 16 * 256


def make_inputs(rows, seed=0):
    torch.manual_seed(seed)
    normed_kv = torch.randn(rows, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16) * 0.1
    weight    = torch.randn(OUT_DIM, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16) * 0.02
    return normed_kv, weight


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_shape():
    print("  [1] Shape test ...", end=" ")
    x, w = make_inputs(rows=8)
    out  = ds_kernels.kv_upproj(x, w)
    assert out.shape == (8, OUT_DIM), f"Expected (8, {OUT_DIM}), got {out.shape}"
    print("PASS")


def test_dtype():
    print("  [2] Dtype test ...", end=" ")
    x, w = make_inputs(rows=4)
    out  = ds_kernels.kv_upproj(x, w)
    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"
    print("PASS")


def test_accuracy():
    print("  [3] Accuracy test (rows=16) ...", end=" ")
    x, w = make_inputs(rows=16, seed=42)

    cuda_out = ds_kernels.kv_upproj(x, w)
    ref_out  = F.linear(x.float(), w.float()).to(torch.bfloat16)

    max_diff = (cuda_out.float() - ref_out.float()).abs().max().item()
    assert max_diff < ATOL, f"Max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_3d_input():
    print("  [4] 3-D input test (bsz=2, seq=32, kv_lora_rank=512) ...", end=" ")
    bsz, seq = 2, 32
    torch.manual_seed(5)
    x = torch.randn(bsz, seq, KV_LORA_RANK, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w = torch.randn(OUT_DIM, KV_LORA_RANK,  device=DEVICE, dtype=torch.bfloat16) * 0.02

    cuda_out = ds_kernels.kv_upproj(x, w)
    assert cuda_out.shape == (bsz, seq, OUT_DIM), \
        f"Shape mismatch: {cuda_out.shape} vs {(bsz, seq, OUT_DIM)}"

    ref_out = F.linear(x.float(), w.float()).to(torch.bfloat16)
    max_diff = (cuda_out.float() - ref_out.float()).abs().max().item()
    assert max_diff < ATOL, f"Max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_stability():
    print("  [5] Stability test (no NaN/Inf) ...", end=" ")
    x, w = make_inputs(rows=64, seed=99)
    out  = ds_kernels.kv_upproj(x, w)
    assert not torch.isnan(out).any(), "Found NaN in output"
    assert not torch.isinf(out).any(), "Found Inf in output"
    print("PASS")


def test_against_reference_harness():
    """
    Validates the CUDA kernel against the kv_up tensor saved by the reference harness.
    Uses the model's actual kv_b_proj weight — loaded from the harness file.
    """
    harness_path = os.path.join(ROOT, "benchmarks", "results", "mla_reference_outputs.pt")
    if not os.path.exists(harness_path):
        print("  [6] Harness cross-check ... SKIPPED (run test_mla_reference_harness.py first)")
        return

    print("  [6] Harness cross-check ...")
    all_data = torch.load(harness_path)

    # We need the reference model to get kv_b_proj.weight
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from reference.mla import DeepSeekMLA
    torch.manual_seed(42)
    model  = DeepSeekMLA().to(torch.bfloat16).to(DEVICE).eval()
    weight = model.kv_b_proj.weight.detach()  # [4096, 512] BF16

    for config_key, tensors in all_data.items():
        normed_kv = tensors["normed_kv"].to(DEVICE, torch.bfloat16)
        kv_up_ref = tensors["kv_up"].to(DEVICE, torch.bfloat16)

        # Flatten to 2D for the kernel
        rows      = normed_kv.shape[0] * normed_kv.shape[1]
        normed_2d = normed_kv.view(rows, KV_LORA_RANK)

        cuda_out  = ds_kernels.kv_upproj(normed_2d, weight)
        ref_2d    = kv_up_ref.view(rows, OUT_DIM)

        max_diff  = (cuda_out.float() - ref_2d.float()).abs().max().item()
        assert max_diff < ATOL, \
            f"Harness mismatch for config {config_key}: max_diff={max_diff:.4e}"
        print(f"    PASS  config={config_key}  (max_delta={max_diff:.2e})")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  KV Up-Projection CUDA Kernel Tests")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    test_shape()
    test_dtype()
    test_accuracy()
    test_3d_input()
    test_stability()
    test_against_reference_harness()

    print("=" * 60)
    print("  All KV Up-Projection tests PASSED")
    print("=" * 60)

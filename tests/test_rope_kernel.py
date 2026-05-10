"""
tests/test_rope_kernel.py
==========================
Unit test for the CUDA RoPE kernel (ds_kernels.rope).

Tests:
  1. Shape test     — output shapes match inputs.
  2. Dtype test     — outputs are bfloat16.
  3. Accuracy test  — matches reference apply_rotary_pos_emb() within atol=1e-2.
  4. k_pe broadcast — k_pe [bsz,1,seq,64] is handled independently.
  5. Stability test — no NaN or Inf in outputs.
  6. Harness check  — validates against saved reference tensors (if available).

Run:
    cd /workspace
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/test_rope_kernel.py
"""

import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import ds_kernels
from reference.mla import DeepSeekMLA, apply_rotary_pos_emb

DEVICE   = "cuda"
ATOL     = 1e-2

# Architecture constants
NUM_HEADS     = 16
ROPE_DIM      = 64   # qk_rope_head_dim


def build_model_and_cos_sin(seq, bsz=1, seed=42):
    """
    Creates a reference MLA model and returns cos/sin tables indexed by position_ids.
    Also returns q_pe_raw and k_pe_raw as they appear before RoPE.
    """
    torch.manual_seed(seed)
    model = DeepSeekMLA().to(torch.bfloat16).to(DEVICE).eval()
    x = torch.randn(bsz, seq, 2048, device=DEVICE, dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        # Q projection → split into q_nope, q_pe
        q = model.q_proj(x).view(bsz, seq, NUM_HEADS, 192).transpose(1, 2)
        q_pe = q[:, :, :, 128:].contiguous()   # [bsz, 16, seq, 64]

        # KV compression → k_pe
        kv_raw   = model.kv_a_proj_with_mqa(x)         # [bsz, seq, 576]
        k_pe_raw = kv_raw[:, :, 512:]                   # [bsz, seq, 64]
        k_pe     = k_pe_raw.view(bsz, seq, 1, ROPE_DIM).transpose(1, 2).contiguous()
        # k_pe: [bsz, 1, seq, 64]

        # Get cos/sin tables indexed by position_ids
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0).expand(bsz, -1)
        cos_raw, sin_raw = model.rotary_emb(q_pe, seq_len=seq)
        # cos_raw, sin_raw: [seq, 64]

    return model, q_pe, k_pe, cos_raw, sin_raw, position_ids


def prepare_cos_sin(cos_raw, sin_raw, position_ids, bsz, seq, dtype=torch.bfloat16):
    """
    Indexes cos/sin by position_ids and expands to [bsz, 1, seq, rope_dim].
    Matches the reference apply_rotary_pos_emb indexing exactly.
    """
    cos = cos_raw[position_ids].unsqueeze(1).to(dtype)   # [bsz, 1, seq, 64]
    sin = sin_raw[position_ids].unsqueeze(1).to(dtype)   # [bsz, 1, seq, 64]
    return cos, sin


def ref_rope(q_pe, k_pe, cos_raw, sin_raw, position_ids):
    """Runs the reference apply_rotary_pos_emb on float32 tensors."""
    cos_f = cos_raw.float()
    sin_f = sin_raw.float()
    q_f   = q_pe.float()
    k_f   = k_pe.float()
    q_rot_f, k_rot_f = apply_rotary_pos_emb(q_f, k_f, cos_f, sin_f, position_ids)
    return q_rot_f.to(torch.bfloat16), k_rot_f.to(torch.bfloat16)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_shape():
    print("  [1] Shape test ...", end=" ")
    seq = 16
    _, q_pe, k_pe, cos_raw, sin_raw, pos_ids = build_model_and_cos_sin(seq)
    cos, sin = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz=1, seq=seq)

    q_rot, k_rot = ds_kernels.rope(q_pe, k_pe, cos, sin)

    assert q_rot.shape == q_pe.shape, f"q_rot shape {q_rot.shape} != {q_pe.shape}"
    assert k_rot.shape == k_pe.shape, f"k_rot shape {k_rot.shape} != {k_pe.shape}"
    print("PASS")


def test_dtype():
    print("  [2] Dtype test ...", end=" ")
    seq = 16
    _, q_pe, k_pe, cos_raw, sin_raw, pos_ids = build_model_and_cos_sin(seq)
    cos, sin = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz=1, seq=seq)

    q_rot, k_rot = ds_kernels.rope(q_pe, k_pe, cos, sin)
    assert q_rot.dtype == torch.bfloat16, f"q_rot dtype {q_rot.dtype}"
    assert k_rot.dtype == torch.bfloat16, f"k_rot dtype {k_rot.dtype}"
    print("PASS")


def test_accuracy_q_pe():
    print("  [3] Accuracy test — q_pe (bsz=1, seq=16) ...", end=" ")
    seq = 16
    _, q_pe, k_pe, cos_raw, sin_raw, pos_ids = build_model_and_cos_sin(seq, seed=7)
    cos, sin = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz=1, seq=seq)

    q_rot_cuda, _ = ds_kernels.rope(q_pe, k_pe, cos, sin)
    q_rot_ref, _  = ref_rope(q_pe, k_pe, cos_raw, sin_raw, pos_ids)

    max_diff = (q_rot_cuda.float() - q_rot_ref.float()).abs().max().item()
    assert max_diff < ATOL, f"q_pe max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_accuracy_k_pe():
    print("  [4] Accuracy test — k_pe (bsz=1, seq=16) ...", end=" ")
    seq = 16
    _, q_pe, k_pe, cos_raw, sin_raw, pos_ids = build_model_and_cos_sin(seq, seed=7)
    cos, sin = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz=1, seq=seq)

    _, k_rot_cuda = ds_kernels.rope(q_pe, k_pe, cos, sin)
    _, k_rot_ref  = ref_rope(q_pe, k_pe, cos_raw, sin_raw, pos_ids)

    max_diff = (k_rot_cuda.float() - k_rot_ref.float()).abs().max().item()
    assert max_diff < ATOL, f"k_pe max diff {max_diff:.4e} exceeds atol={ATOL}"
    print(f"PASS  (max_delta={max_diff:.2e})")


def test_longer_sequence():
    print("  [5] Accuracy test — longer sequence (bsz=2, seq=64) ...", end=" ")
    bsz, seq = 2, 64
    _, q_pe, k_pe, cos_raw, sin_raw, pos_ids = build_model_and_cos_sin(seq, bsz=bsz, seed=99)
    cos, sin = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz=bsz, seq=seq)

    q_rot_cuda, k_rot_cuda = ds_kernels.rope(q_pe, k_pe, cos, sin)
    q_rot_ref,  k_rot_ref  = ref_rope(q_pe, k_pe, cos_raw, sin_raw, pos_ids)

    q_diff = (q_rot_cuda.float() - q_rot_ref.float()).abs().max().item()
    k_diff = (k_rot_cuda.float() - k_rot_ref.float()).abs().max().item()
    assert q_diff < ATOL, f"q max diff {q_diff:.4e}"
    assert k_diff < ATOL, f"k max diff {k_diff:.4e}"
    print(f"PASS  (q_max={q_diff:.2e}, k_max={k_diff:.2e})")


def test_stability():
    print("  [6] Stability test (no NaN/Inf) ...", end=" ")
    seq = 32
    _, q_pe, k_pe, cos_raw, sin_raw, pos_ids = build_model_and_cos_sin(seq, seed=0)
    cos, sin = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz=1, seq=seq)

    q_rot, k_rot = ds_kernels.rope(q_pe, k_pe, cos, sin)
    assert not torch.isnan(q_rot).any() and not torch.isnan(k_rot).any(), "NaN in output"
    assert not torch.isinf(q_rot).any() and not torch.isinf(k_rot).any(), "Inf in output"
    print("PASS")


def test_against_reference_harness():
    """
    Validates the CUDA kernel against q_pe_rotated and k_pe_rotated
    saved by the reference harness.
    """
    harness_path = os.path.join(ROOT, "benchmarks", "results", "mla_reference_outputs.pt")
    if not os.path.exists(harness_path):
        print("  [7] Harness cross-check ... SKIPPED (run test_mla_reference_harness.py first)")
        return

    print("  [7] Harness cross-check ...")
    all_data = torch.load(harness_path)

    torch.manual_seed(42)
    model = DeepSeekMLA().to(torch.bfloat16).to(DEVICE).eval()

    for config_key, tensors in all_data.items():
        bsz_seq = config_key.strip("()").split(", ")
        bsz, seq = int(bsz_seq[0]), int(bsz_seq[1])

        q_pe_raw   = tensors["q_pe_raw"].to(DEVICE)       # [bsz, 16, seq, 64]
        k_pe_raw   = tensors["k_pe_raw"].to(DEVICE)       # [bsz, 1,  seq, 64]
        cos_raw    = tensors["cos"].to(DEVICE)             # [seq, 64]
        sin_raw    = tensors["sin"].to(DEVICE)             # [seq, 64]
        q_rot_ref  = tensors["q_pe_rotated"].to(DEVICE)   # [bsz, 16, seq, 64]
        k_rot_ref  = tensors["k_pe_rotated"].to(DEVICE)   # [bsz, 1,  seq, 64]
        pos_ids    = tensors["position_ids"].to(DEVICE)   # [bsz, seq]

        cos, sin   = prepare_cos_sin(cos_raw, sin_raw, pos_ids, bsz, seq)

        q_rot_cuda, k_rot_cuda = ds_kernels.rope(q_pe_raw, k_pe_raw, cos, sin)

        q_diff = (q_rot_cuda.float() - q_rot_ref.float()).abs().max().item()
        k_diff = (k_rot_cuda.float() - k_rot_ref.float()).abs().max().item()
        assert q_diff < ATOL, f"q mismatch config {config_key}: {q_diff:.4e}"
        assert k_diff < ATOL, f"k mismatch config {config_key}: {k_diff:.4e}"
        print(f"    PASS  config={config_key}  (q={q_diff:.2e}, k={k_diff:.2e})")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  YaRN RoPE CUDA Kernel Tests")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    test_shape()
    test_dtype()
    test_accuracy_q_pe()
    test_accuracy_k_pe()
    test_longer_sequence()
    test_stability()
    test_against_reference_harness()

    print("=" * 60)
    print("  All RoPE tests PASSED")
    print("=" * 60)

"""
tests/test_mla_reference_harness.py
====================================
Reference Test Harness for DeepSeekMLA.

Purpose:
    Run the reference MLA forward pass step-by-step, capturing every
    intermediate tensor. Save all tensors to disk as a .pt file.
    These values are the ground truth that all Phase 4 CUDA kernels
    are validated against.

Captured intermediates per config:
    q_proj_out        [bsz, 16, seq, 192]   raw Q before split
    q_nope            [bsz, 16, seq, 128]   non-positional Q
    q_pe              [bsz, 16, seq,  64]   RoPE part of Q
    compressed_kv     [bsz, seq,     512]   raw KV latent (pre-norm)
    k_pe_raw          [bsz,  1, seq,  64]   shared RoPE key (pre-RoPE)
    normed_kv         [bsz, seq,     512]   kv after kv_a_layernorm
    kv_up             [bsz, seq,    4096]   kv_b_proj output (pre-split)
    k_nope            [bsz, 16, seq, 128]   non-positional K
    value             [bsz, 16, seq, 128]   V
    cos               [seq,  64]            YaRN cos table
    sin               [seq,  64]            YaRN sin table
    q_pe_rotated      [bsz, 16, seq,  64]   Q after RoPE
    k_pe_rotated      [bsz,  1, seq,  64]   K_pe after RoPE
    query_states      [bsz, 16, seq, 192]   full Q = cat(q_nope, q_pe_rot)
    key_states        [bsz, 16, seq, 192]   full K = cat(k_nope, k_pe_rot)
    attn_weights      [bsz, 16, seq, seq]   pre-softmax (causal masked)
    attn_probs        [bsz, 16, seq, seq]   post-softmax
    attn_output_raw   [bsz, 16, seq, 128]   @ V result
    final_output      [bsz, seq,    2048]   after o_proj

Output:
    benchmarks/results/mla_reference_outputs.pt

    A dict with keys like "(1, 16)" -> dict of the above tensors.

Run:
    cd /workspace
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/test_mla_reference_harness.py
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from reference.mla import (
    DeepSeekMLA,
    apply_rotary_pos_emb,
)

# ── Architecture constants ─────────────────────────────────────────────────────
HIDDEN_SIZE        = 2048
NUM_HEADS          = 16
QK_NOPE_HEAD_DIM   = 128
QK_ROPE_HEAD_DIM   = 64
Q_HEAD_DIM         = 192   # = 128 + 64
V_HEAD_DIM         = 128
KV_LORA_RANK       = 512
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE              = torch.bfloat16

# ── Test configurations ────────────────────────────────────────────────────────
CONFIGS = [
    (1, 16),    # smallest: 1 batch, 16 tokens
    (1, 64),    # medium sequence
    (2, 32),    # batched
    (1, 256),   # longer sequence
]

OUTPUT_PATH = os.path.join(ROOT, "benchmarks", "results", "mla_reference_outputs.pt")


# ── Step-by-step forward pass ─────────────────────────────────────────────────

def run_step_by_step(model: DeepSeekMLA, hidden_states: torch.Tensor) -> dict:
    """
    Replicates DeepSeekMLA.forward() step by step, capturing every
    intermediate tensor. Uses the model's own weights and parameters.
    """
    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    tensors = {}
    tensors["input"] = hidden_states.detach().clone()

    position_ids = torch.arange(q_len, device=device).unsqueeze(0).expand(bsz, -1)
    tensors["position_ids"] = position_ids.detach().clone()

    # ── Step 1: Q Projection ──────────────────────────────────────────────────
    q = model.q_proj(hidden_states)                              # [bsz, seq, 16*192]
    q = q.view(bsz, q_len, NUM_HEADS, Q_HEAD_DIM).transpose(1, 2)   # [bsz, 16, seq, 192]
    tensors["q_proj_out"] = q.detach().clone()

    q_nope, q_pe = torch.split(q, [QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)
    tensors["q_nope"]   = q_nope.detach().clone()   # [bsz, 16, seq, 128]
    tensors["q_pe_raw"] = q_pe.detach().clone()      # [bsz, 16, seq, 64]

    # ── Step 2: KV Compression ────────────────────────────────────────────────
    kv_compressed = model.kv_a_proj_with_mqa(hidden_states)     # [bsz, seq, 576]
    compressed_kv, k_pe = torch.split(
        kv_compressed, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1
    )
    # compressed_kv: [bsz, seq, 512]
    # k_pe:          [bsz, seq,  64]
    tensors["compressed_kv"] = compressed_kv.detach().clone()

    k_pe = k_pe.view(bsz, q_len, 1, QK_ROPE_HEAD_DIM).transpose(1, 2)
    tensors["k_pe_raw"] = k_pe.detach().clone()     # [bsz, 1, seq, 64]

    # ── Step 3: kv_a_layernorm + kv_b_proj ───────────────────────────────────
    normed_kv = model.kv_a_layernorm(compressed_kv)             # [bsz, seq, 512]
    tensors["normed_kv"] = normed_kv.detach().clone()

    kv_up = model.kv_b_proj(normed_kv)                          # [bsz, seq, 4096]
    tensors["kv_up"] = kv_up.detach().clone()

    kv = kv_up.view(
        bsz, q_len, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM
    ).transpose(1, 2)                                            # [bsz, 16, seq, 256]

    k_nope, value = torch.split(kv, [QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    tensors["k_nope"] = k_nope.detach().clone()    # [bsz, 16, seq, 128]
    tensors["value"]  = value.detach().clone()      # [bsz, 16, seq, 128]

    # ── Step 4: YaRN RoPE ────────────────────────────────────────────────────
    cos, sin = model.rotary_emb(value, seq_len=q_len)
    tensors["cos"] = cos.detach().clone()            # [seq, 64]
    tensors["sin"] = sin.detach().clone()            # [seq, 64]

    q_pe_rot, k_pe_rot = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    tensors["q_pe_rotated"] = q_pe_rot.detach().clone()    # [bsz, 16, seq, 64]
    tensors["k_pe_rotated"] = k_pe_rot.detach().clone()    # [bsz,  1, seq, 64]

    # ── Step 5: Assemble full Q and K ─────────────────────────────────────────
    query_states = k_pe_rot.new_empty(bsz, NUM_HEADS, q_len, Q_HEAD_DIM)
    query_states[:, :, :, :QK_NOPE_HEAD_DIM] = q_nope
    query_states[:, :, :, QK_NOPE_HEAD_DIM:] = q_pe_rot
    tensors["query_states"] = query_states.detach().clone()  # [bsz, 16, seq, 192]

    key_states = k_pe_rot.new_empty(bsz, NUM_HEADS, q_len, Q_HEAD_DIM)
    key_states[:, :, :, :QK_NOPE_HEAD_DIM] = k_nope
    key_states[:, :, :, QK_NOPE_HEAD_DIM:] = k_pe_rot
    tensors["key_states"] = key_states.detach().clone()      # [bsz, 16, seq, 192]

    # ── Step 6: Scaled Dot-Product Attention ──────────────────────────────────
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * model.softmax_scale
    # attn_weights: [bsz, 16, seq, seq]

    # Causal mask
    causal_mask = torch.full(
        (q_len, q_len), float("-inf"), device=device, dtype=query_states.dtype
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
    tensors["attn_weights_masked"] = attn_weights.detach().clone()

    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    tensors["attn_probs"] = attn_probs.detach().clone()       # [bsz, 16, seq, seq]

    attn_out = torch.matmul(attn_probs, value)
    tensors["attn_output_raw"] = attn_out.detach().clone()    # [bsz, 16, seq, 128]

    # ── Step 7: Output Projection ─────────────────────────────────────────────
    attn_out = attn_out.transpose(1, 2).contiguous()
    attn_out = attn_out.reshape(bsz, q_len, NUM_HEADS * V_HEAD_DIM)
    final_output = model.o_proj(attn_out)
    tensors["final_output"] = final_output.detach().clone()   # [bsz, seq, 2048]

    return tensors


# ── Shape verification ─────────────────────────────────────────────────────────

EXPECTED_SHAPES = {
    "q_proj_out":       lambda b, s: (b, NUM_HEADS, s, Q_HEAD_DIM),
    "q_nope":           lambda b, s: (b, NUM_HEADS, s, QK_NOPE_HEAD_DIM),
    "q_pe_raw":         lambda b, s: (b, NUM_HEADS, s, QK_ROPE_HEAD_DIM),
    "compressed_kv":    lambda b, s: (b, s, KV_LORA_RANK),
    "k_pe_raw":         lambda b, s: (b, 1, s, QK_ROPE_HEAD_DIM),
    "normed_kv":        lambda b, s: (b, s, KV_LORA_RANK),
    "kv_up":            lambda b, s: (b, s, NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)),
    "k_nope":           lambda b, s: (b, NUM_HEADS, s, QK_NOPE_HEAD_DIM),
    "value":            lambda b, s: (b, NUM_HEADS, s, V_HEAD_DIM),
    "cos":              lambda b, s: (s, QK_ROPE_HEAD_DIM),
    "sin":              lambda b, s: (s, QK_ROPE_HEAD_DIM),
    "q_pe_rotated":     lambda b, s: (b, NUM_HEADS, s, QK_ROPE_HEAD_DIM),
    "k_pe_rotated":     lambda b, s: (b, 1, s, QK_ROPE_HEAD_DIM),
    "query_states":     lambda b, s: (b, NUM_HEADS, s, Q_HEAD_DIM),
    "key_states":       lambda b, s: (b, NUM_HEADS, s, Q_HEAD_DIM),
    "attn_probs":       lambda b, s: (b, NUM_HEADS, s, s),
    "attn_output_raw":  lambda b, s: (b, NUM_HEADS, s, V_HEAD_DIM),
    "final_output":     lambda b, s: (b, s, HIDDEN_SIZE),
}


def verify_shapes(tensors: dict, bsz: int, seq: int) -> bool:
    all_ok = True
    for name, shape_fn in EXPECTED_SHAPES.items():
        expected = shape_fn(bsz, seq)
        actual   = tuple(tensors[name].shape)
        ok       = actual == expected
        status   = "✓" if ok else "✗ MISMATCH"
        print(f"    {status}  {name:<25} {str(actual):<30} (expected {expected})")
        if not ok:
            all_ok = False
    return all_ok


def verify_numerics(tensors: dict, bsz: int, seq: int):
    """Check for NaN/Inf in all intermediate tensors."""
    all_ok = True
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            continue
        has_nan = torch.isnan(t.float()).any().item()
        has_inf = torch.isinf(t.float()).any().item()
        if has_nan or has_inf:
            print(f"    ✗  {name}: NaN={has_nan}  Inf={has_inf}")
            all_ok = False
    if all_ok:
        print("    All intermediates: clean (no NaN/Inf)")
    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  DeepSeekMLA Reference Test Harness — Phase 4, Task 1")
    print(f"  Device : {DEVICE}")
    print(f"  Dtype  : {DTYPE}")
    print("=" * 70)

    torch.manual_seed(42)
    model = DeepSeekMLA().to(DTYPE).to(DEVICE).eval()
    print(f"\nModel weights loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    all_results = {}
    all_passed  = True

    for bsz, seq in CONFIGS:
        key = f"({bsz}, {seq})"
        print(f"\n{'─' * 70}")
        print(f"  Config: bsz={bsz}, seq={seq}")
        print(f"{'─' * 70}")

        torch.manual_seed(42)
        x = torch.randn(bsz, seq, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE) * 0.1

        with torch.no_grad():
            tensors = run_step_by_step(model, x)

        # 1. Shape check
        print(f"\n  [Shapes]")
        shapes_ok = verify_shapes(tensors, bsz, seq)

        # 2. Numeric check
        print(f"\n  [Numerics]")
        numerics_ok = verify_numerics(tensors, bsz, seq)

        # 3. Sanity check: step-by-step output == model.forward() output
        with torch.no_grad():
            ref_out = model(x)
        max_diff = (tensors["final_output"].float() - ref_out.float()).abs().max().item()
        step_ok  = max_diff < 1e-4
        print(f"\n  [Step-by-step vs model.forward()] max_delta = {max_diff:.6e} → {'✓ PASS' if step_ok else '✗ FAIL'}")

        config_ok = shapes_ok and numerics_ok and step_ok
        all_passed = all_passed and config_ok
        print(f"\n  Config ({bsz}, {seq}): {'PASS ✓' if config_ok else 'FAIL ✗'}")

        all_results[key] = {k: v.cpu() for k, v in tensors.items()}

    # ── Save to disk ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(all_results, OUTPUT_PATH)

    print(f"\n{'=' * 70}")
    if all_passed:
        print("  ALL CONFIGS PASSED")
        print(f"  Ground truth saved to: {OUTPUT_PATH}")
        print(f"  Keys saved: {list(all_results.keys())}")
        print(f"\n  Tensor keys per config:")
        first = list(all_results.values())[0]
        for k, v in first.items():
            print(f"    {k:<25} {tuple(v.shape)}  {v.dtype}")
    else:
        print("  SOME CONFIGS FAILED")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()

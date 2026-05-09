"""
test_fused_moe_e2e.py
=====================
End-to-end numerical verification of the fully fused MoE pipeline.

Compares FusedMoELayer and fused_moe_forward against the reference
DeepSeekMoE.forward() implementation.

Run on Vast.ai GPU instance:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/test_fused_moe_e2e.py
"""

import sys
import os
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build"))
sys.path.insert(0, os.path.join(ROOT, "src", "kernels"))
sys.path.insert(0, os.path.join(ROOT, "src", "reference"))

from moe import DeepSeekMoE                     # reference model
from moe_fused import FusedMoELayer, fused_moe_forward

# ── Architecture constants ────────────────────────────────────────────────────
HIDDEN_SIZE       = 2048
INTERMEDIATE_SIZE = 1408
N_EXPERTS         = 64
TOP_K             = 6
N_SHARED_EXPERTS  = 0   # Disable shared experts for a clean expert-only test.
DEVICE            = "cuda"
DTYPE             = torch.bfloat16
ATOL              = 0.05  # BF16 numerical tolerance


# ── Helper ────────────────────────────────────────────────────────────────────
def make_model_and_input(num_tokens: int, seed: int = 42):
    torch.manual_seed(seed)
    model = DeepSeekMoE(
        hidden_size        = HIDDEN_SIZE,
        moe_intermediate_size = INTERMEDIATE_SIZE,
        n_routed_experts   = N_EXPERTS,
        num_experts_per_tok= TOP_K,
        n_shared_experts   = N_SHARED_EXPERTS,
    ).to(DTYPE).to(DEVICE)
    model.eval()

    x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE) * 0.1
    return model, x


# ── Test 1: FusedMoELayer vs reference ───────────────────────────────────────
def test_fused_layer(num_tokens: int) -> float:
    model, x = make_model_and_input(num_tokens)

    # Reference (Python loop in reference moe.py)
    with torch.no_grad():
        ref_out = model(x)

    # Fused (all CUDA kernels)
    fused = FusedMoELayer(model)
    with torch.no_grad():
        fused_out = fused(x)

    assert fused_out.shape == ref_out.shape, \
        f"Shape mismatch: {fused_out.shape} vs {ref_out.shape}"
    assert fused_out.dtype == DTYPE, \
        f"dtype mismatch: {fused_out.dtype}"

    diff = (fused_out.float() - ref_out.float()).abs()
    return diff.max().item(), diff.mean().item()


# ── Test 2: fused_moe_forward (functional API) vs reference ──────────────────
def test_functional_api(num_tokens: int) -> float:
    model, x = make_model_and_input(num_tokens, seed=99)

    with torch.no_grad():
        ref_out = model(x)

    experts    = list(model.experts)
    gate_weight = model.gate.weight

    with torch.no_grad():
        fused_out = fused_moe_forward(x, gate_weight, experts, top_k=TOP_K)

    diff = (fused_out.float() - ref_out.float()).abs()
    return diff.max().item(), diff.mean().item()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device. Run on the Vast.ai GPU instance.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print("=" * 70)
    print("  End-to-End Fused MoE Test")
    print(f"  GPU              : {gpu}")
    print(f"  num_experts      : {N_EXPERTS}   top_k: {TOP_K}")
    print(f"  hidden / intermediate: {HIDDEN_SIZE} / {INTERMEDIATE_SIZE}")
    print(f"  tolerance (atol) : {ATOL}")
    print("=" * 70)

    test_cases   = [1, 4, 16, 32, 128]
    all_passed   = True

    # ── FusedMoELayer tests ──────────────────────────────────────────────────
    print("\n[FusedMoELayer]")
    print(f"  {'tokens':>8}  {'max_delta':>12}  {'mean_delta':>12}  {'status':>8}")
    print("  " + "-" * 50)

    for tokens in test_cases:
        max_d, mean_d = test_fused_layer(tokens)
        passed       = max_d < ATOL
        all_passed   = all_passed and passed
        status       = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {tokens:>8}  {max_d:>12.6f}  {mean_d:>12.6f}  {status:>8}")

    # ── Functional API tests ─────────────────────────────────────────────────
    print("\n[fused_moe_forward]")
    print(f"  {'tokens':>8}  {'max_delta':>12}  {'mean_delta':>12}  {'status':>8}")
    print("  " + "-" * 50)

    for tokens in test_cases:
        max_d, mean_d = test_functional_api(tokens)
        passed       = max_d < ATOL
        all_passed   = all_passed and passed
        status       = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {tokens:>8}  {max_d:>12.6f}  {mean_d:>12.6f}  {status:>8}")

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED — Full fused pipeline is numerically verified.")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

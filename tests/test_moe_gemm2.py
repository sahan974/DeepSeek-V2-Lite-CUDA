"""
test_moe_gemm2.py
=================
Unit test for the GEMM2 kernel (moe/gemm2.cu).

Verifies that ds_kernels.moe_gemm2(swiglu_out, w2, expert_offsets) matches
the reference F.linear per expert within BF16 tolerance.

"""

import sys
import torch
import torch.nn.functional as F

try:
    import ds_kernels
except ImportError:
    print("ERROR: Could not import ds_kernels. Build first with ./build.sh")
    sys.exit(1)

# ── Architecture constants ────────────────────────────────────────────────────
HIDDEN_SIZE       = 2048
INTERMEDIATE_SIZE = 1408
NUM_EXPERTS       = 64
TOP_K             = 6
DEVICE            = "cuda"
DTYPE             = torch.bfloat16


# ── Reference implementation ─────────────────────────────────────────────────
def reference_gemm2(
    swiglu_out:     torch.Tensor,
    w2:             torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch reference: for each expert e, compute swiglu_slice @ w2[e]^T.
    swiglu_out    : [total_slots, intermediate_size]
    w2            : [num_experts, hidden_size, intermediate_size]
    expert_offsets: [num_experts + 1]
    Returns       : [total_slots, hidden_size]
    """
    total_slots = swiglu_out.shape[0]
    num_experts = w2.shape[0]
    out = torch.zeros(total_slots, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    h_offsets = expert_offsets.cpu().tolist()
    for e in range(num_experts):
        start, end = h_offsets[e], h_offsets[e + 1]
        if end <= start:
            continue
        # F.linear(x, W) = x @ W^T
        out[start:end] = F.linear(swiglu_out[start:end], w2[e])
    return out


# ── Helper: build fake dispatch state ────────────────────────────────────────
def make_state(num_tokens: int, seed: int = 42):
    torch.manual_seed(seed)
    total_slots = num_tokens * TOP_K

    # Fake SwiGLU output in [-1, 1].
    swiglu_out = (torch.rand(total_slots, INTERMEDIATE_SIZE, device=DEVICE) * 2 - 1).to(DTYPE)

    # Random down_proj weights (small scale to avoid overflow).
    w2 = (torch.rand(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=DEVICE) * 0.1).to(DTYPE)

    # Round-robin expert distribution.
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE)
    for s in range(total_slots):
        counts[s % NUM_EXPERTS] += 1

    offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    offsets[1:] = counts.cumsum(dim=0)

    return swiglu_out, w2, offsets


# ── Single test case ─────────────────────────────────────────────────────────
def run_test(num_tokens: int, seed: int = 42):
    swiglu_out, w2, offsets = make_state(num_tokens, seed)

    ref_out  = reference_gemm2(swiglu_out, w2, offsets)
    cuda_out = ds_kernels.moe_gemm2(swiglu_out, w2, offsets)

    assert cuda_out.shape == ref_out.shape, \
        f"Shape mismatch: CUDA={cuda_out.shape} vs Ref={ref_out.shape}"
    assert cuda_out.dtype == DTYPE, \
        f"dtype mismatch: expected {DTYPE}, got {cuda_out.dtype}"

    diff = (cuda_out.float() - ref_out.float()).abs()
    return diff.max().item(), diff.mean().item()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run on the Vast.ai GPU instance.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print("=" * 65)
    print("  GEMM2 Kernel Test  (down_proj)")
    print(f"  GPU              : {gpu}")
    print(f"  intermediate_size: {INTERMEDIATE_SIZE}")
    print(f"  hidden_size      : {HIDDEN_SIZE}")
    print(f"  num_experts      : {NUM_EXPERTS}")
    print("=" * 65)

    ATOL = 5e-2  # BF16 output rounding tolerance

    test_cases = [1, 8, 32, 128]
    all_passed = True

    print(f"\n{'tokens':>8}  {'max_delta':>12}  {'mean_delta':>12}  {'status':>8}")
    print("-" * 50)

    for tokens in test_cases:
        max_delta, mean_delta = run_test(tokens)
        passed = max_delta < ATOL
        all_passed = all_passed and passed
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{tokens:>8}  {max_delta:>12.6f}  {mean_delta:>12.6f}  {status:>8}")

    print("=" * 65)

    # ── Edge case: zero tokens ────────────────────────────────────────────────
    print("\n[Edge case] zero total_slots ... ", end="")
    empty_in  = torch.empty(0, INTERMEDIATE_SIZE, device=DEVICE, dtype=DTYPE)
    dummy_w2  = torch.zeros(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                            device=DEVICE, dtype=DTYPE)
    zero_off  = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    out_empty = ds_kernels.moe_gemm2(empty_in, dummy_w2, zero_off)
    assert out_empty.shape == (0, HIDDEN_SIZE), \
        f"Wrong shape for zero-slot input: {out_empty.shape}"
    print("PASS ✓")

    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
test_moe_gemm1.py
=================
Unit test for the GEMM1 kernel (moe/gemm1.cu).

Verifies that ds_kernels.moe_gemm1(dispatched, packed_w1, expert_offsets)
matches reference F.linear per expert within BF16 tolerance.

Run on Vast.ai GPU instance:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/test_moe_gemm1.py
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
DOUBLE_INTERMED   = 2 * INTERMEDIATE_SIZE   # 2816
NUM_EXPERTS       = 64
TOP_K             = 6
DEVICE            = "cuda"
DTYPE             = torch.bfloat16


# ── Reference implementation ─────────────────────────────────────────────────
def reference_gemm1(
    dispatched: torch.Tensor,
    packed_w1:  torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch reference: for each expert, compute dispatched_slice @ w^T.
    dispatched   : [total_slots, hidden_size]
    packed_w1    : [num_experts, 2*intermediate, hidden_size]
    expert_offsets: [num_experts + 1]
    Returns       : [total_slots, 2*intermediate_size]
    """
    total_slots = dispatched.shape[0]
    num_experts = packed_w1.shape[0]
    out = torch.zeros(total_slots, DOUBLE_INTERMED, device=DEVICE, dtype=DTYPE)
    h_offsets = expert_offsets.cpu().tolist()
    for e in range(num_experts):
        start, end = h_offsets[e], h_offsets[e + 1]
        if end <= start:
            continue
        # F.linear(x, W) computes x @ W^T
        out[start:end] = F.linear(dispatched[start:end], packed_w1[e])
    return out


# ── Helper: build fake dispatch state ────────────────────────────────────────
def make_dispatch_state(num_tokens: int, seed: int = 42):
    """
    Generates random dispatched activations and expert offsets.
    Mimics the output of the dispatch kernel for a given batch.
    """
    torch.manual_seed(seed)
    total_slots = num_tokens * TOP_K

    # Random BF16 activations in [-1, 1].
    dispatched = (torch.rand(total_slots, HIDDEN_SIZE, device=DEVICE) * 2 - 1).to(DTYPE)

    # Random packed weights.
    packed_w1 = (torch.rand(NUM_EXPERTS, DOUBLE_INTERMED, HIDDEN_SIZE, device=DEVICE) * 0.1).to(DTYPE)

    # Distribute slots evenly across experts (round-robin).
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE)
    for s in range(total_slots):
        counts[s % NUM_EXPERTS] += 1

    # Compute exclusive prefix sum for offsets.
    offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    offsets[1:] = counts.cumsum(dim=0)

    return dispatched, packed_w1, offsets


# ── Single test case ─────────────────────────────────────────────────────────
def run_test(num_tokens: int, seed: int = 42) -> tuple:
    dispatched, packed_w1, offsets = make_dispatch_state(num_tokens, seed)

    ref_out  = reference_gemm1(dispatched, packed_w1, offsets)
    cuda_out = ds_kernels.moe_gemm1(dispatched, packed_w1, offsets)

    assert cuda_out.shape == ref_out.shape, \
        f"Shape mismatch: CUDA={cuda_out.shape} vs Ref={ref_out.shape}"
    assert cuda_out.dtype == DTYPE, \
        f"dtype mismatch: expected {DTYPE}, got {cuda_out.dtype}"

    diff = (cuda_out.float() - ref_out.float()).abs()
    max_delta  = diff.max().item()
    mean_delta = diff.mean().item()
    return max_delta, mean_delta


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run on the Vast.ai GPU instance.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    print("=" * 65)
    print("  GEMM1 Kernel Test  (gate_proj + up_proj fused)")
    print(f"  GPU              : {gpu}")
    print(f"  hidden_size      : {HIDDEN_SIZE}")
    print(f"  intermediate_size: {INTERMEDIATE_SIZE}  (double={DOUBLE_INTERMED})")
    print(f"  num_experts      : {NUM_EXPERTS}")
    print("=" * 65)

    # BF16 tolerance: FP32 accumulation in cuBLAS should be tight.
    # Mismatch comes from BF16 output rounding, so atol=5e-2 is conservative.
    ATOL = 5e-2

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
    dispatched_empty = torch.empty(0, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    packed_w1_dummy  = torch.zeros(NUM_EXPERTS, DOUBLE_INTERMED, HIDDEN_SIZE,
                                   device=DEVICE, dtype=DTYPE)
    offsets_zero     = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=DEVICE)
    out_empty = ds_kernels.moe_gemm1(dispatched_empty, packed_w1_dummy, offsets_zero)
    assert out_empty.shape == (0, DOUBLE_INTERMED), \
        f"Wrong shape for zero-slot input: {out_empty.shape}"
    print("PASS ✓")

    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

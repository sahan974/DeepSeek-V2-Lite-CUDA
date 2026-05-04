"""
test_swiglu_kernel.py
=====================
Unit test for the SwiGLU CUDA kernel (moe/swiglu.cu).

Verifies that ds_kernels.swiglu(input) produces results matching the
reference PyTorch implementation within the expected BF16 tolerance.
"""

import sys
import torch
import torch.nn.functional as F

# ── Import the compiled extension ────────────────────────────────────────────
try:
    import ds_kernels
except ImportError:
    print("ERROR: Could not import ds_kernels.")
    print("       Build first with:  ./build.sh")
    print("       Then run with:     export PYTHONPATH=$PYTHONPATH:$(pwd)/build")
    sys.exit(1)

# ── Constants matching DeepSeek-V2-Lite architecture ─────────────────────────
INTERMEDIATE_SIZE = 1408   # Per-expert intermediate dimension.
HIDDEN_SIZE       = 2048   # Token hidden dimension.
DTYPE             = torch.bfloat16
DEVICE            = "cuda"

# ── Reference implementation (pure PyTorch) ───────────────────────────────────
def reference_swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    x: [total_tokens, 2 * intermediate_size]
    Returns: [total_tokens, intermediate_size]
    """
    gate = x[:, :INTERMEDIATE_SIZE]
    up   = x[:, INTERMEDIATE_SIZE:]
    return F.silu(gate) * up


# ── Test runner ───────────────────────────────────────────────────────────────
def run_test(total_tokens: int, seed: int = 42) -> float:
    """
    Runs the CUDA SwiGLU kernel and compares it against the reference.
    Returns the maximum absolute difference.
    """
    torch.manual_seed(seed)

    # Random BF16 input in [-2, 2] — representative of post-GEMM activation range.
    x = (torch.rand(total_tokens, 2 * INTERMEDIATE_SIZE, device=DEVICE) * 4.0 - 2.0).to(DTYPE)

    # Reference
    ref_out = reference_swiglu(x)

    # CUDA kernel
    cuda_out = ds_kernels.swiglu(x)

    # Validate shape
    assert cuda_out.shape == ref_out.shape, (
        f"Shape mismatch: CUDA={cuda_out.shape} vs Ref={ref_out.shape}"
    )
    assert cuda_out.dtype == DTYPE, (
        f"dtype mismatch: expected {DTYPE}, got {cuda_out.dtype}"
    )

    # Compute max absolute difference
    diff = (cuda_out.float() - ref_out.float()).abs()
    max_delta = diff.max().item()
    mean_delta = diff.mean().item()

    return max_delta, mean_delta


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run this test on the Vast.ai GPU instance.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("  SwiGLU Kernel Test")
    print(f"  GPU             : {gpu_name}")
    print(f"  intermediate_size: {INTERMEDIATE_SIZE}")
    print(f"  dtype           : {DTYPE}")
    print("=" * 60)

    # Tolerance: BF16 has ~7 bits of mantissa. For SiLU + multiply, a max delta
    # of 1e-2 in BF16 is well within the expected range.
    ATOL = 1e-2

    test_cases = [1, 8, 32, 128, 256]
    all_passed = True

    print(f"\n{'tokens':>8}  {'max_delta':>12}  {'mean_delta':>12}  {'status':>8}")
    print("-" * 50)

    for tokens in test_cases:
        max_delta, mean_delta = run_test(tokens)
        passed = max_delta < ATOL
        status = "PASS ✓" if passed else "FAIL ✗"
        all_passed = all_passed and passed
        print(f"{tokens:>8}  {max_delta:>12.6f}  {mean_delta:>12.6f}  {status:>8}")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)

    # ── Edge case: zero tokens ────────────────────────────────────────────────
    print("\n[Edge case] zero tokens ... ", end="")
    x_empty = torch.empty(0, 2 * INTERMEDIATE_SIZE, device=DEVICE, dtype=DTYPE)
    out_empty = ds_kernels.swiglu(x_empty)
    assert out_empty.shape == (0, INTERMEDIATE_SIZE), \
        f"Wrong shape for zero-token input: {out_empty.shape}"
    print("PASS ✓")


if __name__ == "__main__":
    main()

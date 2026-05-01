"""
tests/test_moe_scan_kernel.py

Validates the moe_scan CUDA kernel against a NumPy reference implementation
of an exclusive prefix sum.

Unlike the routing kernel, the scan operates purely on integer arithmetic and
does not require a physical GPU for correctness validation. The test executes
on CPU tensors where CUDA is unavailable, or on GPU when CUDA is present.

Test cases:
  1. Correctness            — offsets match numpy.cumsum with exclusive shift.
  2. Sentinel element       — offsets[num_experts] equals the total slot count.
  3. Zero counts            — all-zero input produces all-zero offsets.
  4. Single expert          — degenerate case with one expert.
  5. Uniform distribution   — all experts receive the same token count.
  6. Skewed distribution    — one expert receives all tokens, the rest receive none.
"""

import sys
import os

import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "build"))

# torch must be imported before ds_kernels. Additionally, torch._C must be
# reloaded with RTLD_GLOBAL so that its PyBind11 type_caster symbols (e.g.,
# type_caster<at::Tensor>) are visible to the dynamic linker when ds_kernels.so
# is subsequently loaded. Without RTLD_GLOBAL, Python loads C extensions with
# RTLD_LOCAL by default, making their symbols invisible to other extensions.
import ctypes
import torch
ctypes.CDLL(torch._C.__file__, ctypes.RTLD_GLOBAL)

NUM_EXPERTS = 64
TOP_K       = 6


def reference_exclusive_prefix_sum(counts: np.ndarray) -> np.ndarray:
    """
    Computes the exclusive prefix sum of counts using NumPy.
    The output has len(counts) + 1 elements. The final element is the grand total.
    """
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    return offsets


def run_cuda_scan(counts_np: np.ndarray) -> np.ndarray:
    """
    Wraps the CUDA moe_scan kernel. Accepts a NumPy int32 array and returns
    a NumPy int32 array of length num_experts + 1.
    """
    import ds_kernels
    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts_tensor = torch.from_numpy(counts_np).to(torch.int32).to(device)
    offsets_tensor = ds_kernels.moe_scan(counts_tensor)
    return offsets_tensor.cpu().numpy()


def test_correctness():
    """Validates offset values against the NumPy exclusive prefix sum reference."""
    print("  [correctness]  random expert counts")
    np.random.seed(42)
    counts = np.random.randint(0, 20, size=NUM_EXPERTS).astype(np.int32)

    ref_offsets  = reference_exclusive_prefix_sum(counts)
    cuda_offsets = run_cuda_scan(counts)

    assert np.array_equal(ref_offsets, cuda_offsets), (
        f"Offset mismatch.\nReference: {ref_offsets}\nKernel:    {cuda_offsets}"
    )
    print("    PASSED")


def test_sentinel_element():
    """Validates that offsets[num_experts] equals the total number of dispatched slots."""
    print("  [sentinel]     offsets[64] == total slots")
    num_tokens = 32
    total_slots = num_tokens * TOP_K  # 192

    # Construct counts that sum to total_slots.
    counts = np.zeros(NUM_EXPERTS, dtype=np.int32)
    # Distribute 192 tokens evenly: 3 per expert.
    counts[:] = num_tokens * TOP_K // NUM_EXPERTS  # 192 / 64 = 3

    cuda_offsets = run_cuda_scan(counts)

    assert cuda_offsets[NUM_EXPERTS] == total_slots, (
        f"Sentinel element mismatch: expected {total_slots}, got {cuda_offsets[NUM_EXPERTS]}"
    )
    print(f"    PASSED  (total_slots={cuda_offsets[NUM_EXPERTS]})")


def test_zero_counts():
    """Validates that all-zero input produces all-zero offsets."""
    print("  [zeros]        all expert counts are zero")
    counts = np.zeros(NUM_EXPERTS, dtype=np.int32)

    ref_offsets  = reference_exclusive_prefix_sum(counts)
    cuda_offsets = run_cuda_scan(counts)

    assert np.array_equal(ref_offsets, cuda_offsets), (
        "Zero-count case produced non-zero offsets."
    )
    assert cuda_offsets[NUM_EXPERTS] == 0, (
        f"Sentinel element should be 0 for zero counts, got {cuda_offsets[NUM_EXPERTS]}"
    )
    print("    PASSED")


def test_single_expert():
    """Validates the degenerate case of a single expert."""
    print("  [single]       num_experts = 1")
    counts = np.array([7], dtype=np.int32)

    ref_offsets  = reference_exclusive_prefix_sum(counts)
    cuda_offsets = run_cuda_scan(counts)

    assert np.array_equal(ref_offsets, cuda_offsets), (
        f"Single-expert mismatch.\nReference: {ref_offsets}\nKernel: {cuda_offsets}"
    )
    print("    PASSED")


def test_uniform_distribution():
    """Validates correctness when all experts receive an equal number of tokens."""
    print("  [uniform]      all experts receive equal token counts")
    tokens_per_expert = 5
    counts = np.full(NUM_EXPERTS, tokens_per_expert, dtype=np.int32)

    ref_offsets  = reference_exclusive_prefix_sum(counts)
    cuda_offsets = run_cuda_scan(counts)

    assert np.array_equal(ref_offsets, cuda_offsets), (
        f"Uniform-distribution mismatch.\nReference: {ref_offsets}\nKernel: {cuda_offsets}"
    )
    expected_total = NUM_EXPERTS * tokens_per_expert
    assert cuda_offsets[NUM_EXPERTS] == expected_total, (
        f"Sentinel element mismatch: expected {expected_total}, got {cuda_offsets[NUM_EXPERTS]}"
    )
    print(f"    PASSED  (total_slots={cuda_offsets[NUM_EXPERTS]})")


def test_skewed_distribution():
    """Validates correctness when one expert receives all tokens."""
    print("  [skewed]       single expert receives all tokens")
    total_slots = 192
    counts = np.zeros(NUM_EXPERTS, dtype=np.int32)
    counts[0] = total_slots  # Expert 0 receives all tokens.

    ref_offsets  = reference_exclusive_prefix_sum(counts)
    cuda_offsets = run_cuda_scan(counts)

    assert np.array_equal(ref_offsets, cuda_offsets), (
        f"Skewed-distribution mismatch.\nReference: {ref_offsets}\nKernel: {cuda_offsets}"
    )
    assert cuda_offsets[NUM_EXPERTS] == total_slots, (
        f"Sentinel element mismatch: expected {total_slots}, got {cuda_offsets[NUM_EXPERTS]}"
    )
    print("    PASSED")


def main():
    try:
        import ds_kernels
    except ImportError:
        print("ds_kernels library not found. Ensure the build directory is on PYTHONPATH.")
        print("  export PYTHONPATH=$PYTHONPATH:$(pwd)/build")
        return

    device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"Running moe_scan kernel tests on {device_str}")
    print()

    print("=== Correctness Test ===")
    test_correctness()

    print("\n=== Sentinel Element Test ===")
    test_sentinel_element()

    print("\n=== Zero Counts Test ===")
    test_zero_counts()

    print("\n=== Single Expert Test ===")
    test_single_expert()

    print("\n=== Uniform Distribution Test ===")
    test_uniform_distribution()

    print("\n=== Skewed Distribution Test ===")
    test_skewed_distribution()

    print("\n✅  All moe_scan kernel tests passed.")


if __name__ == "__main__":
    main()

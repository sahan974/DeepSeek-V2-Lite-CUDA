"""
scripts/benchmark_routing.py

Latency comparison benchmark: Reference Python-loop MoE vs CUDA-fused MoE pipeline.

Measures wall-clock latency (ms) for the routing + dispatch + expert compute +
combine step of a single DeepSeekMoE layer at batch sizes 1, 8, and 32.

Two paths are benchmarked:
    (A) REFERENCE: Pure PyTorch — the per-expert Python for-loop in DeepSeekMoE.forward()
    (B) CUDA-FUSED: moe_routing → moe_scan → moe_dispatch → reference MLPs → moe_combine

Both paths use the SAME expert MLP weights to ensure a fair comparison.
The shared expert is excluded from the benchmark to isolate the routing pipeline.

Output format:
    Batch  | Ref (ms) | CUDA (ms) | Speedup
    -------|----------|-----------|--------
    1      | ...      | ...       | ...x
    8      | ...      | ...       | ...x
    32     | ...      | ...       | ...x

Usage (on Vast.ai GPU instance):
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 scripts/benchmark_routing.py
"""

import sys
import os
import ctypes
import torch
import torch.nn.functional as F

# Ensure PyBind11 type_caster symbols are visible before importing ds_kernels.
ctypes.CDLL(torch._C.__file__, ctypes.RTLD_GLOBAL)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "build"))

from reference.moe import DeepSeekMoE
from kernels.moe_fused import fused_moe_forward

# -----------------------------------------------------------------------
# Configuration — matches DeepSeek-V2-Lite architecture.
# -----------------------------------------------------------------------
HIDDEN_SIZE          = 2048
MOE_INTERMEDIATE     = 1408
N_ROUTED_EXPERTS     = 64
TOP_K                = 6
WARMUP_ITERS         = 20
TIMED_ITERS          = 100
BATCH_SIZES          = [1, 8, 32]


def cuda_time_ms(fn, warmup: int, timed: int) -> float:
    """
    Measures average GPU execution time of fn() in milliseconds.
    Uses CUDA events for accurate GPU-side timing.
    """
    # Warmup — ensures kernels are compiled and caches are warm.
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(timed):
        with torch.no_grad():
            fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / timed


def reference_forward(moe: DeepSeekMoE, x: torch.Tensor) -> torch.Tensor:
    """
    Pure-reference forward: uses the per-expert Python for-loop.
    Shared expert is excluded to isolate routing pipeline cost.
    """
    orig_shared = moe.shared_experts
    moe.shared_experts = None          # Temporarily disable shared expert.
    out = moe(x)
    moe.shared_experts = orig_shared
    return out


def fused_forward(moe: DeepSeekMoE, x: torch.Tensor) -> torch.Tensor:
    """
    CUDA-fused forward: routing → scan → dispatch → MLPs → combine.
    Shared expert is excluded to match the reference benchmark scope.
    """
    # Flatten [batch, seq, hidden] → [num_tokens, hidden].
    flat = x.view(-1, x.shape[-1])
    return fused_moe_forward(
        flat,
        moe.gate.weight,
        list(moe.experts),
        top_k=TOP_K,
    )


def verify_correctness(moe: DeepSeekMoE, x: torch.Tensor) -> float:
    """
    Runs both paths and returns the max absolute delta.
    Used to confirm the CUDA path is numerically correct.
    """
    with torch.no_grad():
        ref  = reference_forward(moe, x)
        flat = x.view(-1, x.shape[-1])
        cuda = fused_moe_forward(flat, moe.gate.weight, list(moe.experts), TOP_K)

    # Align shapes: ref has the full MoE output (without shared expert).
    ref_flat = ref.view(-1, HIDDEN_SIZE)
    return (cuda.float() - ref_flat.float()).abs().max().item()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Run this on the Vast.ai GPU instance.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu    = torch.cuda.get_device_name(0)

    print("=" * 72)
    print("DeepSeek-V2-Lite  |  MoE Routing Benchmark")
    print(f"GPU     : {gpu}")
    print(f"Warmup  : {WARMUP_ITERS} iterations   Timed: {TIMED_ITERS} iterations")
    print(f"Config  : hidden={HIDDEN_SIZE}, experts={N_ROUTED_EXPERTS}, top_k={TOP_K}")
    print("=" * 72)

    # Build a single MoE layer and move to GPU in BF16.
    moe = DeepSeekMoE(
        hidden_size=HIDDEN_SIZE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=N_ROUTED_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=0,           # Exclude shared expert from benchmark.
    ).to(device).to(torch.bfloat16).eval()

    # -----------------------------------------------------------------------
    # Correctness check at batch=32 before benchmarking.
    # -----------------------------------------------------------------------
    print("\n--- Correctness Verification (batch=32, seq=1) ---")
    x_check = torch.randn(32, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    max_delta = verify_correctness(moe, x_check)
    status = "✅  PASS" if max_delta < 5e-2 else "❌  FAIL"
    print(f"  Max absolute delta (CUDA vs Reference): {max_delta:.6f}  →  {status}")
    if max_delta >= 5e-2:
        print("  WARNING: Numerical mismatch is large. Check kernel correctness.")

    # -----------------------------------------------------------------------
    # Latency benchmark.
    # -----------------------------------------------------------------------
    print()
    print(f"{'Batch':>6}  {'Ref (ms)':>10}  {'CUDA (ms)':>10}  {'Speedup':>9}  {'Tokens/s (CUDA)':>16}")
    print("-" * 72)

    for batch in BATCH_SIZES:
        # Shape: [batch, 1, hidden] — single token per sequence (decode step).
        x = torch.randn(batch, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

        ref_ms  = cuda_time_ms(lambda: reference_forward(moe, x), WARMUP_ITERS, TIMED_ITERS)
        cuda_ms = cuda_time_ms(lambda: fused_forward(moe, x),     WARMUP_ITERS, TIMED_ITERS)

        speedup      = ref_ms / cuda_ms
        tokens_per_s = batch / (cuda_ms / 1000.0)

        print(f"{batch:>6}  {ref_ms:>10.3f}  {cuda_ms:>10.3f}  {speedup:>8.2f}x  {tokens_per_s:>16,.0f}")

    print("=" * 72)
    print("Benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()

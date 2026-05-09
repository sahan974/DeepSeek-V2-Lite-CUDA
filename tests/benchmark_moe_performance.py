"""
benchmark_moe_performance.py
============================
Measures and compares latency and throughput across three MoE implementations:

  Tier 0 — Baseline : Pure PyTorch (Python expert loop, no CUDA kernels)
  Tier 1 — Phase 2  : Fused routing/dispatch/combine kernels + Python expert loop
  Tier 2 — Phase 3  : Fully fused CUDA pipeline (GEMM1 → SwiGLU → GEMM2)

Benchmark methodology:
  - 20 warm-up iterations (GPU cache / JIT stabilization)
  - 200 timed iterations measured with CUDA events (zero Python overhead)
  - Reports: mean latency (ms), p95 latency (ms), throughput (tokens/sec),
             and speedup vs Tier 0

Run on Vast.ai GPU instance:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build
    python3 tests/benchmark_moe_performance.py
"""

import sys
import os
import time

import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build"))
sys.path.insert(0, os.path.join(ROOT, "src", "kernels"))
sys.path.insert(0, os.path.join(ROOT, "src", "reference"))

from moe import DeepSeekMoE
from moe_fused import FusedMoELayer, fused_moe_forward
from expert_prepack import prepack_experts

import ds_kernels

# ── Architecture constants ────────────────────────────────────────────────────
HIDDEN_SIZE       = 2048
INTERMEDIATE_SIZE = 1408
N_EXPERTS         = 64
TOP_K             = 6
N_SHARED_EXPERTS  = 0   # Disable shared experts for a focused MoE benchmark
DEVICE            = "cuda"
DTYPE             = torch.bfloat16

WARMUP_ITERS = 20
BENCH_ITERS  = 200


# ─────────────────────────────────────────────────────────────────────────────
# Tier 0: Pure PyTorch baseline (reference model)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(model: DeepSeekMoE, x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch: reference DeepSeekMoE.forward()."""
    return model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Phase 2 — Fused routing/dispatch/combine, Python expert loop
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(
    model: DeepSeekMoE,
    x: torch.Tensor,
    experts: list,
) -> torch.Tensor:
    """
    Uses Phase 2 CUDA kernels for routing, dispatch, and combine.
    Expert MLP computation still uses the Python loop over nn.Modules.
    """
    num_tokens, hidden_size = x.shape

    logits = F.linear(x.float(), model.gate.weight.float(), None)
    topk_indices, topk_weights, expert_counts = ds_kernels.moe_routing(logits, TOP_K)
    expert_offsets = ds_kernels.moe_scan(expert_counts)
    dispatched, token_map, dispatched_weights = ds_kernels.moe_dispatch(
        x, topk_indices, topk_weights, expert_offsets
    )

    total_slots = num_tokens * TOP_K
    expert_out  = torch.zeros(total_slots, hidden_size, device=DEVICE, dtype=DTYPE)
    h_offsets   = expert_offsets.cpu().tolist()

    for e, expert in enumerate(experts):
        start, end = h_offsets[e], h_offsets[e + 1]
        if end <= start:
            continue
        with torch.no_grad():
            expert_out[start:end] = expert(dispatched[start:end])

    return ds_kernels.moe_combine(expert_out, token_map, dispatched_weights, num_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Phase 3 — Fully fused CUDA pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3(
    fused_layer: FusedMoELayer,
    x: torch.Tensor,
) -> torch.Tensor:
    """FusedMoELayer: zero Python loops, all CUDA."""
    return fused_layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA event benchmark harness
# ─────────────────────────────────────────────────────────────────────────────

def cuda_benchmark(fn, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS):
    """
    Returns (mean_ms, p95_ms) measured with CUDA events.
    All Python overhead is excluded — only GPU kernel time is measured.
    """
    # Warm-up: populates caches, JIT compiles any lazy ops.
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

    # Timed iterations.
    latencies_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start_event.record()
        with torch.no_grad():
            fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms  = latencies_ms[int(0.95 * len(latencies_ms))]
    return mean_ms, p95_ms


# ─────────────────────────────────────────────────────────────────────────────
# Single benchmark run for one token batch size
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(num_tokens: int, model: DeepSeekMoE, fused_layer: FusedMoELayer):
    experts = list(model.experts)
    x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE) * 0.1

    mean0, p95_0 = cuda_benchmark(lambda: run_baseline(model, x))
    mean1, p95_1 = cuda_benchmark(lambda: run_phase2(model, x, experts))
    mean2, p95_2 = cuda_benchmark(lambda: run_phase3(fused_layer, x))

    throughput0 = (num_tokens / mean0) * 1000  # tokens/sec
    throughput1 = (num_tokens / mean1) * 1000
    throughput2 = (num_tokens / mean2) * 1000

    speedup1 = mean0 / mean1
    speedup2 = mean0 / mean2

    return {
        "tokens":      num_tokens,
        "base_mean":   mean0,   "base_p95":   p95_0,   "base_tps":   throughput0,
        "p2_mean":     mean1,   "p2_p95":     p95_1,   "p2_tps":     throughput1,
        "p3_mean":     mean2,   "p3_p95":     p95_2,   "p3_tps":     throughput2,
        "speedup1":    speedup1,
        "speedup2":    speedup2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device. Run on the Vast.ai GPU instance.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("=" * 75)
    print("  DeepSeek-V2-Lite MoE Kernel Benchmark — Phase 2 vs Phase 3")
    print(f"  GPU        : {gpu}  ({vram_gb:.1f} GB VRAM)")
    print(f"  Experts    : {N_EXPERTS}  top-k: {TOP_K}")
    print(f"  Hidden/Int : {HIDDEN_SIZE} / {INTERMEDIATE_SIZE}")
    print(f"  Warmup     : {WARMUP_ITERS} iters   Bench: {BENCH_ITERS} iters")
    print("=" * 75)

    torch.manual_seed(42)
    model = DeepSeekMoE(
        hidden_size          = HIDDEN_SIZE,
        moe_intermediate_size= INTERMEDIATE_SIZE,
        n_routed_experts     = N_EXPERTS,
        num_experts_per_tok  = TOP_K,
        n_shared_experts     = N_SHARED_EXPERTS,
    ).to(DTYPE).to(DEVICE).eval()

    print("\nPre-packing expert weights for Phase 3...")
    fused_layer = FusedMoELayer(model)

    token_sizes = [1, 4, 16, 32, 64, 128, 256]
    results = []

    print("\nRunning benchmarks...\n")

    for num_tokens in token_sizes:
        print(f"  tokens={num_tokens:<5}", end="", flush=True)
        r = benchmark_one(num_tokens, model, fused_layer)
        results.append(r)
        print(f"  base={r['base_mean']:.2f}ms  p2={r['p2_mean']:.2f}ms  "
              f"p3={r['p3_mean']:.2f}ms  "
              f"speedup(p3/base)={r['speedup2']:.2f}x")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"  {'Tokens':>7} │ {'Baseline':>13} │ {'Phase 2':>13} │ {'Phase 3':>13} │ {'Speedup':>9}")
    print(f"  {'':>7} │ {'mean (ms)':>13} │ {'mean (ms)':>13} │ {'mean (ms)':>13} │ {'P3/Base':>9}")
    print("  " + "─" * 67)

    for r in results:
        print(
            f"  {r['tokens']:>7} │ "
            f"{r['base_mean']:>12.3f} │ "
            f"{r['p2_mean']:>12.3f} │ "
            f"{r['p3_mean']:>12.3f} │ "
            f"{r['speedup2']:>8.2f}x"
        )

    print("=" * 75)

    # ── Throughput summary ────────────────────────────────────────────────────
    print(f"\n  {'Tokens':>7} │ {'Baseline tok/s':>16} │ {'Phase 2 tok/s':>16} │ {'Phase 3 tok/s':>16}")
    print("  " + "─" * 63)

    for r in results:
        print(
            f"  {r['tokens']:>7} │ "
            f"{r['base_tps']:>15,.0f} │ "
            f"{r['p2_tps']:>15,.0f} │ "
            f"{r['p3_tps']:>15,.0f}"
        )

    print("=" * 75)

    # ── Peak speedup ──────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["speedup2"])
    print(f"\n  Peak speedup (Phase 3 vs Baseline): {best['speedup2']:.2f}x "
          f"at {best['tokens']} tokens")
    print(f"  Peak speedup (Phase 2 vs Baseline): "
          f"{max(r['speedup1'] for r in results):.2f}x")
    print()


if __name__ == "__main__":
    main()

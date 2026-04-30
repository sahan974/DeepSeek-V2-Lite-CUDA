"""
Baseline latency benchmark for the DeepSeek-V2-Lite reference decoder layer.

Measures wall-clock time (ms) for a single MoE decoder layer forward pass
under the following conditions:
    - Batch sizes: 1 and 8
    - Sequence lengths: 128, 512, 1024
    - Device: CUDA if available, otherwise CPU
    - Precision: bfloat16 on CUDA, float32 on CPU
    - 20 warmup iterations, 100 timed iterations (averaged)

These numbers establish the PyTorch eager-mode baseline that custom CUDA
kernels will be compared against. Results are printed to stdout in a
structured table format.

Usage:
    python scripts/benchmark_baseline.py
"""

import sys
import os
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from reference.decoder_layer import DeepSeekDecoderLayer

WARMUP_ITERS   = 20
TIMED_ITERS    = 100
BATCH_SIZES    = [1, 8]
SEQ_LENGTHS    = [128, 512, 1024]
HIDDEN_SIZE    = 2048


def benchmark_one(layer, x, device: str) -> float:
    """
    Returns average forward-pass latency in milliseconds over TIMED_ITERS runs.
    CUDA events are used for GPU timing; time.perf_counter for CPU.
    """
    if device == "cuda":
        # Warmup
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                _ = layer(x)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(TIMED_ITERS):
            with torch.no_grad():
                _ = layer(x)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        return elapsed_ms / TIMED_ITERS

    else:
        # CPU path — use perf_counter
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                _ = layer(x)

        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            with torch.no_grad():
                _ = layer(x)
        end = time.perf_counter()

        return ((end - start) / TIMED_ITERS) * 1000.0  # convert to ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("DeepSeek-V2-Lite  |  Decoder Layer Baseline Benchmark")
    print(f"Device  : {device.upper()}")
    if device == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Dtype   : {dtype}")
    print(f"Warmup  : {WARMUP_ITERS} iterations")
    print(f"Timed   : {TIMED_ITERS} iterations (averaged)")
    print("=" * 70)
    print(f"{'Batch':>6}  {'Seq':>6}  {'Latency (ms)':>14}  {'Tokens/sec':>14}")
    print("-" * 70)

    # A single layer instance is reused across all configurations
    layer = DeepSeekDecoderLayer().to(device).to(dtype)
    layer.eval()

    for batch_size in BATCH_SIZES:
        for seq_len in SEQ_LENGTHS:
            x = torch.randn(batch_size, seq_len, HIDDEN_SIZE, device=device, dtype=dtype)

            latency_ms = benchmark_one(layer, x, device)
            tokens_per_sec = (batch_size * seq_len) / (latency_ms / 1000.0)

            print(f"{batch_size:>6}  {seq_len:>6}  {latency_ms:>14.3f}  {tokens_per_sec:>14,.0f}")

        print()  # blank line between batch sizes

    print("=" * 70)
    print("Benchmark complete.")
    print("Record these numbers before proceeding to Phase 2 (CUDA kernels).")
    print("=" * 70)


if __name__ == "__main__":
    main()

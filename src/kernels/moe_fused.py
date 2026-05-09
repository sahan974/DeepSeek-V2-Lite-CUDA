"""
src/kernels/moe_fused.py
========================
Full fused CUDA MoE forward pass — Phase 3 integrated pipeline.

This module provides two APIs:

  1. FusedMoELayer  (production use)
     Drop-in replacement for DeepSeekMoE that pre-packs expert weights at
     construction time and uses all Phase 2 + Phase 3 CUDA kernels.

  2. fused_moe_forward  (functional, backward-compatible)
     Accepts the original experts list and optional pre-packed weight tensors.
     Pre-packs on first call (lazy) and caches the result for subsequent calls.

Pipeline executed per forward call:
    ┌─ hidden_states [T, H]
    │
    ├─[Step 1] F.linear(float32)       → logits  [T, E]
    │
    ├─[Step 2] ds_kernels.moe_routing  → topk_indices  [T, top_k]
    │                                    topk_weights  [T, top_k]
    │                                    expert_counts [E]
    │
    ├─[Step 3] ds_kernels.moe_scan     → expert_offsets [E+1]
    │
    ├─[Step 4] ds_kernels.moe_dispatch → dispatched         [S, H]
    │                                    token_map          [S]
    │                                    dispatched_weights [S]
    │                     (S = T * top_k total slots)
    │
    ├─[Step 5] ds_kernels.moe_gemm1    → gemm1_out  [S, 2*I]
    │          ds_kernels.swiglu       → swiglu_out [S, I]
    │          ds_kernels.moe_gemm2    → expert_out [S, H]
    │                     (I = intermediate_size = 1408)
    │
    └─[Step 6] ds_kernels.moe_combine  → final_output [T, H]

Terminology:
    T = num_tokens     H = hidden_size (2048)
    E = num_experts    I = intermediate_size (1408)
    S = total_slots = T * top_k

Usage:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build

    # Production (pre-packs once):
    from src.kernels.moe_fused import FusedMoELayer
    fused_moe = FusedMoELayer(model.model.layers[0].mlp)
    output = fused_moe(hidden_states)

    # Functional (lazy pre-pack):
    from src.kernels.moe_fused import fused_moe_forward
    output = fused_moe_forward(hidden_states, gate_weight, experts, top_k=6)
"""

import sys
import os
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Ensure PyBind11 type_caster symbols are visible to the dynamic linker.
ctypes.CDLL(torch._C.__file__, ctypes.RTLD_GLOBAL)

# ds_kernels must be on sys.path (add build/ directory before importing).
try:
    import ds_kernels
except ImportError as exc:
    raise ImportError(
        "ds_kernels not found. Add the build/ directory to PYTHONPATH:\n"
        "  export PYTHONPATH=$PYTHONPATH:$(pwd)/build"
    ) from exc

# expert_prepack.py is a sibling of this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from expert_prepack import prepack_experts


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: Expert MLP
# ─────────────────────────────────────────────────────────────────────────────

def _fused_expert_mlp(
    dispatched:     torch.Tensor,   # BF16 [S, H]
    packed_w1:      torch.Tensor,   # BF16 [E, 2*I, H]
    w2:             torch.Tensor,   # BF16 [E, H, I]
    expert_offsets: torch.Tensor,   # int32 [E+1]
) -> torch.Tensor:                  # BF16 [S, H]
    """
    Executes the three-stage expert MLP using CUDA kernels:
        GEMM1  →  SwiGLU  →  GEMM2

    All three stages are fused at the Python level — no intermediate
    Python-side loops over experts. cuBLAS handles expert-level batching.
    """
    # Gate + Up projection: [S, H] @ [E, 2I, H]^T  →  [S, 2I]
    gemm1_out  = ds_kernels.moe_gemm1(dispatched, packed_w1, expert_offsets)

    # SwiGLU activation: [S, 2I]  →  [S, I]
    swiglu_out = ds_kernels.swiglu(gemm1_out)

    # Down projection:   [S, I] @ [E, H, I]^T       →  [S, H]
    expert_out = ds_kernels.moe_gemm2(swiglu_out, w2, expert_offsets)

    return expert_out


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: Routing + Dispatch + Combine pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _route_dispatch_combine(
    hidden_states: torch.Tensor,   # BF16 [T, H]
    gate_weight:   torch.Tensor,   # FP32 [E, H]
    packed_w1:     torch.Tensor,   # BF16 [E, 2I, H]
    w2:            torch.Tensor,   # BF16 [E, H, I]
    top_k:         int,
) -> torch.Tensor:                 # BF16 [T, H]
    """
    Full fused MoE forward: routing → dispatch → expert MLP → combine.
    """
    num_tokens, hidden_size = hidden_states.shape

    # ── Step 1: Gating ───────────────────────────────────────────────────────
    logits = F.linear(hidden_states.float(), gate_weight.float(), None)

    # ── Step 2: Fused Softmax + Top-K ────────────────────────────────────────
    topk_indices, topk_weights, expert_counts = ds_kernels.moe_routing(
        logits, top_k
    )

    # ── Step 3: Exclusive prefix sum for dispatch offsets ────────────────────
    expert_offsets = ds_kernels.moe_scan(expert_counts)

    # ── Step 4: Token dispatch ────────────────────────────────────────────────
    dispatched, token_map, dispatched_weights = ds_kernels.moe_dispatch(
        hidden_states, topk_indices, topk_weights, expert_offsets
    )

    # ── Step 5: Expert MLP (GEMM1 → SwiGLU → GEMM2) ─────────────────────────
    expert_out = _fused_expert_mlp(dispatched, packed_w1, w2, expert_offsets)

    # ── Step 6: Weighted combine ──────────────────────────────────────────────
    final_output = ds_kernels.moe_combine(
        expert_out, token_map, dispatched_weights, num_tokens
    )

    return final_output


# ─────────────────────────────────────────────────────────────────────────────
# Public API 1: FusedMoELayer
# ─────────────────────────────────────────────────────────────────────────────

class FusedMoELayer(nn.Module):
    """
    Drop-in replacement for DeepSeekMoE that executes the complete fused
    CUDA pipeline (Phase 2 + Phase 3) with no Python-side expert loops.

    Expert weights are pre-packed once at construction time and stored as
    CUDA BF16 tensors for zero-copy access during inference.

    Args:
        moe_module: A DeepSeekMoE nn.Module instance whose experts, gate
                    weight, and top_k setting will be extracted.

    Example:
        fused_moe = FusedMoELayer(model.model.layers[0].mlp)
        output = fused_moe(hidden_states)
    """

    def __init__(self, moe_module: nn.Module):
        super().__init__()

        # Extract gate weight and routing config.
        if not hasattr(moe_module, "gate") or not hasattr(moe_module, "experts"):
            raise ValueError(
                "FusedMoELayer: moe_module must have 'gate' and 'experts' attributes."
            )

        # Store as plain tensors (not nn.Parameter) — inference only.
        self.gate_weight = moe_module.gate.weight.detach()
        self.top_k       = moe_module.num_experts_per_tok
        self.num_experts  = len(moe_module.experts)

        # Pre-pack expert weights into [E, 2I, H] and [E, H, I] tensors.
        # This runs once at construction time.
        print(f"[FusedMoELayer] Pre-packing {self.num_experts} expert weight tensors...")
        self.packed_w1, self.w2 = prepack_experts(list(moe_module.experts))
        print(f"  packed_w1 : {tuple(self.packed_w1.shape)}  dtype={self.packed_w1.dtype}")
        print(f"  w2        : {tuple(self.w2.shape)}  dtype={self.w2.dtype}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: BF16 CUDA tensor [num_tokens, hidden_size].
        Returns:
            BF16 CUDA tensor [num_tokens, hidden_size].
        """
        assert hidden_states.is_cuda,          "hidden_states must be on CUDA."
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be BF16."
        assert hidden_states.dim() == 2,       "hidden_states must be 2-D [T, H]."

        return _route_dispatch_combine(
            hidden_states,
            self.gate_weight,
            self.packed_w1,
            self.w2,
            self.top_k,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API 2: fused_moe_forward  (functional, backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level weight cache: maps id(experts list) → (packed_w1, w2)
_WEIGHT_CACHE: dict = {}


def fused_moe_forward(
    hidden_states: torch.Tensor,
    gate_weight:   torch.Tensor,
    experts:       list,
    top_k:         int = 6,
    packed_w1:     Optional[torch.Tensor] = None,
    w2:            Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional fused CUDA MoE forward pass (backward-compatible API).

    If packed_w1 and w2 are not provided, the function pre-packs the expert
    weights on the first call and caches them for subsequent calls using
    the id() of the experts list as the cache key.

    For maximum performance, pre-pack weights once and pass them directly:
        packed_w1, w2 = prepack_experts(experts)
        output = fused_moe_forward(h, gate_w, experts, packed_w1=packed_w1, w2=w2)

    Or use FusedMoELayer for the cleanest interface.

    Args:
        hidden_states: BF16 CUDA tensor [num_tokens, hidden_size].
        gate_weight:   FP32 CUDA tensor [num_experts, hidden_size].
        experts:       List of ExpertMLP nn.Module objects.
        top_k:         Experts selected per token (default 6).
        packed_w1:     Optional pre-packed BF16 [E, 2I, H] weight tensor.
        w2:            Optional pre-packed BF16 [E, H, I] weight tensor.

    Returns:
        BF16 CUDA tensor [num_tokens, hidden_size].
    """
    assert hidden_states.is_cuda,          "hidden_states must be on CUDA."
    assert hidden_states.dim() == 2,       "hidden_states must be [num_tokens, hidden_size]."
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be BF16."

    # ── Weight resolution: provided > cache > pack ────────────────────────────
    if packed_w1 is None or w2 is None:
        cache_key = id(experts)
        if cache_key not in _WEIGHT_CACHE:
            _WEIGHT_CACHE[cache_key] = prepack_experts(experts)
        packed_w1, w2 = _WEIGHT_CACHE[cache_key]

    return _route_dispatch_combine(
        hidden_states, gate_weight, packed_w1, w2, top_k
    )

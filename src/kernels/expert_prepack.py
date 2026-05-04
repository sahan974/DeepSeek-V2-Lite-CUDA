"""
expert_prepack.py
=================
Weight pre-packing utility for Phase 3 CUTLASS grouped GEMM kernels.

The DeepSeek-V2-Lite ExpertMLP has three projections per expert:
    gate_proj : Linear(hidden_size=2048, intermediate_size=1408, bias=False)
    up_proj   : Linear(hidden_size=2048, intermediate_size=1408, bias=False)
    down_proj : Linear(intermediate_size=1408, hidden_size=2048, bias=False)

The forward pass is:
    output = down_proj( silu(gate_proj(x)) * up_proj(x) )

GEMM1 fusion insight:
    gate_proj and up_proj have *identical* input shape [tokens, 2048].
    Stacking their weights vertically produces a [2816, 2048] matrix.
    A single GEMM computes both projections in one launch:
        gemm1_out = x @ packed_w1[e].T   -> [tokens, 2816]
    The SwiGLU kernel then splits and activates in-place:
        output = silu(gemm1_out[:, :1408]) * gemm1_out[:, 1408:]   -> [tokens, 1408]

GEMM2:
    down_proj weight is [2048, 1408] (nn.Linear stores as [out, in]).
    gemm2_out = swiglu_out @ w2[e].T     -> [tokens, 2048]

Memory layout note:
    nn.Linear stores weights as [out_features, in_features].
    CUTLASS N-major (row-major) expects the same layout.
    No transposition needed — we pass the matrix directly and let the GEMM
    compute  X @ W^T  which matches F.linear(x, W) semantics.

Returns:
    packed_w1 : torch.Tensor, shape [num_experts, 2*intermediate_size, hidden_size]
                dtype bfloat16, on CUDA.
                packed_w1[e] = cat([gate_proj.weight, up_proj.weight], dim=0)

    w2        : torch.Tensor, shape [num_experts, hidden_size, intermediate_size]
                dtype bfloat16, on CUDA.
                w2[e] = down_proj.weight
"""

import torch
from typing import List, Tuple
from torch import nn


def prepack_experts(
    experts: List[nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-packs a list of ExpertMLP modules into stacked weight tensors
    suitable for batched CUTLASS grouped GEMM calls.

    Args:
        experts: List of ExpertMLP modules, each having:
                 - gate_proj : nn.Linear [intermediate_size, hidden_size]
                 - up_proj   : nn.Linear [intermediate_size, hidden_size]
                 - down_proj : nn.Linear [hidden_size, intermediate_size]

    Returns:
        packed_w1 : Tensor [num_experts, 2*intermediate_size, hidden_size] BF16 CUDA
                    Row order per expert: [gate_proj rows ; up_proj rows]
        w2        : Tensor [num_experts, hidden_size, intermediate_size]   BF16 CUDA
    """
    if len(experts) == 0:
        raise ValueError("prepack_experts: experts list is empty.")

    # Validate interface on the first expert.
    e0 = experts[0]
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if not hasattr(e0, attr):
            raise AttributeError(
                f"prepack_experts: expert module is missing attribute '{attr}'. "
                "Expected an ExpertMLP with gate_proj, up_proj, and down_proj."
            )

    num_experts      = len(experts)
    intermediate_size = e0.gate_proj.weight.shape[0]  # [intermediate, hidden]
    hidden_size       = e0.gate_proj.weight.shape[1]

    # Verify that all experts share the same geometry.
    for i, expert in enumerate(experts):
        g_shape = expert.gate_proj.weight.shape
        u_shape = expert.up_proj.weight.shape
        d_shape = expert.down_proj.weight.shape
        assert g_shape == (intermediate_size, hidden_size), \
            f"Expert {i} gate_proj shape mismatch: expected {(intermediate_size, hidden_size)}, got {g_shape}"
        assert u_shape == (intermediate_size, hidden_size), \
            f"Expert {i} up_proj shape mismatch: expected {(intermediate_size, hidden_size)}, got {u_shape}"
        assert d_shape == (hidden_size, intermediate_size), \
            f"Expert {i} down_proj shape mismatch: expected {(hidden_size, intermediate_size)}, got {d_shape}"

    # Determine target device from the first expert's weights.
    device = e0.gate_proj.weight.device
    if not device.type == "cuda":
        raise RuntimeError(
            "prepack_experts: expert weights must be on a CUDA device. "
            f"Got device: {device}"
        )

    # ------------------------------------------------------------------ #
    # Pack GEMM1 weights: [num_experts, 2*intermediate_size, hidden_size] #
    # Row layout per expert: gate rows (0:intermediate) then up rows       #
    # ------------------------------------------------------------------ #
    packed_w1 = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )

    # Pack GEMM2 weights: [num_experts, hidden_size, intermediate_size]   #
    w2 = torch.empty(
        (num_experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )

    for i, expert in enumerate(experts):
        # gate_proj.weight is [intermediate_size, hidden_size]
        packed_w1[i, :intermediate_size, :] = expert.gate_proj.weight.to(
            dtype=torch.bfloat16, device=device
        )
        # up_proj.weight is [intermediate_size, hidden_size]
        packed_w1[i, intermediate_size:, :] = expert.up_proj.weight.to(
            dtype=torch.bfloat16, device=device
        )
        # down_proj.weight is [hidden_size, intermediate_size]
        w2[i] = expert.down_proj.weight.to(
            dtype=torch.bfloat16, device=device
        )

    return packed_w1, w2


def verify_prepack(experts: List[nn.Module]) -> bool:
    """
    Test that verifies packed_w1 and w2 match the original expert weights.

    Returns True if all assertions pass.
    Raises AssertionError with a descriptive message if any check fails.
    """
    packed_w1, w2 = prepack_experts(experts)

    for i, expert in enumerate(experts):
        gate_w = expert.gate_proj.weight.to(torch.bfloat16)
        up_w   = expert.up_proj.weight.to(torch.bfloat16)
        down_w = expert.down_proj.weight.to(torch.bfloat16)

        intermediate_size = gate_w.shape[0]

        assert torch.equal(packed_w1[i, :intermediate_size, :], gate_w), \
            f"Expert {i}: packed_w1 gate rows do not match gate_proj.weight"
        assert torch.equal(packed_w1[i, intermediate_size:, :], up_w), \
            f"Expert {i}: packed_w1 up rows do not match up_proj.weight"
        assert torch.equal(w2[i], down_w), \
            f"Expert {i}: w2 does not match down_proj.weight"

    print(f"[prepack] Verification PASSED for {len(experts)} experts.")
    print(f"  packed_w1 : {tuple(packed_w1.shape)}  dtype={packed_w1.dtype}")
    print(f"  w2        : {tuple(w2.shape)}  dtype={w2.dtype}")
    return True

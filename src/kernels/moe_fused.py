"""
src/kernels/moe_fused.py

Clean Python API wrapping the four CUDA MoE kernels behind a single function.
This replaces the Python-loop-based `DeepSeekMoE.forward()` for the routing,
dispatch, and combine steps while keeping the reference expert MLPs intact.

Public API:
    fused_moe_forward(hidden_states, gate_weight, experts, top_k) -> output

The function:
    1. Computes gating logits via a matrix multiply.
    2. Runs the fused softmax + top-K routing kernel (moe_routing).
    3. Computes expert slot offsets via prefix sum (moe_scan).
    4. Scatters tokens into expert-contiguous order (moe_dispatch).
    5. Runs each expert MLP on its assigned token slice (reference MLPs, in FP32).
    6. Gathers and accumulates weighted expert outputs back to token order (moe_combine).

Usage:
    import sys, os
    sys.path.insert(0, os.path.join(ROOT, "build"))
    from src.kernels.moe_fused import fused_moe_forward
"""

import sys
import os
import ctypes

import torch
import torch.nn.functional as F

# Ensure PyBind11 type_caster symbols are visible to the dynamic linker.
ctypes.CDLL(torch._C.__file__, ctypes.RTLD_GLOBAL)

# ds_kernels must be on sys.path (add build/ directory before importing).
try:
    import ds_kernels  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "ds_kernels not found. Add the build/ directory to PYTHONPATH:\n"
        "  export PYTHONPATH=$PYTHONPATH:$(pwd)/build"
    ) from exc


def fused_moe_forward(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    experts: list,
    top_k: int = 6,
) -> torch.Tensor:
    """
    Fused CUDA MoE forward pass.

    Replaces the per-expert Python loop in DeepSeekMoE.forward() with four
    tightly-coupled CUDA kernels for routing, offset scan, token dispatch,
    and weighted combine. Expert MLP computation uses the reference PyTorch
    modules unchanged.

    Args:
        hidden_states:  BF16 CUDA tensor, shape [num_tokens, hidden_size].
        gate_weight:    FP32 CUDA parameter, shape [n_routed_experts, hidden_size].
                        This is `model.gate.weight`.
        experts:        List of ExpertMLP nn.Module objects (reference MLPs).
        top_k:          Number of experts selected per token (default 6).

    Returns:
        BF16 CUDA tensor, shape [num_tokens, hidden_size].
        Contains the accumulated weighted expert outputs.
    """
    assert hidden_states.is_cuda,   "hidden_states must be on CUDA."
    assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_size]."
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be BF16."

    num_tokens, hidden_size = hidden_states.shape
    num_experts = len(experts)

    # ------------------------------------------------------------------
    # Step 1: Gating — compute logits [num_tokens, num_experts] in FP32.
    # ------------------------------------------------------------------
    logits = F.linear(hidden_states.float(), gate_weight.float(), None)

    # ------------------------------------------------------------------
    # Step 2: Fused Softmax + Top-K routing kernel.
    # Outputs:
    #   topk_indices [num_tokens, top_k]  int32
    #   topk_weights [num_tokens, top_k]  float32
    #   expert_counts[num_experts]        int32
    # ------------------------------------------------------------------
    topk_indices, topk_weights, expert_counts = ds_kernels.moe_routing(logits, top_k)

    # ------------------------------------------------------------------
    # Step 3: Expert offset prefix sum.
    # Outputs:
    #   expert_offsets [num_experts + 1]  int32
    #   expert_offsets[num_experts] == num_tokens * top_k
    # ------------------------------------------------------------------
    expert_offsets = ds_kernels.moe_scan(expert_counts)

    # ------------------------------------------------------------------
    # Step 4: Token dispatch — scatter into expert-contiguous order.
    # Outputs:
    #   dispatched        [num_tokens * top_k, hidden_size]  BF16
    #   token_map         [num_tokens * top_k]               int32
    #   dispatched_weights[num_tokens * top_k]               float32
    # ------------------------------------------------------------------
    dispatched, token_map, dispatched_weights = ds_kernels.moe_dispatch(
        hidden_states, topk_indices, topk_weights, expert_offsets
    )

    # ------------------------------------------------------------------
    # Step 5: Expert MLP computation — each expert processes its slice.
    # We allocate a BF16 expert_output buffer and fill it expert by expert.
    # ------------------------------------------------------------------
    total_slots = num_tokens * top_k
    expert_output = torch.zeros(
        total_slots, hidden_size,
        dtype=torch.bfloat16, device=hidden_states.device
    )

    expert_offsets_cpu = expert_offsets.cpu().tolist()

    for e, expert in enumerate(experts):
        start = expert_offsets_cpu[e]
        end   = expert_offsets_cpu[e + 1]
        if end <= start:
            continue  # No tokens assigned to this expert.

        slot_input = dispatched[start:end]          # [n_e, hidden_size]
        with torch.no_grad():
            expert_out = expert(slot_input)
        expert_output[start:end] = expert_out

    # ------------------------------------------------------------------
    # Step 6: Weighted combine — gather back to token order.
    # Output:
    #   final_output [num_tokens, hidden_size]  BF16
    # ------------------------------------------------------------------
    final_output = ds_kernels.moe_combine(
        expert_output, token_map, dispatched_weights, num_tokens
    )

    return final_output

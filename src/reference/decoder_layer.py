"""
Reference implementation of a single DeepSeek-V2-Lite Decoder Layer.

This module assembles one full transformer decoder layer from the reference
components defined in moe.py and mla.py. It matches the structure of the
official DeepseekV2DecoderLayer (layers 1-26, which use MoE).

Layer 0 (first_k_dense_replace=1) uses a dense MLP instead of MoE and is
not implemented here. This reference targets MoE layers (indices 1-26).

Data flow:
    hidden_states  ──────────────────────────────────────────┐
         │                                                    │
    input_layernorm (RMSNorm)                                │
         │                                                    │
    self_attn (MLA)                                          │
         │                                                    │
         └──── + residual ──────────────────────────────────┘
                    │  ───────────────────────────────────────┐
                    │                                          │
    post_attention_layernorm (RMSNorm)                        │
                    │                                          │
    ┌───────────────┤                                          │
    │               │                                          │
    │         MoEGate → top-6 routed experts → weighted sum  │
    │               │                                          │
    └── shared_experts (always active) ──── +                │
                                             │                │
                                        + residual ──────────┘
"""

import torch
import torch.nn as nn
from typing import Optional

from reference.mla import DeepSeekMLA, RMSNorm
from reference.moe import DeepSeekMoE


class DeepSeekDecoderLayer(nn.Module):
    """
    Single MoE decoder layer for DeepSeek-V2-Lite (layers 1-26).

    Architecture parameters (from official config.json):
        hidden_size             = 2048
        num_attention_heads     = 16
        kv_lora_rank            = 512
        qk_nope_head_dim        = 128
        qk_rope_head_dim        = 64
        v_head_dim              = 128
        moe_intermediate_size   = 1408
        n_routed_experts        = 64
        num_experts_per_tok     = 6
        n_shared_experts        = 2
        rms_norm_eps            = 1e-6
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        kv_lora_rank: int = 512,
        moe_intermediate_size: int = 1408,
        n_routed_experts: int = 64,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 2,
        attention_dropout: float = 0.0,
        rope_scaling_factor: float = 40.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 163840,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        # Pre-attention RMSNorm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Multi-head Latent Attention
        self.self_attn = DeepSeekMLA(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            attention_dropout=attention_dropout,
            rope_scaling_factor=rope_scaling_factor,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
        )

        # Pre-MoE RMSNorm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Mixture of Experts
        self.mlp = DeepSeekMoE(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_shared_experts=n_shared_experts,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:   [batch, seq_len, hidden_size]
            attention_mask:  [batch, 1, seq_len, seq_len], additive (-inf masked)
            position_ids:    [batch, seq_len], defaults to [0..seq_len-1]

        Returns:
            hidden_states:   [batch, seq_len, hidden_size]
        """
        # --- Attention sub-layer with pre-norm and residual ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # --- MoE sub-layer with pre-norm and residual ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

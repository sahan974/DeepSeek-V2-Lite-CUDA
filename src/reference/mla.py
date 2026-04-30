"""
Reference implementation of DeepSeek-V2-Lite Multi-head Latent Attention (MLA).

This is an EXACT port of the official modeling_deepseek.py `DeepseekV2Attention`
with the following simplifications for clarity:
  - No Flash Attention 2 path (the "eager" path is used)
  - No KV Cache (past_key_value) for the reference — focus is on prefill
  - attention_mask is generated internally as a causal mask

Key architectural facts (from official config.json):
  - q_lora_rank      = null   → Direct q_proj (no low-rank Q compression)
  - kv_lora_rank     = 512    → KV IS compressed to 512-dim latent
  - qk_nope_head_dim = 128    → Non-positional part of Q/K per head
  - qk_rope_head_dim = 64     → RoPE part of Q/K per head (applied to k_pe only)
  - q_head_dim       = 192    → (= 128 + 64)
  - v_head_dim       = 128
  - num_heads        = 16
  - hidden_size      = 2048
  - RoPE type        = YaRN (factor=40, original_max=4096)

Forward pass data flow:
  hidden_states [bsz, seq, 2048]
      │
      ├─ q_proj → [bsz, seq, 16*192] → reshape [bsz, 16, seq, 192]
      │     └─ split → q_nope [bsz,16,seq,128]  q_pe [bsz,16,seq,64]
      │
      ├─ kv_a_proj_with_mqa → [bsz, seq, 576]
      │     └─ split → compressed_kv [bsz,seq,512]  k_pe [bsz,1,seq,64]
      │           └─ kv_a_layernorm → kv_b_proj → [bsz, seq, 16*(128+128)]
      │                 └─ split → k_nope [bsz,16,seq,128]  v [bsz,16,seq,128]
      │
      ├─ YaRN RoPE applied to q_pe and k_pe (only the 64-dim parts)
      │
      ├─ Assemble full Q [bsz,16,seq,192] = cat(q_nope, q_pe)
      │   Assemble full K [bsz,16,seq,192] = cat(k_nope, k_pe) [broadcast k_pe]
      │
      └─ Scaled Dot-Product Attention:
            Q @ K^T * scale → causal mask → softmax (fp32) → @ V
            → reshape [bsz, seq, 16*128] → o_proj → [bsz, seq, 2048]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility: RMSNorm
# Matches DeepseekV2RMSNorm from official modeling_deepseek.py
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Utility: YaRN Rotary Embedding
# Matches DeepseekV2YarnRotaryEmbedding from official modeling_deepseek.py
# Applied ONLY to the 64-dim rope parts of Q and K.
# ---------------------------------------------------------------------------
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)

def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN Rotary Embedding.
    Config values for DeepSeek-V2-Lite:
        dim                           = qk_rope_head_dim = 64
        max_position_embeddings       = 163840
        base                          = 10000
        scaling_factor                = 40
        original_max_position_embeddings = 4096
        beta_fast                     = 32
        beta_slow                     = 1
        mscale                        = 0.707
        mscale_all_dim                = 0.707
    """
    def __init__(
        self,
        dim: int = 64,
        max_position_embeddings: int = 163840,
        base: float = 10000.0,
        scaling_factor: float = 40.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 0.707,
        mscale_all_dim: float = 0.707,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        # Compute the mscale adjustment for softmax scaling
        self._mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, self.dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", (emb.cos() * self._mscale), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self._mscale), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to q and k.
    Matches apply_rotary_pos_emb from official modeling_deepseek.py exactly.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # Official code reshapes before applying RoPE
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Main: DeepSeekMLA (Multi-head Latent Attention)
# Exact port of DeepseekV2Attention "eager" forward pass.
# ---------------------------------------------------------------------------
class DeepSeekMLA(nn.Module):
    """
    DeepSeek-V2-Lite Multi-head Latent Attention (MLA).

    Dimensions (all from official config.json):
        hidden_size      = 2048
        num_heads        = 16
        q_head_dim       = 192  (= qk_nope_head_dim + qk_rope_head_dim = 128 + 64)
        v_head_dim       = 128
        kv_lora_rank     = 512  (compressed KV latent dim)
        qk_rope_head_dim = 64   (RoPE applied only to this slice)
        qk_nope_head_dim = 128  (non-positional slice of Q/K)
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        kv_lora_rank: int = 512,
        attention_dropout: float = 0.0,
        rope_scaling_factor: float = 40.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 163840,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.attention_dropout = attention_dropout

        # --- Q Projection ---
        # q_lora_rank is null for V2-Lite → direct projection (no LoRA)
        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.q_head_dim, bias=False
        )  # [2048 → 3072]

        # --- KV Compression ---
        # Projects hidden_states to [kv_lora_rank + qk_rope_head_dim] = 576
        # The 64-dim tail is the shared RoPE key (k_pe), not compressed
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )  # [2048 → 576]

        # Normalise the compressed latent before up-projection
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)

        # Up-projects compressed KV to per-head k_nope and v
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )  # [512 → 4096]

        # Output projection
        self.o_proj = nn.Linear(
            num_heads * v_head_dim, hidden_size, bias=False
        )  # [2048 → 2048]

        # YaRN RoPE (applied only to the 64-dim rope parts of Q and K)
        self.rotary_emb = YaRNRotaryEmbedding(
            dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=rope_scaling_factor,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0.707,
        )

        # Softmax scale: accounts for q_head_dim=192 AND YaRN mscale
        mscale = yarn_get_mscale(rope_scaling_factor, 0.707)
        self.softmax_scale = (self.q_head_dim ** -0.5) * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [bsz, seq_len, hidden_size]
            attention_mask: [bsz, 1, seq_len, seq_len] (additive mask, -inf for masked)
            position_ids: [bsz, seq_len] — if None, defaults to [0, 1, ..., seq_len-1]

        Returns:
            attn_output: [bsz, seq_len, hidden_size]
        """
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # Default position_ids
        if position_ids is None:
            position_ids = torch.arange(q_len, device=device).unsqueeze(0).expand(bsz, -1)

        # --- Step 1: Q Projection ---
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # q: [bsz, 16, seq, 192]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope: [bsz, 16, seq, 128]
        # q_pe:   [bsz, 16, seq, 64]

        # --- Step 2: KV Compression & Up-Projection ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # compressed_kv: [bsz, seq, 576]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # compressed_kv: [bsz, seq, 512]
        # k_pe: [bsz, seq, 64]

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        # k_pe: [bsz, 1, seq, 64] — shared across all 16 heads

        # Up-project the compressed KV latent
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        # kv: [bsz, 16, seq, 256]
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # k_nope:       [bsz, 16, seq, 128]
        # value_states: [bsz, 16, seq, 128]

        # --- Step 3: YaRN RoPE on the 64-dim parts only ---
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # --- Step 4: Assemble full Q and K tensors ---
        # q_head_dim = 192 = 128 (nope) + 64 (rope)
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        # k_pe is [bsz, 1, seq, 64] and gets broadcast across 16 heads
        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        # --- Step 5: Scaled Dot-Product Attention ---
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        # attn_weights: [bsz, 16, seq, seq]

        # Apply causal attention mask if not provided
        if attention_mask is None:
            # Build a causal mask: upper triangle is -inf
            causal_mask = torch.full(
                (q_len, q_len), float("-inf"), device=device, dtype=query_states.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_weights = attn_weights + attention_mask

        # Upcast to fp32 for numerical stability (matches official code)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        # attn_output: [bsz, 16, seq, 128]

        # --- Step 6: Output Projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        # attn_output: [bsz, seq, 2048]

        attn_output = self.o_proj(attn_output)
        return attn_output

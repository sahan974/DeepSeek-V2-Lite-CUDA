"""
mla_cache.py
============
Latent KV Cache for DeepSeek-V2-Lite MLA attention.

The reference implementation caches the fully up-projected K and V:
    K : [bsz, 16 heads, seq, 192 dims]  -> 3072 values per token
    V : [bsz, 16 heads, seq, 128 dims]  -> 2048 values per token
    Total: 5120 BF16 values per token per layer

MLA's compression insight: all information needed to reconstruct K and V
is contained in the compressed latent:
    compressed_kv : [bsz, seq, 512]  (KV latent)
    k_pe          : [bsz, seq,  64]  (shared RoPE key, not compressed)
    Total: 576 BF16 values per token per layer  (~9x smaller)

This class manages that compressed cache. The kv_b_proj up-projection
and RoPE are applied on-the-fly during attention computation.

Memory comparison at seq=4096, bsz=1:
    Reference cache : 5120 * 4096 * 2 bytes =  40.0 MB per layer
    Latent cache    :  576 * 4096 * 2 bytes =   4.5 MB per layer
"""

import torch
from typing import Tuple, Optional


class LatentKVCache:
    """
    Manages the compressed KV cache for MLA attention.

    Stores:
        kv_latent  [bsz, max_seq, kv_lora_rank]   BF16 — raw compressed KV (pre-norm)
        k_pe_cache [bsz, max_seq, rope_head_dim]  BF16 — shared RoPE key per token

    The cache is pre-allocated to max_seq capacity at construction time.
    Tokens are appended via update() and read back via get().

    Args:
        bsz            : Batch size.
        max_seq        : Maximum sequence length to pre-allocate for.
        kv_lora_rank   : Compressed KV latent dimension. Default: 512.
        rope_head_dim  : RoPE key dimension. Default: 64.
        device         : CUDA device string or torch.device.
        dtype          : Tensor dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        bsz: int,
        max_seq: int,
        kv_lora_rank: int = 512,
        rope_head_dim: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not isinstance(bsz, int) or bsz < 1:
            raise ValueError(f"bsz must be a positive integer, got {bsz}")
        if not isinstance(max_seq, int) or max_seq < 1:
            raise ValueError(f"max_seq must be a positive integer, got {max_seq}")

        self.bsz          = bsz
        self.max_seq      = max_seq
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.device       = torch.device(device)
        self.dtype        = dtype
        self._len         = 0  # number of tokens currently in the cache

        # Pre-allocate cache buffers
        self.kv_latent  = torch.zeros(
            (bsz, max_seq, kv_lora_rank),
            dtype=dtype,
            device=self.device,
        )
        self.k_pe_cache = torch.zeros(
            (bsz, max_seq, rope_head_dim),
            dtype=dtype,
            device=self.device,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_len(self) -> int:
        """Number of tokens currently stored in the cache."""
        return self._len

    @property
    def remaining_capacity(self) -> int:
        """Number of additional tokens that can be appended."""
        return self.max_seq - self._len

    @property
    def memory_bytes(self) -> int:
        """Total bytes consumed by both cache buffers."""
        elem_size = self.kv_latent.element_size()
        return (
            self.kv_latent.numel() + self.k_pe_cache.numel()
        ) * elem_size

    # ── Core operations ───────────────────────────────────────────────────────

    def update(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """
        Append one or more new tokens to the cache.

        Args:
            compressed_kv : [bsz, new_tokens, kv_lora_rank]  BF16
            k_pe          : [bsz, new_tokens, rope_head_dim] BF16
                            (k_pe must NOT have the head dim — pass it as
                             the raw output of kv_a_proj before reshape)

        Raises:
            RuntimeError if the cache would overflow max_seq.
        """
        if compressed_kv.dim() != 3:
            raise ValueError(
                f"compressed_kv must be 3D [bsz, new_tokens, kv_lora_rank], "
                f"got shape {tuple(compressed_kv.shape)}"
            )
        if k_pe.dim() != 3:
            raise ValueError(
                f"k_pe must be 3D [bsz, new_tokens, rope_head_dim], "
                f"got shape {tuple(k_pe.shape)}"
            )

        new_tokens = compressed_kv.shape[1]

        if self._len + new_tokens > self.max_seq:
            raise RuntimeError(
                f"KV cache overflow: current_len={self._len}, "
                f"new_tokens={new_tokens}, max_seq={self.max_seq}. "
                "Increase max_seq or reset the cache."
            )

        slot = slice(self._len, self._len + new_tokens)
        self.kv_latent[:, slot, :]  = compressed_kv.to(self.device, self.dtype)
        self.k_pe_cache[:, slot, :] = k_pe.to(self.device, self.dtype)
        self._len += new_tokens

    def get(
        self,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the cached latent and k_pe up to seq_len tokens.

        Args:
            seq_len : Number of tokens to return. Defaults to current_len.

        Returns:
            kv_latent  : [bsz, seq_len, kv_lora_rank]   — zero-copy view
            k_pe_cache : [bsz, seq_len, rope_head_dim]  — zero-copy view
        """
        if seq_len is None:
            seq_len = self._len
        if seq_len > self._len:
            raise ValueError(
                f"Requested seq_len={seq_len} but cache only has {self._len} tokens."
            )
        return (
            self.kv_latent[:, :seq_len, :],
            self.k_pe_cache[:, :seq_len, :],
        )

    def reset(self) -> None:
        """
        Clear the cache for a new sequence.
        Does NOT deallocate memory — just resets the write pointer.
        """
        self._len = 0

    # ── Utility ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        used_mb = (
            self._len
            * (self.kv_lora_rank + self.rope_head_dim)
            * self.kv_latent.element_size()
            * self.bsz
        ) / (1024 ** 2)
        total_mb = self.memory_bytes / (1024 ** 2)
        return (
            f"LatentKVCache("
            f"bsz={self.bsz}, "
            f"tokens={self._len}/{self.max_seq}, "
            f"used={used_mb:.2f}MB/{total_mb:.2f}MB, "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )

    @classmethod
    def for_inference(
        cls,
        bsz: int,
        max_seq: int = 4096,
        device: str = "cuda",
    ) -> "LatentKVCache":
        """
        Convenience constructor with production-ready defaults.

        Creates a BF16 cache on CUDA with 4096 token capacity.
        Memory footprint: (512+64) * max_seq * bsz * 2 bytes
            = 576 * 4096 * 1 * 2 = 4.5 MB per layer at bsz=1, max_seq=4096.
        """
        return cls(
            bsz=bsz,
            max_seq=max_seq,
            kv_lora_rank=512,
            rope_head_dim=64,
            device=device,
            dtype=torch.bfloat16,
        )

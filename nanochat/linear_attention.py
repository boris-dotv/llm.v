"""
Lightning Attention (SimpleGLA) module for SALA hybrid model.

Uses chunk_simple_gla for training (parallel, O(n) per chunk) and
fused_recurrent_simple_gla for inference (recurrent, constant state size).
Full MHA (n_kv_head = n_head) for linear layers.
Same forward signature as CausalSelfAttention for drop-in use in Block.
"""

import math

import torch
import torch.nn as nn

from nanochat.gpt import norm, apply_rotary_emb

# Lazy import fla kernels — may not be installed
_fla_available = False
chunk_simple_gla = None
fused_recurrent_simple_gla = None

def _ensure_fla():
    global _fla_available, chunk_simple_gla, fused_recurrent_simple_gla
    if _fla_available:
        return
    try:
        from fla.ops.simple_gla import chunk_simple_gla as _chunk, fused_recurrent_simple_gla as _recur
        chunk_simple_gla = _chunk
        fused_recurrent_simple_gla = _recur
        _fla_available = True
    except ImportError:
        raise ImportError(
            "flash-linear-attention (fla) is required for LightningAttention. "
            "Install it with: pip install fla"
        )


def _build_slope_tensor(n_head):
    """ALiBi-style geometric decay slopes. Returns (n_head,) tensor in (0, 1)."""
    closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
    base = 2 ** (-8.0 / closest_power_of_2)
    powers = torch.arange(1, closest_power_of_2 + 1, dtype=torch.float32)
    slopes = base ** powers
    if closest_power_of_2 != n_head:
        extra_base = 2 ** (-4.0 / closest_power_of_2)
        extra_powers = torch.arange(1, 2 * (n_head - closest_power_of_2) + 1, 2, dtype=torch.float32)
        slopes = torch.cat([slopes, extra_base ** extra_powers])
    return slopes  # (n_head,), values in (0, 1)


class LightningAttention(nn.Module):
    """Lightning/SimpleGLA attention for SALA hybrid model."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_head  # full MHA for linear layers
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        # Full MHA projections (no GQA — linear layers use all heads for KV)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Output gate (z_proj in SALA terminology)
        self.z_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False) if config.use_output_gate else None

        # QK normalization — learnable per-head RMSNorm (unlike the parameter-free norm() used elsewhere)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.o_norm = nn.RMSNorm(self.n_embd)

        # ALiBi-style decay slopes stored in log-space for fla
        # slopes in (0, 1), log(slopes) in (-inf, 0) — fla expects log-space g
        slopes = _build_slope_tensor(self.n_head)
        self.register_buffer("g_gamma", torch.log(slopes), persistent=False)

        self.scale = self.head_dim ** -0.5
        # HyPE: linear layers always get RoPE
        self.use_rope = True

    def forward(self, x, cos_sin, window_size, kv_cache):
        _ensure_fla()
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # RoPE (HyPE: linear layers get RoPE)
        if self.use_rope:
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Learnable QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        if kv_cache is None:
            # Training: chunk-based parallel SimpleGLA
            # fla expects (B, H, T, D) layout
            q_t = q.transpose(1, 2)  # (B, H, T, D)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            # g_gamma is (H,) — expand to (1, H, 1, 1) for broadcasting
            g = self.g_gamma[None, :, None, None].expand(B, -1, T, 1)
            y, _ = chunk_simple_gla(q_t, k_t, v_t, g=g, scale=self.scale)
            y = y.transpose(1, 2)  # back to (B, T, H, D)
        else:
            # Inference: recurrent mode with state
            state = kv_cache.get_linear_state(self.layer_idx)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            g = self.g_gamma[None, :, None, None].expand(B, -1, T, 1)
            y, new_state = fused_recurrent_simple_gla(
                q_t, k_t, v_t, g=g, scale=self.scale,
                initial_state=state, output_final_state=True
            )
            kv_cache.set_linear_state(self.layer_idx, new_state)
            y = y.transpose(1, 2)

        # Output norm + gate + projection
        y = y.contiguous().view(B, T, -1)
        y = self.o_norm(y)

        if self.z_proj is not None:
            z = self.z_proj(x)  # gate from original input
            y = y * torch.sigmoid(z)

        y = self.c_proj(y)
        return y

    def init_from_dense(self, dense_attn):
        """HALO: initialize linear attention from a dense CausalSelfAttention.

        Copies Q/K/V/O weights. If dense uses GQA (fewer KV heads), expands
        K/V weights by repeating to match full MHA.
        """
        self.c_q.weight.data.copy_(dense_attn.c_q.weight.data)
        self.c_proj.weight.data.copy_(dense_attn.c_proj.weight.data)

        # Handle GQA -> MHA expansion for K and V
        if dense_attn.c_k.weight.shape[0] != self.c_k.weight.shape[0]:
            dense_kv_heads = dense_attn.c_k.weight.shape[0] // self.head_dim
            repeat_factor = self.n_head // dense_kv_heads
            # Reshape: (kv_heads * head_dim, n_embd) -> (kv_heads, head_dim, n_embd)
            k_w = dense_attn.c_k.weight.data.view(dense_kv_heads, self.head_dim, -1)
            self.c_k.weight.data.copy_(k_w.repeat(repeat_factor, 1, 1).view(-1, k_w.shape[2]))
            v_w = dense_attn.c_v.weight.data.view(dense_kv_heads, self.head_dim, -1)
            self.c_v.weight.data.copy_(v_w.repeat(repeat_factor, 1, 1).view(-1, v_w.shape[2]))
        else:
            self.c_k.weight.data.copy_(dense_attn.c_k.weight.data)
            self.c_v.weight.data.copy_(dense_attn.c_v.weight.data)

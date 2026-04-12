"""
InfLLM-V2 sparse attention: 3-stage block selection + sparse forward.

All InfLLM-V2 CUDA kernel interaction is contained here.
The CUDA extension must be built from source:
  cd oldMoney-Project/_official_pkgs/infllmv2_cuda_impl && pip install -e .
"""

from dataclasses import dataclass

import torch

# Conditional import — infllm_v2 requires a custom CUDA build
HAS_INFLLM = False
try:
    from infllm_v2 import (
        infllmv2_attn_varlen_func,
        infllmv2_attn_stage1,
        infllmv2_attn_with_kvcache,
        max_pooling_1d_varlen,
    )
    HAS_INFLLM = True
except ImportError:
    pass


@dataclass
class SparseConfig:
    """Hyperparameters for InfLLM-V2 sparse attention."""
    block_size: int = 64        # attention block granularity (tokens)
    kernel_size: int = 32       # k1 fine-grained compression kernel
    kernel_stride: int = 16     # k1 compression stride
    k2_kernel_size: int = 128   # k2 coarse compression kernel (4x k1)
    k2_kernel_stride: int = 64  # k2 compression stride (4x k1)
    topk: int = 63              # number of top-k blocks to select
    init_blocks: int = 1        # initial blocks always attended
    local_blocks: int = 32      # local/sliding window blocks

    @classmethod
    def from_gpt_config(cls, config):
        return cls(
            block_size=config.sparse_block_size,
            kernel_size=config.sparse_kernel_size,
            kernel_stride=config.sparse_kernel_stride,
            topk=config.sparse_topk,
            init_blocks=config.sparse_init_blocks,
            local_blocks=config.sparse_local_blocks,
        )


def compress_keys(k, kernel_size, stride):
    """Mean-pool keys along the token dimension for block scoring.

    Args:
        k: (total_k, n_kv_head, head_dim) — full-resolution keys
        kernel_size: pooling window size in tokens
        stride: pooling stride in tokens

    Returns:
        (total_compressed, n_kv_head, head_dim) — compressed keys
    """
    total_k, n_kv_head, head_dim = k.shape
    # Pad to make unfold work cleanly
    pad_len = (stride - (total_k % stride)) % stride
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
    # Transpose to (H, D, T) for unfold along T
    k_t = k.permute(1, 2, 0)  # (H, D, T)
    n_windows = (k_t.shape[2] - kernel_size) // stride + 1
    if n_windows <= 0:
        # Sequence too short for this compression level — return mean of entire sequence
        return k.mean(dim=0, keepdim=True)
    k_unfolded = k_t.unfold(2, kernel_size, stride)  # (H, D, n_windows, kernel_size)
    k_pooled = k_unfolded.mean(dim=-1)  # (H, D, n_windows)
    return k_pooled.permute(2, 0, 1).contiguous()  # (n_windows, H, D)


def _build_cu_seqlens(seqlens):
    """Build cumulative sequence lengths from a list/tensor of per-sequence lengths."""
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=seqlens.device)
    torch.cumsum(seqlens, dim=0, out=cu[1:])
    return cu


def sparse_block_selection(q, k, sparse_config, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_context_len):
    """3-stage block selection pipeline for InfLLM-V2.

    Stage 1: Compute attention scores between Q and mean-pooled K at two granularities
    Stage 2: Max-pool scores to block-level granularity
    Stage 3: Top-K block selection

    Args:
        q: (total_q, n_heads, head_dim)
        k: (total_k, n_kv_heads, head_dim)
        sparse_config: SparseConfig instance
        cu_seqlens_q: (batch+1,) int32
        cu_seqlens_k: (batch+1,) int32
        max_seqlen_q: int
        max_context_len: int

    Returns:
        topk_idx: (n_kv_heads, total_q, topk) int32
    """
    assert HAS_INFLLM, (
        "infllm_v2 CUDA extension not installed. "
        "Build from: oldMoney-Project/_official_pkgs/infllmv2_cuda_impl/"
    )

    cfg = sparse_config

    # Step 1: Compress K at two granularities
    k1 = compress_keys(k, cfg.kernel_size, cfg.kernel_stride)
    k2 = compress_keys(k, cfg.k2_kernel_size, cfg.k2_kernel_stride)

    # Build cu_seqlens for compressed sequences
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    seqlens_k1 = (seqlens_k - cfg.kernel_size) // cfg.kernel_stride + 1
    seqlens_k1 = seqlens_k1.clamp(min=1)
    seqlens_k2 = (seqlens_k - cfg.k2_kernel_size) // cfg.k2_kernel_stride + 1
    seqlens_k2 = seqlens_k2.clamp(min=1)
    cu_seqlens_k1 = _build_cu_seqlens(seqlens_k1)
    cu_seqlens_k2 = _build_cu_seqlens(seqlens_k2)

    # Ensure GQA ratio >= 16 for stage1 kernel (repeat q heads if needed)
    n_heads_q = q.shape[1]
    n_heads_k = k.shape[1]
    ratio = n_heads_q // n_heads_k
    q_for_stage1 = q
    if ratio < 16:
        repeat_factor = 16 // ratio
        q_for_stage1 = q.repeat_interleave(repeat_factor, dim=1)
        # Adjust cu_seqlens_q for the expanded heads
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_q_expanded = seqlens_q * (n_heads_q * repeat_factor // n_heads_k)
        cu_seqlens_q_adj = _build_cu_seqlens(seqlens_q_expanded)
        max_seqlen_q_adj = max_seqlen_q * (n_heads_q * repeat_factor // n_heads_k)
    else:
        cu_seqlens_q_adj = cu_seqlens_q
        max_seqlen_q_adj = max_seqlen_q

    # Step 2: Stage 1 — attention-based scoring
    max_seqlen_k1 = max_context_len // cfg.kernel_stride
    scores = infllmv2_attn_stage1(
        q_for_stage1.contiguous(), k1.contiguous(), k2.contiguous(),
        cu_seqlens_q=cu_seqlens_q_adj,
        cu_seqlens_k=cu_seqlens_k1,
        cu_seqlens_v=cu_seqlens_k2,
        max_seqlen_q=max_seqlen_q_adj,
        max_seqlen_k=max_seqlen_k1,
        causal=True,
        return_attn_probs=True,
    )  # (n_kv_heads, total_q, max_seqlen_k1)

    # Step 3: Max pooling to block granularity
    cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
    block_scores = max_pooling_1d_varlen(
        scores.contiguous(),
        cu_seqlens_q, cu_seqlens_k, cache_lens,
        max_seqlen_q, max_context_len,
        local_blocks=cfg.local_blocks,
        init_blocks=cfg.init_blocks,
        block_size=cfg.block_size,
        stride=cfg.kernel_stride,
    )  # (n_kv_heads, total_q, n_blocks)

    # Step 4: Top-K selection
    topk_idx = block_scores.topk(cfg.topk, dim=-1).indices.sort(-1).values
    return topk_idx.to(torch.int32)


def sparse_attention_forward(q, k, v, topk_idx, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, causal=True):
    """Run sparse attention with InfLLM-V2 block mask.

    Args:
        q: (total_q, n_heads, head_dim)
        k: (total_k, n_kv_heads, head_dim)
        v: (total_k, n_kv_heads, head_dim)
        topk_idx: (n_kv_heads, total_q, topk) int32 — selected block indices
        cu_seqlens_q, cu_seqlens_k: (batch+1,) int32
        max_seqlen_q, max_seqlen_k: int
        causal: bool

    Returns:
        (total_q, n_heads, head_dim)
    """
    assert HAS_INFLLM, "infllm_v2 CUDA extension not installed"
    return infllmv2_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal=causal,
        topk_idx=topk_idx,
    )

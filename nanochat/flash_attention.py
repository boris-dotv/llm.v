"""
Unified Flash Attention interface.

Exports `flash_attn` module that automatically selects the best available backend:
1. Flash Attention 2 (Official Library) - Best for H100/A100 (Your Setup!)
2. PyTorch SDPA (Scaled Dot Product Attention) - Fallback for CPU/Login Nodes
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Backend Detection & Setup
# =============================================================================
FLASH_ATTN_INSTALLED = False
_fa_func = None
_fa_kvcache = None

try:
    import flash_attn
    # 导入官方接口
    from flash_attn import flash_attn_func as _fa_func_impl
    from flash_attn import flash_attn_with_kvcache as _fa_kvcache_impl
    
    _fa_func = _fa_func_impl
    _fa_kvcache = _fa_kvcache_impl
    FLASH_ATTN_INSTALLED = True
except ImportError:
    pass

def _check_hardware_support():
    """Check if hardware supports Flash Attention (Ampere sm80+)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        # H100 is sm90, A100 is sm80. Both are >= 8.
        return major >= 8
    except Exception:
        return False

# 最终判定：既要装了库，又要硬件支持
HAS_FA = FLASH_ATTN_INSTALLED and _check_hardware_support()
HAS_FA3 = HAS_FA

# 打印状态日志
if HAS_FA:
    # 你的 H100 作业里应该看到这行
    logger.info(f"Flash Attention enabled. Using compiled cuda kernels from flash_attn package.")
else:
    # 你的登录节点里应该看到这行
    logger.warning("Flash Attention disabled (Hardware not compatible or lib not found). Using PyTorch SDPA.")

# Override for testing: set to 'fa', 'sdpa', or None (auto)
_override_impl = None

def _use_fa():
    """Determine whether to use FA based on availability and override."""
    if _override_impl == 'fa':
        assert HAS_FA, "Cannot override to FA: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA  # auto


# =============================================================================
# SDPA helpers (Fallback for CPU/Login Nodes)
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    device = q.device
    if Tq == Tk:
        mask = torch.tril(torch.ones(Tq, Tk, device=device, dtype=torch.bool))
        if window > 0 and window < Tq:
            row_idx = torch.arange(Tq, device=device).unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = mask & ((row_idx - col_idx) <= window)
    else:
        prefix_len = Tk - Tq
        mask = torch.zeros(Tq, Tk, device=device, dtype=torch.bool)
        mask[:, :prefix_len] = True
        mask[:, prefix_len:] = torch.tril(torch.ones(Tq, Tq, device=device, dtype=torch.bool))

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).
    """
    if _use_fa():
        # FA standard lib expects (B, T, H, D)
        return _fa_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.
    """
    if _use_fa():
        return _fa_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
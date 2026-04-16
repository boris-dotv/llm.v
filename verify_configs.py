#!/usr/bin/env python3
"""
Verify parameter counts and estimate memory for three model configs.

Uses the real GPT/GPTConfig classes on torch.device("meta") — zero memory cost.
Ground truth: current d32 model has exactly 4,026,540,096 parameters (from count_params.py).

Configs:
  d32 (current):  32 layers × 2048 dim, WITH value embeddings
  Config A:       48 layers × 2048 dim, NO value embeddings
  Config B:       32 layers × 3072 dim, NO value embeddings
"""
import torch
import nanochat.gpt as gpt_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Get real vocab size from the tokenizer (65536 on this machine)
# ---------------------------------------------------------------------------
tokenizer = get_tokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"Tokenizer vocab_size: {VOCAB_SIZE}")

# Save original has_ve
_original_has_ve = gpt_module.has_ve

D32_GROUND_TRUTH = 4_026_540_096

# ---------------------------------------------------------------------------
# Instantiate models
# ---------------------------------------------------------------------------
configs = {}
models = {}

# 1. Current d32 (with VE)
gpt_module.has_ve = _original_has_ve
cfg = GPTConfig(
    sequence_len=2048, vocab_size=VOCAB_SIZE,
    n_layer=32, n_head=16, n_kv_head=16, n_embd=2048,
    window_pattern="SSSL",
)
with torch.device("meta"):
    models["d32 (current, with VE)"] = GPT(cfg)
configs["d32 (current, with VE)"] = cfg

# 2. Config A: 48L × 2048, no VE
gpt_module.has_ve = lambda layer_idx, n_layer: False
cfg_a = GPTConfig(
    sequence_len=2048, vocab_size=VOCAB_SIZE,
    n_layer=48, n_head=16, n_kv_head=16, n_embd=2048,
    window_pattern="SSSL",
)
with torch.device("meta"):
    models["Config A: 48L×2048 (no VE)"] = GPT(cfg_a)
configs["Config A: 48L×2048 (no VE)"] = cfg_a

# 3. Config B: 32L × 3072, no VE
cfg_b = GPTConfig(
    sequence_len=2048, vocab_size=VOCAB_SIZE,
    n_layer=32, n_head=24, n_kv_head=24, n_embd=3072,
    window_pattern="SSSL",
)
with torch.device("meta"):
    models["Config B: 32L×3072 (no VE)"] = GPT(cfg_b)
configs["Config B: 32L×3072 (no VE)"] = cfg_b

# Restore
gpt_module.has_ve = _original_has_ve

# ---------------------------------------------------------------------------
# Print detailed breakdown for each model
# ---------------------------------------------------------------------------
results = {}  # name -> dict of numbers

for name, model in models.items():
    config = configs[name]
    total = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 76}")
    print(f"  {name}")
    print(f"  n_layer={config.n_layer}  n_embd={config.n_embd}  n_head={config.n_head}  "
          f"head_dim={config.n_embd // config.n_head}  vocab={config.vocab_size}")
    print(f"{'=' * 76}")

    # Module breakdown
    wte = model.transformer.wte.weight.numel()
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    blocks = sum(p.numel() for p in model.transformer.h.parameters())
    ve = sum(p.numel() for p in model.value_embeds.parameters())
    scalars = model.resid_lambdas.numel() + model.x0_lambdas.numel()

    print(f"  {'transformer.wte':.<50s} {wte:>15,}")
    print(f"  {'lm_head':.<50s} {lm_head:>15,}")
    print(f"  {'transformer.h (all blocks)':.<50s} {blocks:>15,}")
    print(f"  {'value_embeds':.<50s} {ve:>15,}")
    print(f"  {'resid_lambdas + x0_lambdas':.<50s} {scalars:>15,}")
    check = wte + lm_head + blocks + ve + scalars
    print(f"  {'SUM CHECK':.<50s} {check:>15,}  {'OK' if check == total else 'MISMATCH!'}")
    print(f"  {'TOTAL':.<50s} {total:>15,}")

    # Ground truth check for d32
    if "current" in name:
        diff = total - D32_GROUND_TRUTH
        status = "EXACT MATCH" if diff == 0 else f"OFF BY {diff:+,}"
        print(f"  {'vs ground truth (4,026,540,096)':.<50s} {status}")

    # Per-block detail (block 0)
    block0 = model.transformer.h[0]
    print(f"\n  Block 0 detail:")
    for pname, param in block0.named_parameters():
        print(f"    {pname:.<46s} {param.numel():>11,}  {list(param.shape)}")
    b0 = sum(p.numel() for p in block0.parameters())
    print(f"    {'BLOCK TOTAL':.<46s} {b0:>11,}")

    # VE detail
    n_ve = len(model.value_embeds)
    if n_ve > 0:
        print(f"\n  Value embeddings: {n_ve} tables")
        for k, emb in model.value_embeds.items():
            print(f"    ve[{k}]: {list(emb.weight.shape)} = {emb.weight.numel():,}")

    results[name] = {
        "total": total,
        "config": config,
        "blocks": blocks,
        "ve": ve,
        "wte": wte,
        "lm_head": lm_head,
    }

# ---------------------------------------------------------------------------
# Analytical activation memory estimate
# ---------------------------------------------------------------------------
def estimate_activation_memory(config, batch_size=4, has_ve=False):
    """
    Peak activation memory = all layers' saved-for-backward tensors alive at end of forward.

    Architecture (from gpt.py):
      - relu² MLP with 2 matrices (c_fc, c_proj), 4× expansion, no bias
      - MHA with c_q/c_k/c_v/c_proj, no bias, QK-norm, RoPE
      - FlashAttention (no materialized B×T×T attention matrix)
      - Functional RMSNorm (no learnable params, but saves input for backward)

    Per-layer saved tensors (bf16 = 2 bytes unless noted):

    ATTENTION SUBLAYER  x_out = x + attn(norm(x)):
      Tensor                              Size          Saved by
      ──────────────────────────────────  ────────────  ──────────────────
      1. x (norm input)                   B×T×D         rms_norm backward
      2. norm(x) output                   B×T×D         c_q/c_k/c_v backward (shared ref)
      3. q (c_q output, pre-RoPE)         B×T×D         RoPE ops (kept alive via views)
      4. k (c_k output, pre-RoPE)         B×T×D         RoPE ops (kept alive via views)
      5. q_rotated (RoPE output)          B×T×D         QK-norm backward
      6. k_rotated (RoPE output)          B×T×D         QK-norm backward
      7. q_normed (QK-norm output)        B×T×D         flash_attn backward
      8. k_normed (QK-norm output)        B×T×D         flash_attn backward
      9. v (c_v output)                   B×T×D         flash_attn backward
     10. O (flash output)                 B×T×D         flash_attn + c_proj backward
     11. softmax_lse                      B×H×T         flash_attn backward (fp32)
                                          ─────────
                                          10 × B×T×D (bf16)  +  B×H×T (fp32)

    MLP SUBLAYER  x_out = x + mlp(norm(x)):
      Tensor                              Size          Saved by
      ──────────────────────────────────  ────────────  ──────────────────
      1. x (norm input)                   B×T×D         rms_norm backward
      2. norm(x) output                   B×T×D         c_fc backward (shared ref)
      3. c_fc output (relu input)         B×T×4D        relu backward (saves output, but
                                                         output == input to .square(), and
                                                         relu out is kept alive regardless)
      4. relu output (= square input)     B×T×4D        .square() backward needs input
      5. square output (c_proj input)     B×T×4D        c_proj backward
                                          ─────────
                                          2 × B×T×D + 3 × B×T×4D (bf16)

    PER LAYER TOTAL:
      bf16 elements:  10×B×T×D + 2×B×T×D + 3×B×T×4D = (12 + 12)×B×T×D = 24×B×T×D
      fp32 elements:  B×H×T (softmax_lse)

    GLOBAL (outside layer loop):
      - x0 (initial embedding, kept for x0_lambda blending):  B×T×D (bf16)
      - logits (fp32 for softcap + cross-entropy):            B×T×padded_V (fp32)
    """
    B = batch_size
    T = config.sequence_len
    D = config.n_embd
    H = config.n_head
    L = config.n_layer
    V_pad = ((config.vocab_size + 63) // 64) * 64

    # Per-layer
    attn_bf16_elems = 10 * B * T * D
    mlp_bf16_elems = 2 * B * T * D + 3 * B * T * 4 * D  # = 14 × B×T×D
    lse_fp32_elems = B * H * T

    per_layer_bytes = (attn_bf16_elems + mlp_bf16_elems) * 2 + lse_fp32_elems * 4

    # With VE: gate backward saves x[:,:,:32] (negligible) + ve lookup (B×T×D), plus
    # the gated addition saves gate (B×T×H) and ve (B×T×D) → ~2×B×T×D extra
    # But only on layers that have VE (half the layers for d32)
    ve_extra_bytes = 0
    if has_ve:
        n_ve_layers = sum(1 for i in range(L) if _original_has_ve(i, L))
        ve_extra_bytes = n_ve_layers * 2 * B * T * D * 2  # 2 tensors × bf16

    all_layers_bytes = per_layer_bytes * L + ve_extra_bytes

    # Global
    x0_bytes = B * T * D * 2
    logits_bytes = B * T * V_pad * 4  # fp32

    total_act = all_layers_bytes + x0_bytes + logits_bytes

    # Print itemized
    unit = B * T * D  # one "unit" in elements
    print(f"\n  Activation memory (B={B}, T={T}, flash_attn, bf16):")
    print(f"    Unit (B×T×D):          {unit:,} elems = {unit * 2 / 1e6:.1f} MB")
    print(f"    Attn per layer:        10 × unit = {attn_bf16_elems * 2 / 1e6:.1f} MB  "
          f"+ lse {lse_fp32_elems * 4 / 1e6:.2f} MB")
    print(f"    MLP per layer:         14 × unit = {mlp_bf16_elems * 2 / 1e6:.1f} MB")
    print(f"    Per layer total:       24 × unit = {per_layer_bytes / 1e6:.1f} MB")
    if has_ve:
        print(f"    VE extra ({n_ve_layers} layers):  {ve_extra_bytes / 1e6:.1f} MB")
    print(f"    All {L} layers:         {all_layers_bytes / 1e9:.3f} GB")
    print(f"    x0 (global):           {x0_bytes / 1e6:.1f} MB")
    print(f"    Logits (fp32):         {logits_bytes / 1e6:.1f} MB")
    print(f"    ─────────────────────────────────────")
    print(f"    TOTAL ACTIVATIONS:     {total_act / 1e9:.3f} GB")

    return total_act


# ---------------------------------------------------------------------------
# Full memory estimate per GPU (DDP, 8 GPUs)
# ---------------------------------------------------------------------------
def estimate_total_memory(name, model, config, batch_size=4, has_ve=False):
    total_params = sum(p.numel() for p in model.parameters())

    # Split into Muon (2D block params) vs Adam (everything else)
    muon_n = 0
    adam_n = 0
    for p in model.transformer.h.parameters():
        if p.ndim == 2:
            muon_n += p.numel()
        else:
            adam_n += p.numel()
    adam_n += model.transformer.wte.weight.numel()
    adam_n += sum(p.numel() for p in model.lm_head.parameters())
    adam_n += sum(p.numel() for p in model.value_embeds.parameters())
    adam_n += model.resid_lambdas.numel() + model.x0_lambdas.numel()
    assert muon_n + adam_n == total_params

    param_bytes = total_params * 2          # bf16
    grad_bytes = total_params * 2           # bf16
    muon_opt = muon_n * 2                   # 1 momentum buffer (bf16)
    adam_opt = adam_n * 8                    # m (fp32) + v (fp32)
    ddp_buffer = total_params * 2           # gradient all-reduce bucket

    act_bytes = estimate_activation_memory(config, batch_size, has_ve)

    subtotal = param_bytes + grad_bytes + muon_opt + adam_opt + ddp_buffer + act_bytes

    print(f"\n  Memory breakdown (DDP, per-GPU, B={batch_size}):")
    print(f"    Params (bf16):           {param_bytes / 1e9:.3f} GB")
    print(f"    Gradients (bf16):        {grad_bytes / 1e9:.3f} GB")
    print(f"    Muon momentum (bf16):    {muon_opt / 1e9:.3f} GB  ({muon_n:,} params)")
    print(f"    Adam m+v (fp32):         {adam_opt / 1e9:.3f} GB  ({adam_n:,} params)")
    print(f"    DDP grad buffer:         {ddp_buffer / 1e9:.3f} GB")
    print(f"    Activations:             {act_bytes / 1e9:.3f} GB")
    print(f"    ─────────────────────────────────────")
    print(f"    SUBTOTAL:                {subtotal / 1e9:.2f} GB")
    overhead_lo, overhead_hi = 3.0, 8.0
    peak_lo = subtotal / 1e9 + overhead_lo
    peak_hi = subtotal / 1e9 + overhead_hi
    print(f"    + compile/NCCL/frag:     ~{overhead_lo:.0f}–{overhead_hi:.0f} GB")
    print(f"    ESTIMATED PEAK:          {peak_lo:.1f}–{peak_hi:.1f} GB")

    return {
        "total_params": total_params,
        "act_bytes": act_bytes,
        "subtotal": subtotal,
        "peak_lo": peak_lo,
        "peak_hi": peak_hi,
    }

# ---------------------------------------------------------------------------
# Run estimates
# ---------------------------------------------------------------------------
mem = {}
print("\n" + "#" * 76)
print("# MEMORY ESTIMATES")
print("#" * 76)

for name in models:
    has_ve = "current" in name
    mem[name] = estimate_total_memory(name, models[name], configs[name], has_ve=has_ve)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'=' * 100}")
print(f"  SUMMARY TABLE (DDP, 8×GPU, bsz=4, seq_len=2048, bf16)")
print(f"{'=' * 100}")
print(f"  {'Config':<30s} {'Params':>15s} {'Act. mem':>10s} {'Subtotal':>10s} {'Est. peak':>14s} {'80GB?':>8s}")
print(f"  {'-' * 88}")

for name in models:
    m = mem[name]
    fits = "YES" if m["peak_hi"] < 80 else ("TIGHT" if m["peak_lo"] < 80 else "NO")
    delta = m["total_params"] - 4_000_000_000
    delta_s = f"({delta/1e9:+.2f}B vs 4B)"
    print(f"  {name:<30s} {m['total_params']:>13,}  {m['act_bytes']/1e9:>8.2f}GB"
          f"  {m['subtotal']/1e9:>8.2f}GB  {m['peak_lo']:>5.1f}–{m['peak_hi']:>5.1f}GB  {fits:>8s}")
    print(f"  {'':30s} {delta_s}")

# ---------------------------------------------------------------------------
# Calibration check: compare d32 estimate against observed nvidia-smi
# ---------------------------------------------------------------------------
OBSERVED_GPU_MEM_GB = 66.8  # from nvidia-smi on 8×H100, d32 training with bsz=4

d32_key = "d32 (current, with VE)"
d32_subtotal_gb = mem[d32_key]["subtotal"] / 1e9
# Use midpoint of overhead range for best-guess peak
overhead_mid = 5.5  # midpoint of 3-8 GB
d32_estimated_peak = d32_subtotal_gb + overhead_mid
delta_gb = d32_estimated_peak - OBSERVED_GPU_MEM_GB

print(f"\n{'=' * 100}")
print(f"  CALIBRATION CHECK: d32 estimate vs observed {OBSERVED_GPU_MEM_GB} GB (nvidia-smi)")
print(f"{'=' * 100}")
print(f"  Analytical subtotal (no overhead):   {d32_subtotal_gb:.2f} GB")
print(f"  + estimated overhead (~{overhead_mid:.1f} GB):    {d32_estimated_peak:.2f} GB")
print(f"  Observed (nvidia-smi):               {OBSERVED_GPU_MEM_GB:.1f} GB")
print(f"  Delta (estimated - observed):        {delta_gb:+.2f} GB")

if abs(delta_gb) <= 5.0:
    print(f"  STATUS: Delta within 5 GB tolerance. Estimates are reasonable.")
else:
    print(f"  WARNING: Delta exceeds 5 GB!")
    print(f"  The analytical activation formula may undercount saved tensors")
    print(f"  (torch.compile fusion buffers, autograd intermediates, NCCL state).")
    print(f"  Treat the new-config memory estimates as LOWER BOUNDS.")
    # Compute implied overhead for new configs using observed calibration
    implied_overhead = OBSERVED_GPU_MEM_GB - d32_subtotal_gb
    print(f"\n  Calibrated overhead (observed - subtotal): {implied_overhead:.2f} GB")
    print(f"  If applied to other configs:")
    for name in models:
        if "current" in name:
            continue
        m = mem[name]
        calibrated = m["subtotal"] / 1e9 + implied_overhead
        fits = "YES" if calibrated < 80 else "NO"
        print(f"    {name:<30s}  calibrated peak: {calibrated:.1f} GB  fits 80GB? {fits}")

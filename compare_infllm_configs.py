#!/usr/bin/env python3
"""
Compare candidate ~4B configs compatible with InfLLM-V2 sparse attention.

InfLLM-V2 CUDA kernel constraints (from manual_impl/instruction.md):
  - head_dim must be 64 or 128 (only compiled bf16 variants)
  - GQA ratio G = n_head/n_kv_head: G=16 is NOT required. Code auto-pads
    Q heads via repeat_interleave when G < 16. Any G where n_head % n_kv_head == 0 works.

Original proposals X1/X3/X4 have head_dim=96 or 80 → DISQUALIFIED.
We keep X2 and add better alternatives with head_dim ∈ {64, 128}.
"""
import torch
import nanochat.gpt as gpt_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
V = tokenizer.get_vocab_size()
print(f"Tokenizer vocab_size: {V}")

_original_has_ve = gpt_module.has_ve
gpt_module.has_ve = lambda layer_idx, n_layer: False  # no VE for all candidates

# ---------------------------------------------------------------------------
# Define configs
# ---------------------------------------------------------------------------
candidate_configs = {
    # Original proposals (for reference — some are InfLLM-V2 incompatible)
    "X1: 32L×3072 h32 kv2 (hd=96, INVALID)": dict(
        n_layer=32, n_embd=3072, n_head=32, n_kv_head=2,  # head_dim=96 ← NOT 64/128
    ),
    "X2: 32L×3072 h24 kv2 (hd=128)": dict(
        n_layer=32, n_embd=3072, n_head=24, n_kv_head=2,  # head_dim=128, G=12 (auto-padded)
    ),
    "X3: 32L×2560 h32 kv2 (hd=80, INVALID)": dict(
        n_layer=32, n_embd=2560, n_head=32, n_kv_head=2,  # head_dim=80 ← NOT 64/128
    ),
    "X4: 40L×2560 h32 kv2 (hd=80, INVALID)": dict(
        n_layer=40, n_embd=2560, n_head=32, n_kv_head=2,  # head_dim=80 ← NOT 64/128
    ),
    # Better alternatives with head_dim=128
    "A1: 36L×3072 h24 kv2 (hd=128, deeper)": dict(
        n_layer=36, n_embd=3072, n_head=24, n_kv_head=2,  # G=12, more layers → closer to 4B
    ),
    "A2: 32L×3072 h24 kv4 (hd=128, G=6)": dict(
        n_layer=32, n_embd=3072, n_head=24, n_kv_head=4,  # G=6, more KV heads
    ),
    "A3: 28L×3584 h28 kv2 (hd=128, wider)": dict(
        n_layer=28, n_embd=3584, n_head=28, n_kv_head=2,  # G=14, wider model
    ),
    # Alternatives with head_dim=64 (more heads, native G=16 possible)
    "A4: 32L×3072 h48 kv3 (hd=64, G=16)": dict(
        n_layer=32, n_embd=3072, n_head=48, n_kv_head=3,  # G=16 native, 48 small heads
    ),
    # Reference: current d32 with VE (for comparison)
}

# ---------------------------------------------------------------------------
# Instantiate and measure
# ---------------------------------------------------------------------------
results = []

for name, kwargs in candidate_configs.items():
    head_dim = kwargs["n_embd"] // kwargs["n_head"]
    gqa_ratio = kwargs["n_head"] // kwargs["n_kv_head"]
    infllm_ok = head_dim in (64, 128)

    cfg = GPTConfig(sequence_len=2048, vocab_size=V, window_pattern="SSSL", **kwargs)
    with torch.device("meta"):
        model = GPT(cfg)

    total = sum(p.numel() for p in model.parameters())

    # Per-layer breakdown
    b0 = model.transformer.h[0]
    block_params = sum(p.numel() for p in b0.parameters())

    # Attention params per layer
    attn_params = sum(p.numel() for p in b0.attn.parameters())
    mlp_params = sum(p.numel() for p in b0.mlp.parameters())

    # KV cache size at 32K context (bf16)
    # Per layer: 2 (K+V) × n_kv_head × head_dim × seq_len × 2 bytes
    kv_per_layer = 2 * kwargs["n_kv_head"] * head_dim * 32768 * 2
    kv_total = kv_per_layer * kwargs["n_layer"]

    # Activation memory estimate (24 × B×T×D per layer, bf16)
    # Note: with GQA, K/V are smaller, so attn saves less. Adjusted:
    # Attn: norm_in(D) + norm_out(D) + Q(D) + K(kv_D) + V(kv_D) + q_rot(D) + k_rot(kv_D)
    #     + q_normed(D) + k_normed(kv_D) + O(D) = 7D + 3*kv_D
    # MLP: norm_in(D) + norm_out(D) + fc_out(4D) + relu_out(4D) + sq_out(4D) = 2D + 12D = 14D
    # Total per layer: (7D + 3*kv_D + 14D) × B×T × 2 bytes  + lse
    B, T, D = 4, 2048, kwargs["n_embd"]
    kv_D = kwargs["n_kv_head"] * head_dim
    H = kwargs["n_head"]
    L = kwargs["n_layer"]
    per_layer_act = (21 * D + 3 * kv_D) * B * T * 2 + B * H * T * 4  # +lse fp32
    V_pad = ((V + 63) // 64) * 64
    global_act = B * T * D * 2 + B * T * V_pad * 4  # x0 + logits
    total_act = per_layer_act * L + global_act

    # Full memory estimate
    muon_n = sum(p.numel() for p in model.transformer.h.parameters() if p.ndim == 2)
    adam_n = total - muon_n  # embeddings, lm_head, scalars
    param_bytes = total * 2
    grad_bytes = total * 2
    muon_opt = muon_n * 2
    adam_opt = adam_n * 8
    ddp_buf = total * 2
    subtotal = param_bytes + grad_bytes + muon_opt + adam_opt + ddp_buf + total_act

    # Calibrated peak (using -8.95 GB overhead from d32 calibration)
    calibrated_overhead = -8.95
    calibrated_peak = subtotal / 1e9 + calibrated_overhead

    results.append(dict(
        name=name, total=total, block_params=block_params,
        attn_params=attn_params, mlp_params=mlp_params,
        head_dim=head_dim, gqa_ratio=gqa_ratio, infllm_ok=infllm_ok,
        kv_total_gb=kv_total / 1e9, total_act_gb=total_act / 1e9,
        subtotal_gb=subtotal / 1e9, calibrated_peak=calibrated_peak,
        n_layer=kwargs["n_layer"], n_embd=kwargs["n_embd"],
        n_head=kwargs["n_head"], n_kv_head=kwargs["n_kv_head"],
    ))

    del model

# ---------------------------------------------------------------------------
# Print detailed table
# ---------------------------------------------------------------------------
print(f"\n{'=' * 110}")
print(f"  CANDIDATE COMPARISON (all no VE, bsz=4, seq_len=2048, bf16, DDP 8×GPU)")
print(f"{'=' * 110}")

# Detailed per-config
for r in results:
    valid = "OK" if r["infllm_ok"] else "INVALID head_dim"
    print(f"\n  {r['name']}")
    print(f"    Arch: {r['n_layer']}L × {r['n_embd']}D, {r['n_head']}Q/{r['n_kv_head']}KV heads, "
          f"head_dim={r['head_dim']}, G={r['gqa_ratio']}")
    print(f"    InfLLM-V2: {valid}")
    print(f"    Total params:     {r['total']:>15,}  ({r['total']/1e9:.3f}B, {(r['total']-4e9)/1e9:+.3f}B vs 4B)")
    print(f"    Per block:        {r['block_params']:>15,}  (attn: {r['attn_params']:,}, mlp: {r['mlp_params']:,})")
    print(f"    KV cache @32K:    {r['kv_total_gb']:>13.3f} GB")
    print(f"    Act. memory:      {r['total_act_gb']:>13.3f} GB")
    print(f"    Subtotal:         {r['subtotal_gb']:>13.2f} GB")
    print(f"    Calibrated peak:  {r['calibrated_peak']:>13.1f} GB  {'FITS 80GB' if r['calibrated_peak'] < 80 else 'DOES NOT FIT'}")

# ---------------------------------------------------------------------------
# Summary table (valid configs only)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 130}")
print(f"  SUMMARY — InfLLM-V2 COMPATIBLE CONFIGS ONLY")
print(f"{'=' * 130}")
print(f"  {'Config':<42s} {'Params':>10s} {'hd':>4s} {'G':>3s} {'Block':>10s} "
      f"{'KV@32K':>7s} {'Act':>7s} {'Peak':>7s} {'80GB?':>6s}")
print(f"  {'-' * 105}")

for r in results:
    if not r["infllm_ok"]:
        print(f"  {r['name']:<42s} {'--- INVALID: head_dim=' + str(r['head_dim']) + ' not in {64,128} ---'}")
        continue
    fits = "YES" if r["calibrated_peak"] < 80 else "NO"
    print(f"  {r['name']:<42s} {r['total']/1e9:>9.3f}B {r['head_dim']:>4d} {r['gqa_ratio']:>3d} "
          f"{r['block_params']/1e6:>9.1f}M {r['kv_total_gb']:>6.3f}G {r['total_act_gb']:>6.2f}G "
          f"{r['calibrated_peak']:>6.1f}G {fits:>6s}")

# ---------------------------------------------------------------------------
# FlashAttention head_dim support note
# ---------------------------------------------------------------------------
print(f"\n{'=' * 110}")
print(f"  FLASH ATTENTION head_dim SUPPORT")
print(f"{'=' * 110}")
print(f"  FlashAttention (FA2/FA3): supports any head_dim divisible by 8 (up to 256).")
print(f"  So head_dim=80 and 96 work for standard dense attention training.")
print(f"  BUT InfLLM-V2 sparse kernel: ONLY head_dim=64 or 128 (bf16 variants compiled).")
print(f"  → If you plan to use InfLLM-V2 later, you MUST use head_dim ∈ {{64, 128}} from the start.")
print(f"  → Training with head_dim=96 then switching to InfLLM-V2 is NOT possible without")
print(f"    re-architecting (changing n_head changes all Q/K/V/O weight shapes).")

gpt_module.has_ve = _original_has_ve

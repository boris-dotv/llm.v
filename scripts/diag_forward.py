#!/usr/bin/env python3
"""
Diagnostic forward trace for the 150M dense model.
Instruments every stage of the forward pass to find where activations blow up.
Runs entirely on CPU in fp32.

Usage (on remote GPU server):
    PYTHONPATH=. python scripts/diag_forward.py 2>&1 | tee /tmp/diag_forward.log
"""

import sys
import math
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, norm

# ===========================================================================
# Helpers
# ===========================================================================

def stat(name, t):
    """Print activation statistics for a tensor."""
    t_f = t.float()
    has_nan = t_f.isnan().any().item()
    has_inf = t_f.isinf().any().item()
    absmax = t_f.abs().max().item()
    flag = ""
    if has_nan:
        flag = " *** NaN ***"
    elif has_inf:
        flag = " *** Inf ***"
    elif absmax > 1e4:
        flag = " *** >1e4 ***"
    print(f"  {name:.<60s} shape={str(list(t.shape)):>20s}  std={t_f.std().item():12.6f}  absmax={absmax:12.4f}  nan={has_nan}  inf={has_inf}{flag}")
    return has_nan or has_inf or absmax > 1e4


def param_stat(name, p):
    """Print parameter statistics."""
    print(f"  {name:.<60s} shape={str(list(p.shape)):>20s}  std={p.float().std().item():.6f}  min={p.float().min().item():.6f}  max={p.float().max().item():.6f}")


# ===========================================================================
# Config
# ===========================================================================

BASE_CONFIG = dict(
    n_embd=768, n_layer=12, n_head=6, n_kv_head=6,
    intermediate_size=2048, vocab_size=65536, sequence_len=2048,
    mixer_types=["minicpm4"] * 12, tie_word_embeddings=True,
    scale_emb=8.0, scale_depth=1.4, dim_model_base=256,
)

B, T = 2, 128
torch.manual_seed(42)
INPUT_IDS = torch.randint(0, 65536, (B, T))


# ===========================================================================
# Step 0: Build model, print all parameter stats
# ===========================================================================

def build_model(config_kwargs):
    """Build model on CPU in fp32."""
    config = GPTConfig(**config_kwargs)
    # Use meta device + to_empty like the real training code
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    # Patch rotary embeddings to fp32 for CPU diagnostic (forward asserts bf16)
    model.cos = model.cos.float()
    model.sin = model.sin.float()
    return model


def print_all_params(model, label=""):
    print(f"\n{'='*80}")
    print(f"PARAMETER STATS{': ' + label if label else ''}")
    print(f"{'='*80}")

    # Embeddings
    print("\n--- Embeddings ---")
    param_stat("transformer.wte.weight", model.transformer.wte.weight)
    if model.lm_head is not None:
        param_stat("lm_head.weight", model.lm_head.weight)
    else:
        print("  lm_head: None (tied to wte)")

    for i, block in enumerate(model.transformer.h):
        if i > 0 and i < 11:
            continue  # only print first, last, and a middle layer
        if i == 1:
            print(f"\n--- Layers 1-10: (same init pattern, skipped) ---")
            continue
        print(f"\n--- Layer {i} ---")
        attn = block.attn
        # Attention
        param_stat(f"  h.{i}.attn.c_q.weight", attn.c_q.weight)
        param_stat(f"  h.{i}.attn.c_k.weight", attn.c_k.weight)
        param_stat(f"  h.{i}.attn.c_v.weight", attn.c_v.weight)
        param_stat(f"  h.{i}.attn.c_proj.weight", attn.c_proj.weight)
        if hasattr(attn, 'o_gate') and attn.o_gate is not None:
            param_stat(f"  h.{i}.attn.o_gate.weight", attn.o_gate.weight)
        # MLP
        param_stat(f"  h.{i}.mlp.gate_proj.weight", block.mlp.gate_proj.weight)
        param_stat(f"  h.{i}.mlp.up_proj.weight", block.mlp.up_proj.weight)
        param_stat(f"  h.{i}.mlp.down_proj.weight", block.mlp.down_proj.weight)
        # LayerNorm
        param_stat(f"  h.{i}.input_layernorm.weight", block.input_layernorm.weight)
        param_stat(f"  h.{i}.post_attn_layernorm.weight", block.post_attention_layernorm.weight)

    print(f"\n--- Final norm ---")
    param_stat("transformer.ln_f.weight", model.transformer.ln_f.weight)


# ===========================================================================
# Step 1: Manual forward trace
# ===========================================================================

def forward_trace(model, input_ids, label=""):
    """Run forward manually, printing stats at every stage."""
    print(f"\n{'='*80}")
    print(f"FORWARD TRACE{': ' + label if label else ''}")
    print(f"{'='*80}")

    config = model.config
    B, T = input_ids.shape

    # Rotary embeddings
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    window_size = (-1, 0)

    # 1. Embedding
    x_emb_raw = model.transformer.wte(input_ids)
    stat("wte(x) [before scale_emb]", x_emb_raw)

    x = x_emb_raw * config.scale_emb
    stat(f"wte(x) * scale_emb={config.scale_emb}", x)

    first_blowup_layer = None

    # 2. Each block
    for i, block in enumerate(model.transformer.h):
        # Input layernorm
        normed = block.input_layernorm(x)
        blew = stat(f"block {i:2d} | after input_layernorm", normed)

        # Attention sublayer
        with torch.no_grad():
            attn_out = block.attn(normed, cos_sin, window_size, None)
        blew |= stat(f"block {i:2d} | attn sublayer output (before residual)", attn_out)

        # Attention residual add
        x = x + attn_out * block.residual_scale
        blew |= stat(f"block {i:2d} | after attn residual add", x)

        # Post-attention layernorm
        normed2 = block.post_attention_layernorm(x)
        blew |= stat(f"block {i:2d} | after post_attn_layernorm", normed2)

        # MLP sublayer
        with torch.no_grad():
            mlp_out = block.mlp(normed2)
        blew |= stat(f"block {i:2d} | mlp sublayer output (before residual)", mlp_out)

        # MLP residual add
        x = x + mlp_out * block.residual_scale
        blew |= stat(f"block {i:2d} | after mlp residual add", x)

        if blew and first_blowup_layer is None:
            first_blowup_layer = i

    # 3. Final norm
    x = model.transformer.ln_f(x)
    stat("after ln_f", x)

    # 4. Scale width
    scale_width = config.n_embd / config.dim_model_base
    x_scaled = x / scale_width
    stat(f"after hidden / scale_width={scale_width:.4f}", x_scaled)

    # 5. Logits
    if model.lm_head is not None:
        logits = model.lm_head(x_scaled)
    else:
        logits = F.linear(x_scaled, model.transformer.wte.weight)
    logits = logits[..., :config.vocab_size]
    logits = logits.float()
    stat("logits (final)", logits)

    # 6. Loss
    targets = torch.randint(0, config.vocab_size, (B, T))
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    expected = math.log(config.vocab_size)
    print(f"\n  Loss: {loss.item():.4f}")
    print(f"  Expected ln(V={config.vocab_size}): {expected:.4f}")
    print(f"  Ratio loss/expected: {loss.item()/expected:.2f}x")
    if first_blowup_layer is not None:
        print(f"  First blowup (>1e4 or NaN/Inf): layer {first_blowup_layer}")
    else:
        print(f"  No blowup detected (all activations < 1e4)")

    return loss.item()


# ===========================================================================
# Step 6: Check residual_scale formula
# ===========================================================================

def check_residual_scale():
    print(f"\n{'='*80}")
    print(f"RESIDUAL SCALE CHECK")
    print(f"{'='*80}")
    config = GPTConfig(**BASE_CONFIG)
    expected = 1.4 / (12 ** 0.5)
    # Build a block to check the stored value
    from nanochat.gpt import Block
    block = Block(config, 0, mixer_type="minicpm4")
    actual = block.residual_scale
    print(f"  config.scale_depth = {config.scale_depth}")
    print(f"  config.n_layer = {config.n_layer}")
    print(f"  Expected: scale_depth / sqrt(n_layer) = 1.4 / sqrt(12) = {expected:.6f}")
    print(f"  Actual block.residual_scale = {actual:.6f}")
    print(f"  Match: {abs(actual - expected) < 1e-9}")
    if abs(actual - expected) > 1e-6:
        # Check common mistakes
        wrong1 = 1.4 / 12
        wrong2 = 1.4 * (12 ** 0.5)
        if abs(actual - wrong1) < 1e-6:
            print(f"  BUG: using scale_depth / n_layer = {wrong1:.6f} (division, not sqrt)")
        elif abs(actual - wrong2) < 1e-6:
            print(f"  BUG: using scale_depth * sqrt(n_layer) = {wrong2:.6f} (multiply, not divide)")


# ===========================================================================
# Step 7: Inspect init_weights summary
# ===========================================================================

def inspect_init():
    print(f"\n{'='*80}")
    print(f"INIT_WEIGHTS INSPECTION (from reading nanochat/gpt.py)")
    print(f"{'='*80}")
    print("""
  Embedding init:
    wte.weight:        Normal(mean=0, std=1.0)
    lm_head.weight:    Normal(mean=0, std=0.001)  [if not tied]

  Attention projections (per block):
    c_q.weight:        Uniform(-s, s)  where s = sqrt(3) / sqrt(n_embd)  [std = 1/sqrt(n_embd)]
    c_k.weight:        same
    c_v.weight:        same
    c_proj.weight:     ZEROS
    o_gate.weight:     ZEROS (if present)

  MLP projections (per block):
    gate_proj.weight:  Uniform(-s, s)  [same s]
    up_proj.weight:    Uniform(-s, s)  [same s]
    down_proj.weight:  ZEROS

  Learnable RMSNorm weights:
    input_layernorm.weight:         fill_(1.0)
    post_attention_layernorm.weight: fill_(1.0)
    transformer.ln_f.weight:        fill_(1.0)

  Init-time compensation for scale_emb?    NO — wte is init'd with std=1.0 regardless of scale_emb.
  Init-time compensation for depth?        NO — c_proj and down_proj are zeros, so no depth scaling
                                           is needed at init. But once training starts and these become
                                           non-zero, residual_scale = scale_depth/sqrt(n_layer) handles it.
  GPT-2 style 1/sqrt(2*n_layer) on output projections?  NO — they start at zero instead.

  Key observation for tied embeddings:
    With tie_word_embeddings=True, logits = F.linear(x / scale_width, wte.weight).
    wte.weight has std=1.0 (for embedding lookup). This is 1000x larger than the
    lm_head init (std=0.001) used in the untied case. The scale_width divisor
    (n_embd/dim_model_base) partially compensates but may not be sufficient.
    For hidden=768, dim_model_base=256: scale_width=3.0.
    Effective logit projection "std" ~ 1.0/3.0 = 0.33 vs 0.001 for untied.
    This means logit std ~ sqrt(768) * 0.33 ≈ 9.2 vs sqrt(768) * 0.001 ≈ 0.028.
    Cross-entropy loss will be much higher than ln(V) at init for tied embeddings.
""")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DIAGNOSTIC FORWARD TRACE FOR 150M DENSE MODEL")
    print("=" * 80)
    print(f"Config: {BASE_CONFIG}")
    print(f"Input shape: ({B}, {T})")

    # --- Step 0+1: Build model, print params, trace forward ---
    print("\n\n### RUN 1: Full muP config (scale_emb=8, scale_depth=1.4, dim_model_base=256) ###")
    model = build_model(BASE_CONFIG)
    print_all_params(model, "Full muP")
    loss_mup = forward_trace(model, INPUT_IDS, "Full muP")
    del model

    # --- Step 4: Ablation — disable all muP scaling ---
    print("\n\n### RUN 2: No muP (scale_emb=1.0, scale_depth=1.0, dim_model_base=768) ###")
    no_mup_config = {**BASE_CONFIG, "scale_emb": 1.0, "scale_depth": 1.0, "dim_model_base": 768}
    model_nomup = build_model(no_mup_config)
    loss_nomup = forward_trace(model_nomup, INPUT_IDS, "No muP")
    del model_nomup

    # --- Step 5: muP restored, but embedding rescaled ---
    print("\n\n### RUN 3: muP + embedding rescaled by 1/scale_emb ###")
    model_rescaled = build_model(BASE_CONFIG)
    model_rescaled.transformer.wte.weight.data *= (1.0 / model_rescaled.config.scale_emb)
    loss_rescaled = forward_trace(model_rescaled, INPUT_IDS, "muP + wte *= 1/scale_emb")
    del model_rescaled

    # --- Step 6: Residual scale check ---
    check_residual_scale()

    # --- Step 7: Init inspection ---
    inspect_init()

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    expected = math.log(65536)
    print(f"  Expected initial loss ln(65536) = {expected:.4f}")
    print(f"  Healthy range: [{expected:.2f}, {2*expected:.2f}]")
    print(f"")
    print(f"  Run 1 (full muP):              loss = {loss_mup:.4f}  {'OK' if expected <= loss_mup <= 2*expected else 'BAD'}")
    print(f"  Run 2 (no muP):                loss = {loss_nomup:.4f}  {'OK' if expected <= loss_nomup <= 2*expected else 'BAD'}")
    print(f"  Run 3 (muP + emb rescale):     loss = {loss_rescaled:.4f}  {'OK' if expected <= loss_rescaled <= 2*expected else 'BAD'}")

    print(f"\nDiagnostic complete. Output also saved to /tmp/diag_forward.log if you piped it.")

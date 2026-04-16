#!/usr/bin/env python3
"""
Diagnostic forward trace for the 150M dense model (v2).
Tests candidate fixes for the tied-embedding logit magnitude problem.

Usage (on remote GPU server):
    PYTHONPATH=. python scripts/diag_forward.py 2>&1 | tee /tmp/diag_forward_v2.log
"""

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
    print(f"  {name:.<60s} shape={str(list(t.shape)):>20s}  std={t_f.std().item():12.6f}  absmax={absmax:12.4f}{flag}")
    return has_nan or has_inf or absmax > 1e4


def param_stat(name, p):
    """Print parameter statistics."""
    print(f"  {name:.<60s} shape={str(list(p.shape)):>20s}  std={p.float().std().item():.6f}  min={p.float().min().item():.6f}  max={p.float().max().item():.6f}")


EXPECTED_LOSS = math.log(65536)  # 11.0904

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


# ===========================================================================
# Model builder
# ===========================================================================

def build_model(config_kwargs, seed=0):
    """Build model on CPU in fp32 with deterministic init."""
    torch.manual_seed(seed)
    config = GPTConfig(**config_kwargs)
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    # Patch rotary embeddings to fp32 for CPU diagnostic
    model.cos = model.cos.float()
    model.sin = model.sin.float()
    return model


# ===========================================================================
# Generic forward that returns detailed stats
# ===========================================================================

def custom_forward(model, input_ids, emb_multiplier=None, pre_logit_divisor=None,
                   output_scale=1.0, verbose=True):
    """
    Run a manual forward through the model, returning stats.

    Args:
        emb_multiplier: override for embedding scaling (default: config.scale_emb)
        pre_logit_divisor: override for the divisor before logit projection
                          (default: config.n_embd / config.dim_model_base = scale_width)
        output_scale: multiply final logits by this scalar
        verbose: print per-layer trace
    """
    config = model.config
    B, T = input_ids.shape

    if emb_multiplier is None:
        emb_multiplier = config.scale_emb
    if pre_logit_divisor is None:
        pre_logit_divisor = config.n_embd / config.dim_model_base

    cos_sin = model.cos[:, :T], model.sin[:, :T]
    window_size = (-1, 0)

    # Embedding
    x = model.transformer.wte(input_ids) * emb_multiplier
    if verbose:
        stat(f"wte(x) * emb_multiplier={emb_multiplier:.4f}", x)

    first_blowup = None

    # Blocks
    for i, block in enumerate(model.transformer.h):
        normed = block.input_layernorm(x)
        with torch.no_grad():
            attn_out = block.attn(normed, cos_sin, window_size, None)
        x = x + attn_out * block.residual_scale

        normed2 = block.post_attention_layernorm(x)
        with torch.no_grad():
            mlp_out = block.mlp(normed2)
        x = x + mlp_out * block.residual_scale

        if verbose:
            blew = stat(f"block {i:2d} | after both residuals", x)
            if blew and first_blowup is None:
                first_blowup = i

    # Final norm
    x = model.transformer.ln_f(x)

    # Pre-logit
    x_pre_logit = x / pre_logit_divisor
    pre_logit_std = x_pre_logit.float().std().item()

    # Logits
    if model.lm_head is not None:
        logits = model.lm_head(x_pre_logit)
    else:
        logits = F.linear(x_pre_logit, model.transformer.wte.weight)
    logits = logits[..., :config.vocab_size].float()
    logits = logits * output_scale

    logits_std = logits.std().item()
    logits_absmax = logits.abs().max().item()

    # Loss
    torch.manual_seed(99)  # deterministic targets
    targets = torch.randint(0, config.vocab_size, (B, T))
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).item()

    wte_std = model.transformer.wte.weight.float().std().item()

    return dict(
        wte_std=wte_std,
        pre_logit_std=pre_logit_std,
        logits_std=logits_std,
        logits_absmax=logits_absmax,
        loss=loss,
        ratio=loss / EXPECTED_LOSS,
        first_blowup=first_blowup,
    )


def report(name, r):
    """Print a single-run result line."""
    print(f"  {name:.<45s}  wte_std={r['wte_std']:.6f}  pre_logit_std={r['pre_logit_std']:.6f}"
          f"  logits_std={r['logits_std']:.4f}  logits_absmax={r['logits_absmax']:.4f}"
          f"  loss={r['loss']:.4f}  ratio={r['ratio']:.2f}x")


# ===========================================================================
# Full forward trace (verbose, for Run 1 only)
# ===========================================================================

def forward_trace(model, input_ids, label=""):
    """Full verbose trace for the first run."""
    print(f"\n{'='*80}")
    print(f"FORWARD TRACE: {label}")
    print(f"{'='*80}")

    config = model.config
    B, T = input_ids.shape
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    window_size = (-1, 0)

    x_raw = model.transformer.wte(input_ids)
    stat("wte(x) [before scale_emb]", x_raw)

    x = x_raw * config.scale_emb
    stat(f"wte(x) * scale_emb={config.scale_emb}", x)

    first_blowup = None
    for i, block in enumerate(model.transformer.h):
        normed = block.input_layernorm(x)
        blew = stat(f"block {i:2d} | after input_layernorm", normed)

        with torch.no_grad():
            attn_out = block.attn(normed, cos_sin, window_size, None)
        blew |= stat(f"block {i:2d} | attn output (before residual)", attn_out)

        x = x + attn_out * block.residual_scale
        blew |= stat(f"block {i:2d} | after attn residual", x)

        normed2 = block.post_attention_layernorm(x)
        blew |= stat(f"block {i:2d} | after post_attn_layernorm", normed2)

        with torch.no_grad():
            mlp_out = block.mlp(normed2)
        blew |= stat(f"block {i:2d} | mlp output (before residual)", mlp_out)

        x = x + mlp_out * block.residual_scale
        blew |= stat(f"block {i:2d} | after mlp residual", x)

        if blew and first_blowup is None:
            first_blowup = i

    x = model.transformer.ln_f(x)
    stat("after ln_f", x)

    scale_width = config.n_embd / config.dim_model_base
    x_scaled = x / scale_width
    stat(f"after / scale_width={scale_width:.4f}", x_scaled)

    logits = F.linear(x_scaled, model.transformer.wte.weight)
    logits = logits[..., :config.vocab_size].float()
    stat("logits (final)", logits)

    torch.manual_seed(99)
    targets = torch.randint(0, config.vocab_size, (B, T))
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    print(f"\n  Loss: {loss.item():.4f}  (expected ~{EXPECTED_LOSS:.4f}, ratio: {loss.item()/EXPECTED_LOSS:.2f}x)")
    if first_blowup is not None:
        print(f"  First blowup (>1e4/NaN/Inf): layer {first_blowup}")
    else:
        print(f"  No blowup detected")
    return loss.item()


# ===========================================================================
# Parameter stats
# ===========================================================================

def print_all_params(model, label=""):
    print(f"\n{'='*80}")
    print(f"PARAMETER STATS: {label}")
    print(f"{'='*80}")
    print("\n--- Embeddings ---")
    param_stat("transformer.wte.weight", model.transformer.wte.weight)
    if model.lm_head is not None:
        param_stat("lm_head.weight", model.lm_head.weight)
    else:
        print("  lm_head: None (tied to wte)")
    for i in [0, 11]:
        block = model.transformer.h[i]
        attn = block.attn
        print(f"\n--- Layer {i} ---")
        param_stat(f"  h.{i}.attn.c_q.weight", attn.c_q.weight)
        param_stat(f"  h.{i}.attn.c_proj.weight", attn.c_proj.weight)
        param_stat(f"  h.{i}.mlp.gate_proj.weight", block.mlp.gate_proj.weight)
        param_stat(f"  h.{i}.mlp.down_proj.weight", block.mlp.down_proj.weight)
        param_stat(f"  h.{i}.input_layernorm.weight", block.input_layernorm.weight)
        param_stat(f"  h.{i}.post_attn_layernorm.weight", block.post_attention_layernorm.weight)
    print(f"\n--- Final norm ---")
    param_stat("transformer.ln_f.weight", model.transformer.ln_f.weight)


# ===========================================================================
# Residual scale check
# ===========================================================================

def check_residual_scale():
    print(f"\n{'='*80}")
    print(f"RESIDUAL SCALE CHECK")
    print(f"{'='*80}")
    from nanochat.gpt import Block
    config = GPTConfig(**BASE_CONFIG)
    block = Block(config, 0, mixer_type="minicpm4")
    actual = block.residual_scale
    expected = 1.4 / (12 ** 0.5)
    print(f"  scale_depth={config.scale_depth}, n_layer={config.n_layer}")
    print(f"  Expected: 1.4 / sqrt(12) = {expected:.6f}")
    print(f"  Actual:   {actual:.6f}")
    print(f"  Match: {abs(actual - expected) < 1e-9}")


# ===========================================================================
# Init inspection
# ===========================================================================

def inspect_init():
    print(f"\n{'='*80}")
    print(f"INIT_WEIGHTS INSPECTION")
    print(f"{'='*80}")
    print("""
  Embedding:   wte.weight = Normal(0, std=1.0)
  LM head:     lm_head.weight = Normal(0, std=0.001)  [if untied]
  Attn Q/K/V:  Uniform(-s, s), s = sqrt(3)/sqrt(n_embd)  →  std = 1/sqrt(n_embd)
  Attn c_proj:  ZEROS
  MLP gate/up:  Uniform(-s, s)  [same s]
  MLP down:     ZEROS
  RMSNorm:      weight.fill_(1.0)

  Init-time compensation for scale_emb?  NO
  Init-time compensation for depth?      NO (output projs start at zero)
  GPT-2 1/sqrt(2*n_layer) scaling?       NO

  Tied-embedding problem:
    wte.weight has std=1.0 → used as logit projection → logit_std ≈ sqrt(768)/3 ≈ 9.2
    Untied lm_head has std=0.001 → logit_std ≈ sqrt(768)*0.001 ≈ 0.028
    That's a 330x gap. Need to close it for tied embeddings.
""")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    INPUT_IDS = torch.randint(0, 65536, (B, T))

    print("=" * 80)
    print("DIAGNOSTIC FORWARD TRACE v2 — 150M DENSE MODEL")
    print("=" * 80)
    print(f"Config: {BASE_CONFIG}")
    print(f"Input shape: ({B}, {T})")
    print(f"Expected loss ln(65536) = {EXPECTED_LOSS:.4f}")
    print(f"Healthy range: [{EXPECTED_LOSS:.2f}, {2*EXPECTED_LOSS:.2f}]")

    results = {}

    # ===================================================================
    # RUN 1: Full muP (verbose trace)
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 1: Full muP (tied, scale_emb=8, scale_depth=1.4, dim_model_base=256)")
    print("#" * 80)
    model = build_model(BASE_CONFIG, seed=0)
    print_all_params(model, "Full muP")
    forward_trace(model, INPUT_IDS, "Full muP")
    r1 = custom_forward(model, INPUT_IDS, verbose=False)
    results["Run 1: full muP (tied)"] = r1
    del model

    # ===================================================================
    # RUN 2: No muP
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 2: No muP (tied, scale_emb=1, scale_depth=1, dim_model_base=768)")
    print("#" * 80)
    no_mup = {**BASE_CONFIG, "scale_emb": 1.0, "scale_depth": 1.0, "dim_model_base": 768}
    model = build_model(no_mup, seed=0)
    r2 = custom_forward(model, INPUT_IDS, verbose=False)
    results["Run 2: no muP (tied)"] = r2
    del model

    # ===================================================================
    # RUN 3: muP + wte *= 1/scale_emb
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 3: muP + wte.weight *= 1/scale_emb after init")
    print("#" * 80)
    model = build_model(BASE_CONFIG, seed=0)
    model.transformer.wte.weight.data *= (1.0 / model.config.scale_emb)
    r3 = custom_forward(model, INPUT_IDS, verbose=False)
    results["Run 3: muP + wte/=scale_emb"] = r3
    del model

    # ===================================================================
    # RUN 4: Tied, extra 1/scale_emb in pre-logit (forward-only change)
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 4: Tied, logits = F.linear(x / scale_width / scale_emb, wte.weight)")
    print("#" * 80)
    model = build_model(BASE_CONFIG, seed=0)
    scale_width = model.config.n_embd / model.config.dim_model_base
    r4 = custom_forward(model, INPUT_IDS,
                        pre_logit_divisor=scale_width * model.config.scale_emb,
                        verbose=False)
    results["Run 4: tied, /scale_width/scale_emb"] = r4
    del model

    # ===================================================================
    # RUN 5: Tied, wte init std=1/sqrt(hidden), forward uses sqrt(hidden)
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 5: Tied, wte std=1/sqrt(768), emb_multiplier=sqrt(768)")
    print("#" * 80)
    model = build_model(BASE_CONFIG, seed=0)
    # Rescale wte to std=1/sqrt(768): divide by sqrt(768)
    # (init was std=1.0, we want std=1/sqrt(768))
    hidden = model.config.n_embd
    model.transformer.wte.weight.data /= (hidden ** 0.5)
    wte_std_after = model.transformer.wte.weight.float().std().item()
    print(f"  wte.weight std after rescale: {wte_std_after:.6f} (target: {1/(hidden**0.5):.6f})")
    # Forward: emb_multiplier=sqrt(768) instead of scale_emb=8
    # This means wte(x) has per-element std=1/sqrt(768), multiplied by sqrt(768) → std≈1
    # Logits: F.linear(x/scale_width, wte.weight) where wte has std=1/sqrt(768)
    # logit_std ≈ sqrt(768) * (1/scale_width) * (1/sqrt(768)) = 1/scale_width ≈ 0.33
    r5 = custom_forward(model, INPUT_IDS,
                        emb_multiplier=hidden ** 0.5,
                        verbose=False)
    results["Run 5: wte std=1/sqrt(d), emb*=sqrt(d)"] = r5
    del model

    # ===================================================================
    # RUN 6: Tied, output_scale sweep
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 6: Tied, logits *= output_scale  (sweep)")
    print("#" * 80)
    for output_scale in [1.0, 0.3, 0.1, 0.03]:
        model = build_model(BASE_CONFIG, seed=0)
        r = custom_forward(model, INPUT_IDS, output_scale=output_scale, verbose=False)
        label = f"Run 6: output_scale={output_scale}"
        results[label] = r
        del model

    # ===================================================================
    # RUN 7: Untied baseline (sanity check)
    # ===================================================================
    print("\n\n" + "#" * 80)
    print("### RUN 7: Untied baseline (tie_word_embeddings=False)")
    print("#" * 80)
    untied_config = {**BASE_CONFIG, "tie_word_embeddings": False}
    model = build_model(untied_config, seed=0)
    param_stat("lm_head.weight", model.lm_head.weight)
    r7 = custom_forward(model, INPUT_IDS, verbose=False)
    results["Run 7: UNTIED baseline"] = r7
    del model

    # ===================================================================
    # Residual scale + init check
    # ===================================================================
    check_residual_scale()
    inspect_init()

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"  Expected loss: {EXPECTED_LOSS:.4f}")
    print(f"  Healthy range: [{EXPECTED_LOSS:.2f}, {2*EXPECTED_LOSS:.2f}]")
    print()
    header = f"  {'Run':<45s}  {'wte_std':>10s}  {'pre_logit':>10s}  {'logit_std':>10s}  {'absmax':>10s}  {'loss':>10s}  {'ratio':>6s}  {'|Δ|':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Sort by distance from expected loss
    ranked = sorted(results.items(), key=lambda kv: abs(kv[1]['loss'] - EXPECTED_LOSS))

    for name, r in ranked:
        delta = abs(r['loss'] - EXPECTED_LOSS)
        in_range = EXPECTED_LOSS <= r['loss'] <= 2 * EXPECTED_LOSS
        marker = " <-- OK" if in_range else ""
        loss_str = f"{r['loss']:.4f}" if not math.isnan(r['loss']) else "NaN"
        print(f"  {name:<45s}  {r['wte_std']:10.6f}  {r['pre_logit_std']:10.6f}"
              f"  {r['logits_std']:10.4f}  {r['logits_absmax']:10.4f}"
              f"  {loss_str:>10s}  {r['ratio']:5.2f}x  {delta:8.4f}{marker}")

    print(f"\nDone. Pipe to /tmp/diag_forward_v2.log for the record.")

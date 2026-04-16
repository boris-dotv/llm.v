#!/usr/bin/env python3
"""
Instantiate two candidate 4B configs on meta device (no GPU/data needed),
print exact parameter counts and analytical activation memory estimates.

Configs (no value embeddings, standard MHA, relu² with 4× MLP expansion):
  A) 48 layers × 2048 dim
  B) 32 layers × 3072 dim

Also prints the current d32 model (with VE) for comparison.
"""
import torch
import nanochat.gpt as gpt_module
from nanochat.gpt import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Helper: build config matching base_train.py logic
# ---------------------------------------------------------------------------
def make_config(depth, aspect_ratio=64, head_dim=128, seq_len=2048, vocab_size=32781):
    num_layers = depth
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,  # MHA, no GQA
        n_embd=model_dim,
        window_pattern="SSSL",
    )

# ---------------------------------------------------------------------------
# Helper: print param breakdown
# ---------------------------------------------------------------------------
def print_param_breakdown(model, label, config):
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 72}")
    print(f" {label}")
    print(f" n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, "
          f"head_dim={config.n_embd // config.n_head}")
    print(f"{'=' * 72}")
    print(f"  {'TOTAL':.<52s} {total:>15,}")
    print(f"  {'-' * 68}")

    # Top-level modules
    wte_n = model.transformer.wte.weight.numel()
    lm_n = sum(p.numel() for p in model.lm_head.parameters())
    blocks_n = sum(p.numel() for p in model.transformer.h.parameters())
    ve_n = sum(p.numel() for p in model.value_embeds.parameters())
    scalars_n = model.resid_lambdas.numel() + model.x0_lambdas.numel()
    ve_gate_n = sum(
        b.attn.ve_gate.weight.numel()
        for b in model.transformer.h
        if hasattr(b.attn, 've_gate') and b.attn.ve_gate is not None
    )

    print(f"  {'transformer.wte':.<52s} {wte_n:>15,}")
    print(f"  {'lm_head':.<52s} {lm_n:>15,}")
    print(f"  {'transformer.h (all blocks)':.<52s} {blocks_n:>15,}")
    print(f"  {'value_embeds':.<52s} {ve_n:>15,}")
    print(f"  {'resid_lambdas + x0_lambdas':.<52s} {scalars_n:>15,}")
    print(f"  {'ve_gates (inside blocks)':.<52s} {ve_gate_n:>15,}")

    # Per-block breakdown (block 0)
    block0 = model.transformer.h[0]
    print(f"\n  Per-block breakdown (block 0):")
    for name, param in block0.named_parameters():
        print(f"    h.0.{name:.<44s} {param.numel():>13,}")
    b0_total = sum(p.numel() for p in block0.parameters())
    print(f"    {'block 0 total':.<48s} {b0_total:>13,}")

    # Check if all blocks are the same size
    block_sizes = [sum(p.numel() for p in b.parameters()) for b in model.transformer.h]
    if len(set(block_sizes)) == 1:
        print(f"    (all {config.n_layer} blocks identical)")
    else:
        unique = sorted(set(block_sizes))
        for s in unique:
            count = block_sizes.count(s)
            print(f"    {count} blocks with {s:,} params")

    # VE breakdown
    if ve_n > 0:
        print(f"\n  Value embeddings ({len(model.value_embeds)} tables):")
        for name, mod in model.value_embeds.items():
            n = mod.weight.numel()
            print(f"    ve[{name}]: {mod.weight.shape} = {n:,}")

    return total

# ---------------------------------------------------------------------------
# Analytical activation memory estimate
# ---------------------------------------------------------------------------
def estimate_activations(config, batch_size=4, has_ve=False):
    """
    Estimate peak activation memory (all layers' saved-for-backward tensors alive
    simultaneously at end of forward pass).

    Traces through the forward pass and counts what autograd saves:

    GPT.forward:
      x = wte(idx)                    # embedding lookup: saves idx (int64, negligible)
      x = norm(x)                     # rms_norm: saves x (B,T,D)
      x0 = x                         # reference, no cost
      for each layer:
        x = λ*x + λ0*x0              # elementwise, saves x and x0 refs (no new alloc)
        ve = value_embeds[i](idx)     # if has_ve: saves idx (already alive)
        x = block(x, ve, ...)

    Block.forward (x_in -> x_out = x_in + attn(norm(x_in)) + mlp(norm(...))):
      Attention sublayer:
        norm_x = norm(x)              # rms_norm backward saves: x .............. (B,T,D)
                                      # rms_norm output kept alive by Linear:      (B,T,D)
        q = c_q(norm_x).view(B,T,H,d)  # Linear backward saves: norm_x (shared ref)
        k = c_k(norm_x).view(...)       # Linear backward saves: norm_x (shared ref)
        v = c_v(norm_x).view(...)       # Linear backward saves: norm_x (shared ref)
                                      # c_q/c_k/c_v all share 1 copy of norm_x

        [if has_ve: v = v + gate * ve]  # gate backward saves: x[:,:,:32], ve (B,T,D)
                                        # adds ~2 extra tensors

        q = apply_rotary_emb(q, cos, sin)
          # decomposed ops: x1,x2 = q[...,:d], q[...,d:]  (views keep q alive)
          # mul/add/cat backward saves: q (via views) ... (B,T,D) [= c_q output]
          # and creates output ........................... (B,T,D)
        k = apply_rotary_emb(k, cos, sin)
          # same: keeps k alive .......................... (B,T,D) [= c_k output]
          # creates output ............................... (B,T,D)

        q = norm(q)  # saves q_rotated (input) .......... (B,T,D)
                      # output q_normed is new tensor ... (B,T,D)
        k = norm(k)  # saves k_rotated (input) .......... (B,T,D)
                      # output k_normed is new tensor ... (B,T,D)

        y = flash_attn(q_normed, k_normed, v)
          # FlashAttn backward saves: q, k, v, out ...... 4×(B,T,D)
          # but q_normed and k_normed are already alive (norm outputs)
          # and v is already alive (c_v output, possibly modified by VE)
          # NEW tensors from flash: out .................. (B,T,D)
          # plus softmax_lse ............................. (B,H,T) fp32

        y = c_proj(y.view(B,T,-1))
          # view is free; Linear backward saves input = flash_out (already alive)

      Attention sublayer unique saved tensors (no VE):
        1. x (norm input)                      B×T×D
        2. norm(x) (shared by c_q/c_k/c_v)    B×T×D
        3. q = c_q output (kept by RoPE views) B×T×D
        4. k = c_k output (kept by RoPE views) B×T×D
        5. q_rotated (RoPE output, norm input)  B×T×D
        6. k_rotated (RoPE output, norm input)  B×T×D
        7. q_normed (norm output, flash input)  B×T×D  [shared with flash]
        8. k_normed (norm output, flash input)  B×T×D  [shared with flash]
        9. v = c_v output (flash input)         B×T×D  [shared with flash]
       10. flash_out = O (flash + c_proj input) B×T×D  [shared with c_proj]
       11. softmax_lse                          B×H×T  (fp32)
      ─────────────────────────────────────
        = 10 × B×T×D  (bf16)  +  B×H×T×4 (fp32)

      With VE: add ~2 × B×T×D (gate input slice + ve lookup)

      MLP sublayer:
        norm_x = norm(x)              # saves x ................. (B,T,D)
                                      # norm_x (output) ......... (B,T,D)
        h = c_fc(norm_x)              # saves norm_x (shared ref)
                                      # h = fc output ........... (B,T,4D)
        h = F.relu(h).square()
          # relu backward: saves output (to check >0) .. (B,T,4D)
          # square backward: saves input (=relu out) .... (same tensor)
          # square output is new ........................ (B,T,4D)
        y = c_proj(h_squared)          # saves h_squared (shared ref)

      MLP sublayer unique saved tensors:
        1. x (norm input)                       B×T×D  [= attention sublayer output]
        2. norm(x) (norm output, fc input)      B×T×D
        3. c_fc output / relu input             B×T×4D
        4. relu output (= square input, shared) B×T×4D
        5. square output (= c_proj input)       B×T×4D
      ─────────────────────────────────────
        = 2 × B×T×D + 3 × B×T×4D  (bf16)
        = B×T×(2D + 12D) = 14 × B×T×D

    Per-layer total: (10 + 14) × B×T×D = 24 × B×T×D  (bf16 elements)
    Per-layer bytes: 24 × B×T×D × 2  +  B×H×T×4  (softmax_lse)

    Plus global tensors:
      - x0 (initial embedding, kept for x0_lambda blending): B×T×D
      - logits (fp32 for softcap+loss): B×T×V×4
      - loss targets etc: negligible
    """
    B, T, D = batch_size, config.sequence_len, config.n_embd
    H = config.n_head
    L = config.n_layer
    V = config.vocab_size
    padded_V = ((V + 63) // 64) * 64
    expansion = 4  # MLP expansion ratio

    elem = B * T * D  # one "unit" tensor in elements

    # Per-layer activation elements (bf16)
    attn_bf16 = 10 * elem                        # 10 × B×T×D
    attn_fp32 = B * H * T                         # softmax_lse
    mlp_bf16 = 2 * elem + 3 * expansion * elem   # 2×D + 3×4D = 14×D units
    if has_ve:
        attn_bf16 += 2 * elem                     # ve lookup + gate input

    per_layer_bytes = (attn_bf16 + mlp_bf16) * 2 + attn_fp32 * 4
    all_layers_bytes = per_layer_bytes * L

    # Global tensors
    x0_bytes = elem * 2                                  # initial embedding kept alive
    logits_bytes = B * T * padded_V * 4                  # fp32 logits for softcap + loss
    # The embedding lookup input (idx) is int64: B×T×8 — negligible

    total_activation_bytes = all_layers_bytes + x0_bytes + logits_bytes

    print(f"\n  Analytical activation memory estimate (B={B}, T={T}):")
    print(f"    Per-layer unit (B×T×D): {elem:,} elements = {elem * 2 / 1e6:.1f} MB (bf16)")
    print(f"    Attention sublayer:  10 × unit = {attn_bf16 * 2 / 1e9:.3f} GB (bf16)"
          f" + lse {attn_fp32 * 4 / 1e6:.1f} MB (fp32)")
    print(f"    MLP sublayer:        14 × unit = {mlp_bf16 * 2 / 1e9:.3f} GB (bf16)")
    per_layer_units = 24 + (2 if has_ve else 0)
    print(f"    Per-layer total:     {per_layer_units} × unit = {per_layer_bytes / 1e9:.3f} GB")
    print(f"    All {L} layers:      {all_layers_bytes / 1e9:.2f} GB")
    print(f"    x0 embedding:        {x0_bytes / 1e6:.1f} MB")
    print(f"    Logits (fp32):       {logits_bytes / 1e9:.3f} GB")
    print(f"    ─────────────────────────────────────")
    print(f"    TOTAL ACTIVATIONS:   {total_activation_bytes / 1e9:.2f} GB")
    return total_activation_bytes

# ---------------------------------------------------------------------------
# Full memory estimate
# ---------------------------------------------------------------------------
def estimate_total_memory(model, config, label, batch_size=4, has_ve=False):
    total_params = sum(p.numel() for p in model.parameters())

    # Separate Muon vs Adam params (matches setup_optimizers logic)
    muon_params = 0
    adam_params = 0
    for p in model.transformer.h.parameters():
        if p.ndim == 2:
            muon_params += p.numel()
        else:
            adam_params += p.numel()
    # Embeddings, lm_head, value_embeds, scalars → Adam
    adam_params += model.transformer.wte.weight.numel()
    adam_params += sum(p.numel() for p in model.lm_head.parameters())
    adam_params += sum(p.numel() for p in model.value_embeds.parameters())
    adam_params += model.resid_lambdas.numel() + model.x0_lambdas.numel()

    assert muon_params + adam_params == total_params, \
        f"Param split mismatch: {muon_params} + {adam_params} != {total_params}"

    param_bytes = total_params * 2         # bf16
    grad_bytes = total_params * 2          # bf16 (same dtype as params)
    muon_opt_bytes = muon_params * 2       # 1 momentum buffer, bf16
    adam_opt_bytes = adam_params * (4 + 4)  # m (fp32) + v (fp32)
    ddp_grad_buffer = total_params * 2     # DDP all-reduce gradient bucket

    act_bytes = estimate_activations(config, batch_size, has_ve)

    print(f"\n  Full memory estimate (DDP, 8 GPUs, per-device):")
    print(f"    Model params (bf16):     {param_bytes / 1e9:.2f} GB  ({total_params:,} params)")
    print(f"    Gradients (bf16):        {grad_bytes / 1e9:.2f} GB")
    print(f"    Muon momentum (bf16):    {muon_opt_bytes / 1e9:.2f} GB  ({muon_params:,} params)")
    print(f"    Adam m+v (fp32):         {adam_opt_bytes / 1e9:.2f} GB  ({adam_params:,} params)")
    print(f"    DDP grad buffer:         {ddp_grad_buffer / 1e9:.2f} GB")
    print(f"    Activations:             {act_bytes / 1e9:.2f} GB")

    subtotal = param_bytes + grad_bytes + muon_opt_bytes + adam_opt_bytes + ddp_grad_buffer + act_bytes
    print(f"    ─────────────────────────────────────")
    print(f"    SUBTOTAL:                {subtotal / 1e9:.2f} GB")
    print(f"    (+ torch.compile, NCCL, fragmentation — typically 3-8 GB)")
    print(f"    ESTIMATED PEAK:          {subtotal / 1e9:.2f} – {(subtotal / 1e9) + 8:.2f} GB")

    fits_80 = (subtotal / 1e9 + 8) < 80
    print(f"    Fits 80 GB?              {'YES' if fits_80 else 'NO / TIGHT'}")
    return subtotal

# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    # Monkey-patch has_ve to disable value embeddings for candidate configs
    original_has_ve = gpt_module.has_ve

    # --- Current d32 (WITH value embeddings) ---
    gpt_module.has_ve = original_has_ve  # restore
    config_d32 = make_config(depth=32)
    with torch.device("meta"):
        model_d32 = GPT(config_d32)
    print_param_breakdown(model_d32, "Current d32 (WITH value embeddings)", config_d32)
    estimate_total_memory(model_d32, config_d32, "d32", has_ve=True)

    # --- Disable VE for candidate configs ---
    gpt_module.has_ve = lambda layer_idx, n_layer: False

    # Config A: 48L × 2048
    config_a = make_config(depth=48, aspect_ratio=64//3 * 3)
    # Actually, we want exactly 2048 dim. depth=48, aspect_ratio=64 would give
    # base_dim = 48*64 = 3072, which is NOT 2048. We need to set it directly.
    # So let's build the config manually:
    config_a = GPTConfig(
        sequence_len=2048, vocab_size=32781,
        n_layer=48, n_head=16, n_kv_head=16, n_embd=2048,
        window_pattern="SSSL",
    )
    with torch.device("meta"):
        model_a = GPT(config_a)
    print_param_breakdown(model_a, "Config A: 48L × 2048 (no VE)", config_a)
    estimate_total_memory(model_a, config_a, "48L×2048", has_ve=False)

    # Config B: 32L × 3072
    config_b = GPTConfig(
        sequence_len=2048, vocab_size=32781,
        n_layer=32, n_head=24, n_kv_head=24, n_embd=3072,
        window_pattern="SSSL",
    )
    with torch.device("meta"):
        model_b = GPT(config_b)
    print_param_breakdown(model_b, "Config B: 32L × 3072 (no VE)", config_b)
    estimate_total_memory(model_b, config_b, "32L×3072", has_ve=False)

    # --- Summary table ---
    params = {
        "d32 (current, with VE)": sum(p.numel() for p in model_d32.parameters()),
        "48L × 2048 (no VE)":     sum(p.numel() for p in model_a.parameters()),
        "32L × 3072 (no VE)":     sum(p.numel() for p in model_b.parameters()),
    }
    print(f"\n{'=' * 72}")
    print(f" Summary")
    print(f"{'=' * 72}")
    print(f"  {'Config':.<40s} {'Params':>15s}")
    print(f"  {'-' * 56}")
    for name, n in params.items():
        print(f"  {name:.<40s} {n:>15,}  ({n/1e9:.3f}B)")

    # Restore
    gpt_module.has_ve = original_has_ve

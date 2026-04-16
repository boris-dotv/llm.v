#!/usr/bin/env python3
"""Instantiate the d32 model on CPU and print exact parameter counts."""
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# Replicate the config logic from scripts/base_train.py
depth = 32
aspect_ratio = 64
head_dim_target = 128

num_layers = depth
base_dim = depth * aspect_ratio
model_dim = ((base_dim + head_dim_target - 1) // head_dim_target) * head_dim_target
num_heads = model_dim // head_dim_target
num_kv_heads = num_heads

print(f"vocab_size:  {vocab_size}")
print(f"num_layers:  {num_layers}")
print(f"model_dim:   {model_dim} (base: {base_dim})")
print(f"num_heads:   {num_heads}")
print(f"num_kv_heads:{num_kv_heads}")
print(f"head_dim:    {model_dim // num_heads}")
print()

config = GPTConfig(
    sequence_len=2048,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    window_pattern="SSSL",
)

with torch.device("meta"):
    model = GPT(config)

# --- Parameter counts ---
total = sum(p.numel() for p in model.parameters())
print(f"{'TOTAL':.<50s} {total:>15,}")
print(f"{'':=<66s}")

# Top-level breakdown
for name, mod in [("transformer.wte", model.transformer.wte),
                  ("transformer.h (all blocks)", model.transformer.h),
                  ("lm_head", model.lm_head),
                  ("resid_lambdas", None),
                  ("x0_lambdas", None),
                  ("value_embeds", model.value_embeds)]:
    if mod is not None:
        n = sum(p.numel() for p in mod.parameters())
    elif name == "resid_lambdas":
        n = model.resid_lambdas.numel()
    else:
        n = model.x0_lambdas.numel()
    print(f"  {name:.<48s} {n:>13,}")

print()
print(f"{'Per-block breakdown (block 0 as example)':=<66s}")
block0 = model.transformer.h[0]
for name, param in block0.named_parameters():
    print(f"  h.0.{name:.<42s} {param.numel():>13,}")
block0_total = sum(p.numel() for p in block0.parameters())
print(f"  {'block total':.<48s} {block0_total:>13,}")
print(f"  {'x 32 blocks':.<48s} {block0_total * 32:>13,}")

print()
print(f"{'Value embeddings breakdown':=<66s}")
for name, mod in model.value_embeds.items():
    n = sum(p.numel() for p in mod.parameters())
    print(f"  value_embeds.{name:.<34s} {n:>13,}")
ve_total = sum(p.numel() for p in model.value_embeds.parameters())
print(f"  {'value_embeds total':.<48s} {ve_total:>13,}")

# Scaling params (Kaplan definition: exclude embeddings)
scaling = model.num_scaling_params()
print()
print(f"{'Scaling params (excl. embeddings)':.<50s} {scaling:>15,}")

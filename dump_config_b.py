#!/usr/bin/env python3
"""Print all named_parameters shapes for Config B (32L × 3072, no VE)."""
import torch
import nanochat.gpt as gpt_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
V = tokenizer.get_vocab_size()
print(f"vocab_size: {V}")

gpt_module.has_ve = lambda layer_idx, n_layer: False
config = GPTConfig(
    sequence_len=2048, vocab_size=V,
    n_layer=32, n_head=24, n_kv_head=24, n_embd=3072,
    window_pattern="SSSL",
)
with torch.device("meta"):
    model = GPT(config)

print(f"\nConfig B: 32L × 3072, no VE")
print(f"{'=' * 80}")
print(f"{'Name':<55s} {'Shape':<20s} {'Params':>12s}")
print(f"{'-' * 80}")

total = 0
prev_prefix = ""
for name, p in model.named_parameters():
    # Group separator between top-level modules
    prefix = name.split(".")[0]
    if prefix != prev_prefix and prev_prefix:
        print()
    prev_prefix = prefix

    shape_str = str(list(p.shape))
    n = p.numel()
    total += n

    # For blocks, only print block 0 and 31 in full, summarize others
    if name.startswith("transformer.h."):
        parts = name.split(".")
        block_idx = int(parts[2])
        rest = ".".join(parts[3:])
        if block_idx == 0:
            print(f"  {name:<53s} {shape_str:<20s} {n:>12,}")
        elif block_idx == 1 and rest == "attn.c_q.weight":
            print(f"  ... (blocks 1–30 identical to block 0)")
        elif block_idx == 31:
            print(f"  {name:<53s} {shape_str:<20s} {n:>12,}")
    else:
        print(f"  {name:<53s} {shape_str:<20s} {n:>12,}")

print(f"{'-' * 80}")
print(f"  {'TOTAL':<53s} {'':20s} {total:>12,}  ({total/1e9:.3f}B)")

# Verify no VE tables
n_ve = len(model.value_embeds)
print(f"\n  Value embed tables: {n_ve} (should be 0)")

# Sanity checks
print(f"\n  Sanity checks:")
padded_V = ((V + 63) // 64) * 64
print(f"    Padded vocab: {padded_V}")
print(f"    wte shape: {list(model.transformer.wte.weight.shape)} (expect [{padded_V}, 3072])")
print(f"    lm_head shape: {list(model.lm_head.weight.shape)} (expect [3072, {padded_V}])")
# Note: nn.Linear weight is [out_features, in_features]
print(f"    lm_head.weight is [{model.lm_head.weight.shape[0]}, {model.lm_head.weight.shape[1]}]"
      f" = Linear(in={config.n_embd}, out={padded_V})")
print(f"    MLP expansion: c_fc is [{model.transformer.h[0].mlp.c_fc.weight.shape[0]}, "
      f"{model.transformer.h[0].mlp.c_fc.weight.shape[1]}] = {model.transformer.h[0].mlp.c_fc.weight.shape[0] / config.n_embd:.0f}× expansion")
print(f"    All blocks same size: {len(set(sum(p.numel() for p in b.parameters()) for b in model.transformer.h)) == 1}")

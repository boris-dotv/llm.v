#!/usr/bin/env python3
"""
Smoke test for the rewritten architecture (Track A).
Builds models at various configs, runs forward on random input, prints loss and shapes.

Run on GPU server:
    python scripts/smoke_test_arch.py
"""
import sys
import math
import torch
from nanochat.gpt import GPT, GPTConfig

def smoke_test(config_kwargs, label=""):
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {label}")
    print(f"{'='*60}")
    config = GPTConfig(**config_kwargs)
    print(f"Config: {config_kwargs}")

    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Print state_dict key names
    print(f"State dict keys ({len(model.state_dict())} total):")
    for k, v in sorted(model.state_dict().items()):
        print(f"  {k:60s} {str(list(v.shape)):>20s}")

    # Forward pass
    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        loss = model(input_ids, targets)

    logits = model(input_ids)
    expected_loss = math.log(config.vocab_size)
    print(f"Loss: {loss.item():.4f}")
    print(f"Expected loss ~ ln({config.vocab_size}) = {expected_loss:.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits std: {logits.std().item():.4f}")
    print(f"PASS")
    return total_params


if __name__ == "__main__":
    # Tiny test
    smoke_test(
        dict(n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
             intermediate_size=256, vocab_size=1024, sequence_len=64),
        label="Tiny (hidden=128, layers=2, heads=2, intermediate=256)"
    )

    # A7 verification: tie_word_embeddings
    params_tied = smoke_test(
        dict(n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
             intermediate_size=256, vocab_size=1024, sequence_len=64,
             tie_word_embeddings=True),
        label="Tiny + tied embeddings"
    )
    params_untied = smoke_test(
        dict(n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
             intermediate_size=256, vocab_size=1024, sequence_len=64,
             tie_word_embeddings=False),
        label="Tiny + untied embeddings"
    )
    param_diff = params_untied - params_tied
    expected_diff = 1024 * 128  # vocab * hidden
    print(f"\nTied vs untied param diff: {param_diff:,} (expected: {expected_diff:,})")
    assert param_diff == expected_diff, f"Mismatch! {param_diff} != {expected_diff}"
    print("tie_word_embeddings: VERIFIED")

    # 150M dry-run model
    print("\n" + "="*60)
    print("150M DENSE-ONLY DRY RUN MODEL")
    print("="*60)
    params_150m = smoke_test(
        dict(n_embd=768, n_layer=12, n_head=6, n_kv_head=6,
             intermediate_size=2048, vocab_size=65536, sequence_len=2048,
             mixer_types=["minicpm4"] * 12, tie_word_embeddings=True,
             scale_emb=8.0, scale_depth=1.4, dim_model_base=256),
        label="150M dense (hidden=768, layers=12, heads=6, kv=6, intermediate=2048, tie=True)"
    )
    print(f"\n150M model total params: {params_150m:,}")
    print(f"Target: ~150M. {'OK' if 130_000_000 < params_150m < 170_000_000 else 'CHECK!'}")

    # Save summary
    with open("/tmp/arch_smoke_150M.log", "w") as f:
        f.write(f"150M dense-only model params: {params_150m:,}\n")
        f.write(f"Architecture smoke test: PASS\n")
    print(f"\nSaved to /tmp/arch_smoke_150M.log")

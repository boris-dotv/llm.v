#!/usr/bin/env python3
"""
Smoke test for the rewritten architecture (Track A).
Builds models at various configs, runs forward on random input.
Verifies loss is in [ln(V), 2*ln(V)] and exits 1 on failure.

Usage (on GPU server):
    PYTHONPATH=. python scripts/smoke_test_arch.py
"""
import sys
import math
import torch
from nanochat.gpt import GPT, GPTConfig

failures = []


def smoke_test(config_kwargs, label="", check_loss=True):
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {label}")
    print(f"{'='*60}")
    config = GPTConfig(**config_kwargs)

    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    # Patch rotary to fp32 for CPU
    model.cos = model.cos.float()
    model.sin = model.sin.float()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Forward pass
    B, T = 2, 32
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        loss = model(input_ids, targets)

    logits = model(input_ids)
    loss_val = loss.item()
    expected = math.log(config.vocab_size)
    lo, hi = expected, 2 * expected

    print(f"Loss:     {loss_val:.4f}")
    print(f"Expected: ln({config.vocab_size}) = {expected:.4f}")
    print(f"Range:    [{lo:.2f}, {hi:.2f}]")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits std:   {logits.std().item():.4f}")

    if check_loss:
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"FAIL: loss is {loss_val}")
            failures.append(label)
        elif loss_val < lo or loss_val > hi:
            print(f"FAIL: loss {loss_val:.4f} outside [{lo:.2f}, {hi:.2f}]")
            failures.append(label)
        else:
            print(f"PASS")
    else:
        print(f"(loss check skipped)")

    return total_params


if __name__ == "__main__":
    # --- Tiny untied (default) ---
    smoke_test(
        dict(n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
             intermediate_size=256, vocab_size=1024, sequence_len=64),
        label="Tiny untied (hidden=128, layers=2, vocab=1024)"
    )

    # --- A7 param count verification: tied vs untied ---
    params_tied = smoke_test(
        dict(n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
             intermediate_size=256, vocab_size=1024, sequence_len=64,
             tie_word_embeddings=True),
        label="Tiny tied (A7 param count check)",
        check_loss=False,  # tied has known high loss at init, not the point here
    )
    params_untied = smoke_test(
        dict(n_embd=128, n_layer=2, n_head=2, n_kv_head=2,
             intermediate_size=256, vocab_size=1024, sequence_len=64,
             tie_word_embeddings=False),
        label="Tiny untied (A7 param count check)",
    )
    param_diff = params_untied - params_tied
    expected_diff = 1024 * 128  # vocab * hidden
    print(f"\nTied vs untied param diff: {param_diff:,} (expected: {expected_diff:,})")
    if param_diff != expected_diff:
        print(f"FAIL: param diff mismatch {param_diff} != {expected_diff}")
        failures.append("A7 param diff")
    else:
        print("tie_word_embeddings param count: VERIFIED")

    # --- 150M dense-only dry run model (UNTIED, matching SALA upstream) ---
    print("\n" + "=" * 60)
    print("150M DENSE-ONLY DRY RUN MODEL (untied)")
    print("=" * 60)
    params_150m = smoke_test(
        dict(n_embd=768, n_layer=12, n_head=6, n_kv_head=6,
             intermediate_size=2048, vocab_size=65536, sequence_len=2048,
             mixer_types=["minicpm4"] * 12, tie_word_embeddings=False,
             scale_emb=8.0, scale_depth=1.4, dim_model_base=256),
        label="150M dense untied (hidden=768, layers=12, heads=6, kv=6, intermediate=2048)"
    )
    print(f"\n150M model total params: {params_150m:,}")

    # --- Final verdict ---
    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} failure(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)

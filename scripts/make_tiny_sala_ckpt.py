#!/usr/bin/env python3
"""
Generate a tiny randomly-initialized MiniCPM-SALA checkpoint in HF format
for smoke-testing sglang_sala_llmv inference.

Usage:
    python scripts/make_tiny_sala_ckpt.py

Output: /tmp/tiny_sala_ckpt/ with config.json, model.safetensors, tokenizer files.
"""

import json
import os
import shutil

import torch
from safetensors.torch import save_file


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config — every field that sglang_sala_llmv/sglang/srt/models/minicpm.py
#    reads via config.X or getattr(config, "X", default).
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN = 512
N_LAYERS = 4
N_HEADS = 8           # sparse/minicpm4 query heads
N_KV_HEADS = 2        # sparse/minicpm4 KV heads
SPARSE_HEAD_DIM = HIDDEN // N_HEADS   # 64
INTER = 1408          # SwiGLU intermediate size
VOCAB = 32768

# Lightning attention params (can differ from sparse)
LIGHTNING_NH = 8
LIGHTNING_NKV = 2
LIGHTNING_HD = 128    # independent head_dim for lightning layers

MIXER_TYPES = ["minicpm4", "lightning", "lightning", "lightning"]

config = {
    # ── Base model fields ──
    "hidden_size": HIDDEN,
    "num_hidden_layers": N_LAYERS,
    "num_attention_heads": N_HEADS,
    "num_key_value_heads": N_KV_HEADS,
    "head_dim": LIGHTNING_HD,             # metadata; sparse uses hidden/n_heads
    "intermediate_size": INTER,
    "vocab_size": VOCAB,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rope_scaling": None,                 # getattr default
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,

    # ── MiniCPM scaling fields (MiniCPMModel.forward, MiniCPMForCausalLM.__init__) ──
    "scale_emb": 12.0,                    # multiplied after embed_tokens
    "scale_depth": 1.4,                   # residual scaling: * scale_depth/sqrt(n_layers)
    "dim_model_base": 256,                # pre-logit: hidden / dim_model_base

    # ── Hybrid / mixer fields (MiniCPMDecoderLayer.__init__) ──
    "mixer_types": MIXER_TYPES,

    # ── Sparse/minicpm4 attention fields ──
    "attn_use_rope": True,                # hasattr check, default True
    "attn_use_output_gate": True,         # hasattr check, default False

    # ── Lightning attention fields (MiniCPMLightningMixer.__init__) ──
    "lightning_nh": LIGHTNING_NH,
    "lightning_nkv": LIGHTNING_NKV,
    "lightning_head_dim": LIGHTNING_HD,
    "lightning_use_rope": True,
    "lightning_scale": "1/sqrt(d)",
    "use_output_gate": True,              # lightning z_proj
    "use_output_norm": False,             # lightning o_norm
    "qk_norm": True,                      # lightning q_norm / k_norm
    "attention_bias": False,              # lightning bias

    # ── Architecture registration ──
    "architectures": ["MiniCPMSALAForCausalLM"],
    "model_type": "minicpm_sala",
    "auto_map": {
        "AutoConfig": "configuration_minicpm_sala.MiniCPMSALAConfig",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Weight generation — HF naming with separate q/k/v and gate/up
#    (load_weights uses stacked_params_mapping to merge them)
# ─────────────────────────────────────────────────────────────────────────────

def randn_weight(*shape, std=0.02):
    return (torch.randn(*shape) * std).to(torch.bfloat16)

def ones_weight(*shape):
    return torch.ones(*shape, dtype=torch.bfloat16)


weights = {}
total_params = 0

def add(name, tensor):
    global total_params
    weights[name] = tensor
    total_params += tensor.numel()


# ── Global ──
add("model.embed_tokens.weight", randn_weight(VOCAB, HIDDEN))
add("model.norm.weight", ones_weight(HIDDEN))
# No lm_head.weight because tie_word_embeddings=True

# ── Per-layer ──
for i in range(N_LAYERS):
    prefix = f"model.layers.{i}"
    mixer = MIXER_TYPES[i]

    # -- LayerNorm (same for all layer types) --
    add(f"{prefix}.input_layernorm.weight", ones_weight(HIDDEN))
    add(f"{prefix}.post_attention_layernorm.weight", ones_weight(HIDDEN))

    # -- MLP (same for all layer types: SwiGLU) --
    # load_weights maps gate_proj->gate_up_proj[0], up_proj->gate_up_proj[1]
    add(f"{prefix}.mlp.gate_proj.weight", randn_weight(INTER, HIDDEN))
    add(f"{prefix}.mlp.up_proj.weight", randn_weight(INTER, HIDDEN))
    add(f"{prefix}.mlp.down_proj.weight", randn_weight(HIDDEN, INTER))

    if mixer == "minicpm4":
        # -- Sparse/dense attention --
        # QKV: separate, load_weights merges via stacked_params_mapping
        # q: (n_heads * head_dim, hidden) = (8*64, 512) = (512, 512)
        # k: (n_kv_heads * head_dim, hidden) = (2*64, 512) = (128, 512)
        # v: same as k
        q_dim = N_HEADS * SPARSE_HEAD_DIM
        kv_dim = N_KV_HEADS * SPARSE_HEAD_DIM
        add(f"{prefix}.self_attn.q_proj.weight", randn_weight(q_dim, HIDDEN))
        add(f"{prefix}.self_attn.k_proj.weight", randn_weight(kv_dim, HIDDEN))
        add(f"{prefix}.self_attn.v_proj.weight", randn_weight(kv_dim, HIDDEN))
        # o_proj: (hidden, n_heads * head_dim) = (512, 512)
        add(f"{prefix}.self_attn.o_proj.weight", randn_weight(HIDDEN, q_dim))
        # o_gate (attn_use_output_gate=True)
        add(f"{prefix}.self_attn.o_gate.weight", randn_weight(q_dim, HIDDEN))

    elif mixer in ["lightning", "lightning_attn", "lightning-attn"]:
        # -- Lightning attention --
        # QKV with lightning dimensions (can differ from sparse)
        # q: (lightning_nh * lightning_hd, hidden) = (8*128, 512) = (1024, 512)
        # k: (lightning_nkv * lightning_hd, hidden) = (2*128, 512) = (256, 512)
        # v: same as k
        q_dim = LIGHTNING_NH * LIGHTNING_HD
        kv_dim = LIGHTNING_NKV * LIGHTNING_HD
        add(f"{prefix}.self_attn.q_proj.weight", randn_weight(q_dim, HIDDEN))
        add(f"{prefix}.self_attn.k_proj.weight", randn_weight(kv_dim, HIDDEN))
        add(f"{prefix}.self_attn.v_proj.weight", randn_weight(kv_dim, HIDDEN))
        # o_proj: (hidden, lightning_nh * lightning_hd) = (512, 1024)
        add(f"{prefix}.self_attn.o_proj.weight", randn_weight(HIDDEN, q_dim))
        # z_proj (use_output_gate=True for lightning)
        add(f"{prefix}.self_attn.z_proj.weight", randn_weight(q_dim, HIDDEN))
        # QK norm (qk_norm=True): per-head RMSNorm
        add(f"{prefix}.self_attn.q_norm.weight", ones_weight(LIGHTNING_HD))
        add(f"{prefix}.self_attn.k_norm.weight", ones_weight(LIGHTNING_HD))
        # o_norm is NOT created because use_output_norm=False

    print(f"  Layer {i} ({mixer}): done")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save checkpoint
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = "/tmp/tiny_sala_ckpt"
os.makedirs(OUT_DIR, exist_ok=True)

# config.json
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"\nSaved config.json to {config_path}")

# model.safetensors
safetensors_path = os.path.join(OUT_DIR, "model.safetensors")
save_file(weights, safetensors_path)
print(f"Saved model.safetensors to {safetensors_path}")

# Minimal tokenizer files — create a dummy GPT2-style tokenizer
# that has vocab_size=32768 using the tokenizers library
tokenizer_config = {
    "model_type": "minicpm_sala",
    "tokenizer_class": "PreTrainedTokenizerFast",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}
special_tokens_map = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}

# Build a minimal tokenizer.json using the tokenizers library
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    # Add vocab: 4 special tokens + fill to VOCAB with dummy tokens
    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    vocab = {s: i for i, s in enumerate(specials)}
    for i in range(len(specials), VOCAB):
        vocab[f"token_{i}"] = i
    tok.model = BPE(vocab=vocab, merges=[], unk_token="<unk>")
    tok.save(os.path.join(OUT_DIR, "tokenizer.json"))
    print(f"Saved tokenizer.json (dummy, {VOCAB} tokens)")
except ImportError:
    print("WARNING: tokenizers library not available, skipping tokenizer.json")
    print("  You may need to copy a tokenizer from an existing model")

with open(os.path.join(OUT_DIR, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=2)
with open(os.path.join(OUT_DIR, "special_tokens_map.json"), "w") as f:
    json.dump(special_tokens_map, f, indent=2)
print(f"Saved tokenizer_config.json, special_tokens_map.json")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"CHECKPOINT SUMMARY")
print(f"{'='*60}")
print(f"Output dir: {OUT_DIR}")
print(f"Total parameters: {total_params:,}")
print(f"Number of weight tensors: {len(weights)}")
print(f"Dtype: bfloat16")
print(f"tie_word_embeddings: True (no lm_head.weight)")
print()

# Per-layer breakdown
for i in range(N_LAYERS):
    prefix = f"model.layers.{i}"
    layer_params = sum(t.numel() for k, t in weights.items() if k.startswith(prefix))
    mixer = MIXER_TYPES[i]
    print(f"  Layer {i} ({mixer:>10s}): {layer_params:>10,} params")

global_params = sum(t.numel() for k, t in weights.items() if not k.startswith("model.layers."))
print(f"  {'Global':>18s}: {global_params:>10,} params")
print()

# List all weight names
print("All weight names:")
for name in sorted(weights.keys()):
    shape = list(weights[name].shape)
    print(f"  {name:60s} {str(shape):>20s}")

# Verify completeness: check no unexpected fields
expected_names = set()
expected_names.add("model.embed_tokens.weight")
expected_names.add("model.norm.weight")
for i in range(N_LAYERS):
    p = f"model.layers.{i}"
    expected_names.add(f"{p}.input_layernorm.weight")
    expected_names.add(f"{p}.post_attention_layernorm.weight")
    expected_names.add(f"{p}.mlp.gate_proj.weight")
    expected_names.add(f"{p}.mlp.up_proj.weight")
    expected_names.add(f"{p}.mlp.down_proj.weight")
    expected_names.add(f"{p}.self_attn.q_proj.weight")
    expected_names.add(f"{p}.self_attn.k_proj.weight")
    expected_names.add(f"{p}.self_attn.v_proj.weight")
    expected_names.add(f"{p}.self_attn.o_proj.weight")
    mixer = MIXER_TYPES[i]
    if mixer == "minicpm4":
        expected_names.add(f"{p}.self_attn.o_gate.weight")
    else:
        expected_names.add(f"{p}.self_attn.z_proj.weight")
        expected_names.add(f"{p}.self_attn.q_norm.weight")
        expected_names.add(f"{p}.self_attn.k_norm.weight")

actual_names = set(weights.keys())
missing = expected_names - actual_names
extra = actual_names - expected_names
if missing:
    print(f"\nMISSING weights: {missing}")
if extra:
    print(f"\nEXTRA weights: {extra}")
if not missing and not extra:
    print(f"\nAll {len(expected_names)} expected weight names present. No extras.")

print(f"\nFiles in {OUT_DIR}:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f:40s} {size:>10,} bytes")

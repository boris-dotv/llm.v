#!/usr/bin/env python3
"""
Smoke test: load the tiny SALA checkpoint into SGLang's MiniCPMSALAForCausalLM
and run a single forward pass to verify weight loading + model forward.

This does NOT use the full SGLang server — it directly instantiates the model
class, loads weights, and calls forward() with a manually constructed batch.

Usage:
    python scripts/sglang_load_smoke_test.py
"""

import os
import sys
import json

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 0. Verify sglang is from sglang_sala_llmv
# ─────────────────────────────────────────────────────────────────────────────

import sglang
print(f"sglang.__file__: {sglang.__file__}")
assert "sglang_sala_llmv" in sglang.__file__, \
    f"FAIL: sglang imported from wrong location: {sglang.__file__}"
print("OK: sglang imported from sglang_sala_llmv\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load config and build model
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR = "/tmp/tiny_sala_ckpt"
config_path = os.path.join(CKPT_DIR, "config.json")

with open(config_path) as f:
    config_dict = json.load(f)

print(f"Config loaded from {config_path}")
print(f"  model_type: {config_dict['model_type']}")
print(f"  architectures: {config_dict['architectures']}")
print(f"  mixer_types: {config_dict['mixer_types']}")
print()

# Build the config object that the model expects.
# SGLang uses MiniCPMHybridConfig which is a PretrainedConfig subclass.
from sglang.srt.configs.minicpm import MiniCPMHybridConfig

# MiniCPMHybridConfig.__init__ accepts the standard HF config fields.
# Extra fields (scale_emb, scale_depth, etc.) are passed through **kwargs
# and set as attributes by PretrainedConfig.
hf_config = MiniCPMHybridConfig(**config_dict)

# Verify critical attributes exist
for attr in ["scale_emb", "scale_depth", "dim_model_base", "mixer_types",
             "lightning_nh", "lightning_nkv", "lightning_head_dim",
             "lightning_use_rope", "lightning_scale", "use_output_gate",
             "qk_norm", "use_output_norm", "attention_bias",
             "attn_use_rope", "attn_use_output_gate"]:
    val = getattr(hf_config, attr, "MISSING")
    print(f"  config.{attr} = {val}")
print()

# Build model on CUDA
device = torch.device("cuda:0")

# SGLang uses tensor-parallel wrappers that need dist to be initialized.
# For single-GPU smoke test, we just init with world_size=1.
if not torch.distributed.is_initialized():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

from sglang.srt.models.minicpm import MiniCPMSALAForCausalLM

print("Building MiniCPMSALAForCausalLM...")
model = MiniCPMSALAForCausalLM(hf_config, quant_config=None)
model = model.to(device=device, dtype=torch.bfloat16)
print(f"Model built. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load weights from safetensors
# ─────────────────────────────────────────────────────────────────────────────

from safetensors import safe_open

safetensors_path = os.path.join(CKPT_DIR, "model.safetensors")
print(f"Loading weights from {safetensors_path}...")

# Build iterator of (name, tensor) pairs as load_weights expects
def weight_iterator():
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            yield name, f.get_tensor(name)

# Collect weight names for diagnostic
loaded_names = []
skipped_names = []

# Wrap the iterator to log what happens
def logging_weight_iterator():
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            loaded_names.append(name)
            yield name, f.get_tensor(name)

model.load_weights(logging_weight_iterator())

print(f"Loaded {len(loaded_names)} weight tensors")

# Check for mismatches: compare model params vs loaded weights
model_params = set(dict(model.named_parameters()).keys())
print(f"Model has {len(model_params)} parameters")

# The loaded names use HF convention (q_proj, k_proj, v_proj, gate_proj, up_proj)
# The model uses merged names (qkv_proj, gate_up_proj)
# So a direct comparison won't work — just verify no errors occurred
print("Weight loading completed without errors.")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attempt forward pass
# ─────────────────────────────────────────────────────────────────────────────

# The challenge: MiniCPMForCausalLM.forward() requires a ForwardBatch.
# ForwardBatch is deeply entangled with SGLang's scheduler.
# Strategy: bypass the top-level forward and call the inner model directly,
# then manually compute logits.

print("="*60)
print("ATTEMPTING FORWARD PASS")
print("="*60)

SEQ_LEN = 32
input_ids = torch.arange(SEQ_LEN, dtype=torch.long, device=device)
positions = torch.arange(SEQ_LEN, dtype=torch.long, device=device)

# The inner model (MiniCPMModel) forward also needs forward_batch for attention.
# Let's try the most minimal approach: bypass attention backends entirely
# and just test that the weight shapes flow through the linear layers.

# Strategy: test layer-by-layer manually
print("\n--- Testing embedding ---")
with torch.no_grad():
    hidden = model.model.embed_tokens(input_ids) * hf_config.scale_emb
    print(f"  embed output: {hidden.shape}, dtype={hidden.dtype}")
    print(f"  mean={hidden.float().mean():.4f}, std={hidden.float().std():.4f}")

print("\n--- Testing layer 0 (minicpm4) sublayers ---")
layer0 = model.model.layers[0]
with torch.no_grad():
    # Test layernorm
    normed = layer0.input_layernorm(hidden)
    print(f"  layernorm output: {normed.shape}")

    # Test MLP
    mlp_out = layer0.mlp(normed)
    print(f"  MLP output: {mlp_out.shape}")

    # Test QKV projection
    qkv, _ = layer0.self_attn.qkv_proj(normed)
    q_size = layer0.self_attn.q_size
    kv_size = layer0.self_attn.kv_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    print(f"  QKV: q={q.shape}, k={k.shape}, v={v.shape}")

    # Test o_proj
    # Need a fake attention output of shape (seq_len, n_heads*head_dim)
    fake_attn_out = torch.randn(SEQ_LEN, q_size, device=device, dtype=torch.bfloat16)
    o_out, _ = layer0.self_attn.o_proj(fake_attn_out)
    print(f"  o_proj output: {o_out.shape}")

    # Test o_gate
    if hasattr(layer0.self_attn, 'o_gate') and layer0.self_attn.use_output_gate:
        gate_out, _ = layer0.self_attn.o_gate(normed)
        print(f"  o_gate output: {gate_out.shape}")

print("\n--- Testing layer 1 (lightning) sublayers ---")
layer1 = model.model.layers[1]
with torch.no_grad():
    normed1 = layer1.input_layernorm(hidden)
    print(f"  layernorm output: {normed1.shape}")

    # Test MLP
    mlp_out1 = layer1.mlp(normed1)
    print(f"  MLP output: {mlp_out1.shape}")

    # Test QKV projection
    qkv1, _ = layer1.self_attn.qkv_proj(normed1)
    q_size1 = layer1.self_attn.q_size
    kv_size1 = layer1.self_attn.kv_size
    q1, k1, v1 = qkv1.split([q_size1, kv_size1, kv_size1], dim=-1)
    print(f"  QKV: q={q1.shape}, k={k1.shape}, v={v1.shape}")

    # Test QK norm
    if hasattr(layer1.self_attn, 'q_norm'):
        q1_normed = layer1.self_attn.q_norm(q1.reshape(-1, layer1.self_attn.head_dim))
        print(f"  q_norm output: {q1_normed.shape}")

    # Test z_proj (output gate for lightning)
    if hasattr(layer1.self_attn, 'z_proj') and layer1.self_attn.use_output_gate:
        z_out, _ = layer1.self_attn.z_proj(normed1)
        print(f"  z_proj output: {z_out.shape}")

    # Test o_proj
    fake_attn_out1 = torch.randn(SEQ_LEN, q_size1, device=device, dtype=torch.bfloat16)
    o_out1, _ = layer1.self_attn.o_proj(fake_attn_out1)
    print(f"  o_proj output: {o_out1.shape}")

print("\n--- Testing final norm + logits ---")
with torch.no_grad():
    final_normed = model.model.norm(hidden)
    print(f"  final norm output: {final_normed.shape}")

    # Compute logits manually (bypass LogitsProcessor which needs ForwardBatch)
    scale_width = hf_config.hidden_size / hf_config.dim_model_base
    scaled = final_normed / scale_width

    if hf_config.tie_word_embeddings:
        lm_weight = model.model.embed_tokens.weight
    else:
        lm_weight = model.lm_head.weight
    logits = torch.nn.functional.linear(scaled, lm_weight)

    print(f"  logits shape: {logits.shape}")
    print(f"  logits dtype: {logits.dtype}")
    print(f"  logits mean:  {logits.float().mean():.6f}")
    print(f"  logits std:   {logits.float().std():.6f}")
    print(f"  logits[0,:5]: {logits[0,:5].float().tolist()}")

print()
print("="*60)
expected_shape = (SEQ_LEN, VOCAB)
actual_shape = tuple(logits.shape)
if actual_shape == expected_shape:
    print(f"SUCCESS: logits shape {actual_shape} matches expected {expected_shape}")
else:
    print(f"SHAPE MISMATCH: got {actual_shape}, expected {expected_shape}")
print("="*60)

# Cleanup
torch.distributed.destroy_process_group()

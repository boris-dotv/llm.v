"""
HALO Conversion: Convert a dense pretrained model into a SALA hybrid model.

Determines which layers to convert to linear attention (1:3 ratio),
copies weights from the dense model, and saves the converted checkpoint.

Usage:
    python -m scripts.halo_convert --model-tag d32 --model-step 71680 \
        --sparse-layers 0,9,16,17,22,29,30,31 \
        --n-kv-head-sparse 2 --use-output-gate --use-hype
"""

import os
import json
import argparse
import torch

from nanochat.common import get_base_dir, print0, print_banner
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint, find_last_step, find_largest_model
from nanochat.linear_attention import LightningAttention

print_banner()

parser = argparse.ArgumentParser(description="HALO: Convert dense model to SALA hybrid")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g. d32)")
parser.add_argument("--model-step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("--sparse-layers", type=str, default="0,9,16,17,22,29,30,31",
                    help="Comma-separated layer indices to keep as sparse/dense attention")
parser.add_argument("--n-kv-head-sparse", type=int, default=2,
                    help="Number of KV heads for sparse layers (GQA)")
parser.add_argument("--use-output-gate", action="store_true", help="Add output gates to all layers")
parser.add_argument("--use-hype", action="store_true", help="HyPE: RoPE on linear, NoPE on sparse")
parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE base frequency")
args = parser.parse_args()

# ---- Resolve checkpoint ----
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
if args.model_tag is None:
    args.model_tag = find_largest_model(checkpoints_dir)
    print0(f"Auto-detected model tag: {args.model_tag}")
checkpoint_dir = os.path.join(checkpoints_dir, args.model_tag)
if args.model_step is None:
    args.model_step = find_last_step(checkpoint_dir)
print0(f"Loading dense checkpoint: {checkpoint_dir}/model_{args.model_step:06d}.pt")

# ---- Load dense checkpoint ----
device = torch.device("cpu")  # conversion on CPU to avoid GPU memory issues
model_data, _, meta_data = load_checkpoint(checkpoint_dir, args.model_step, device)
# Fix torch.compile prefix
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
dense_config_kwargs = meta_data["model_config"]

# ---- Determine layer types ----
sparse_layer_indices = set(int(x) for x in args.sparse_layers.split(","))
n_layer = dense_config_kwargs["n_layer"]
attn_types = []
for i in range(n_layer):
    attn_types.append("dense" if i in sparse_layer_indices else "linear")
n_sparse = sum(1 for t in attn_types if t == "dense")
n_linear = sum(1 for t in attn_types if t == "linear")
print0(f"Layer layout: {n_sparse} sparse + {n_linear} linear = {n_layer} total")
print0(f"Sparse layers: {sorted(sparse_layer_indices)}")

# ---- Build hybrid config ----
hybrid_config_kwargs = dict(dense_config_kwargs)
hybrid_config_kwargs["attn_types"] = attn_types
hybrid_config_kwargs["n_kv_head_sparse"] = args.n_kv_head_sparse
hybrid_config_kwargs["use_output_gate"] = args.use_output_gate
hybrid_config_kwargs["use_hype"] = args.use_hype
hybrid_config_kwargs["rope_theta"] = args.rope_theta
hybrid_config = GPTConfig(**hybrid_config_kwargs)

# ---- Build hybrid model ----
print0("Building hybrid model on meta device...")
with torch.device("meta"):
    hybrid_model = GPT(hybrid_config)
hybrid_model.to_empty(device=device)
hybrid_model.init_weights()

# ---- Build dense model to copy weights from ----
print0("Building dense model to extract weights...")
from nanochat.checkpoint_manager import _patch_missing_config_keys, _patch_missing_keys
_patch_missing_config_keys(dense_config_kwargs)
dense_config = GPTConfig(**dense_config_kwargs)
_patch_missing_keys(model_data, dense_config)
with torch.device("meta"):
    dense_model = GPT(dense_config)
dense_model.to_empty(device=device)
dense_model.init_weights()
dense_model.load_state_dict(model_data, strict=True, assign=True)

# ---- Copy weights ----
print0("Copying weights from dense to hybrid model...")

# Non-layer params: wte, lm_head, resid_lambdas, x0_lambdas
hybrid_model.transformer.wte.weight.data.copy_(dense_model.transformer.wte.weight.data)
hybrid_model.lm_head.weight.data.copy_(dense_model.lm_head.weight.data)
hybrid_model.resid_lambdas.data.copy_(dense_model.resid_lambdas.data)
hybrid_model.x0_lambdas.data.copy_(dense_model.x0_lambdas.data)

# Value embeddings (may have different dimensions for linear vs sparse layers)
for key in hybrid_model.value_embeds:
    if key in dense_model.value_embeds:
        dense_ve = dense_model.value_embeds[key]
        hybrid_ve = hybrid_model.value_embeds[key]
        if dense_ve.weight.shape == hybrid_ve.weight.shape:
            hybrid_ve.weight.data.copy_(dense_ve.weight.data)
        else:
            # Linear layers have full MHA, so expand VE from kv_dim to n_embd
            layer_idx = int(key)
            dense_kv_heads = dense_config.n_kv_head
            head_dim = dense_config.n_embd // dense_config.n_head
            n_head = dense_config.n_head
            repeat_factor = n_head // dense_kv_heads
            ve_w = dense_ve.weight.data.view(-1, dense_kv_heads, head_dim)
            hybrid_ve.weight.data.copy_(ve_w.repeat(1, repeat_factor, 1).view(-1, n_head * head_dim))
            print0(f"  Expanded value embedding for layer {key}: {dense_ve.weight.shape} -> {hybrid_ve.weight.shape}")

# Per-layer weights
for i in range(n_layer):
    dense_block = dense_model.transformer.h[i]
    hybrid_block = hybrid_model.transformer.h[i]
    dense_attn = dense_block.attn
    hybrid_attn = hybrid_block.attn

    # MLP: always identical
    hybrid_block.mlp.c_fc.weight.data.copy_(dense_block.mlp.c_fc.weight.data)
    hybrid_block.mlp.c_proj.weight.data.copy_(dense_block.mlp.c_proj.weight.data)

    if attn_types[i] == "linear":
        # Linear layer: use init_from_dense to copy and expand weights
        hybrid_attn.init_from_dense(dense_attn)
        # Output gate (z_proj) stays at zero init
        print0(f"  Layer {i}: converted to linear attention (MHA, {hybrid_attn.n_head} heads)")
    else:
        # Sparse layer: copy Q/K/V/O directly
        hybrid_attn.c_q.weight.data.copy_(dense_attn.c_q.weight.data)
        # K/V may change shape if n_kv_head_sparse != n_kv_head
        if dense_attn.c_k.weight.shape == hybrid_attn.c_k.weight.shape:
            hybrid_attn.c_k.weight.data.copy_(dense_attn.c_k.weight.data)
            hybrid_attn.c_v.weight.data.copy_(dense_attn.c_v.weight.data)
        else:
            # Compress KV from n_kv_head to n_kv_head_sparse by averaging groups
            dense_kv = dense_config.n_kv_head
            sparse_kv = args.n_kv_head_sparse
            head_dim = dense_config.n_embd // dense_config.n_head
            group_size = dense_kv // sparse_kv
            k_w = dense_attn.c_k.weight.data.view(dense_kv, head_dim, -1)
            hybrid_attn.c_k.weight.data.copy_(
                k_w.view(sparse_kv, group_size, head_dim, -1).mean(dim=1).view(-1, k_w.shape[2])
            )
            v_w = dense_attn.c_v.weight.data.view(dense_kv, head_dim, -1)
            hybrid_attn.c_v.weight.data.copy_(
                v_w.view(sparse_kv, group_size, head_dim, -1).mean(dim=1).view(-1, v_w.shape[2])
            )
            print0(f"  Layer {i}: compressed KV heads {dense_kv} -> {sparse_kv}")
        hybrid_attn.c_proj.weight.data.copy_(dense_attn.c_proj.weight.data)
        # ve_gate: copy if shapes match
        if hybrid_attn.ve_gate is not None and dense_attn.ve_gate is not None:
            if dense_attn.ve_gate.weight.shape == hybrid_attn.ve_gate.weight.shape:
                hybrid_attn.ve_gate.weight.data.copy_(dense_attn.ve_gate.weight.data)
            else:
                # Compress ve_gate from n_kv_head to n_kv_head_sparse
                dense_kv = dense_config.n_kv_head
                sparse_kv = args.n_kv_head_sparse
                group_size = dense_kv // sparse_kv
                vg_w = dense_attn.ve_gate.weight.data.view(dense_kv, -1)
                hybrid_attn.ve_gate.weight.data.copy_(
                    vg_w.view(sparse_kv, group_size, -1).mean(dim=1)
                )
        # Output gate (o_gate) stays at zero init
        print0(f"  Layer {i}: kept as sparse attention (GQA, {hybrid_attn.n_kv_head} KV heads)")

# ---- Save converted checkpoint ----
output_dir = os.path.join(base_dir, "halo_checkpoints", args.model_tag)
os.makedirs(output_dir, exist_ok=True)
hybrid_state = hybrid_model.state_dict()
# Remove non-persistent buffers (rotary embeddings, decay slopes) — they're recreated in init_weights
hybrid_state = {k: v for k, v in hybrid_state.items() if "cos" not in k and "sin" not in k and "g_gamma" not in k}

meta = {
    "model_config": hybrid_config_kwargs,
    "user_config": {
        "source": "halo_convert",
        "dense_checkpoint": f"{checkpoint_dir}/model_{args.model_step:06d}.pt",
        "sparse_layers": sorted(sparse_layer_indices),
    },
    "step": 0,
}
save_checkpoint(output_dir, 0, hybrid_state, None, meta, rank=0)
print0(f"Saved HALO-converted checkpoint to: {output_dir}")
print0(f"Hybrid model: {sum(p.numel() for p in hybrid_model.parameters()):,} parameters")

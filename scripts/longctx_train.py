"""
Long-Context Adaptation: Progressive context extension with InfLLM-V2 sparse enabled.

Extends context from 4K to 32K/64K/128K with sparse attention on sparse layers
and SimpleGLA on linear layers. Includes gradient checkpointing for memory.

Usage:
    torchrun --nproc_per_node=8 -m scripts.longctx_train -- \
        --source continual --model-tag d32 \
        --phase 32k --max-seq-len 32768 --device-batch-size 2 \
        --rope-theta 100000 --gradient-checkpointing \
        --num-tokens 10000000000
"""

import os
import argparse
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, mixed_tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.loss_eval import evaluate_bpb

from contextlib import nullcontext

print_banner()

parser = argparse.ArgumentParser(description="Long-context adaptation for SALA hybrid model")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--model-step", type=int, default=None)
parser.add_argument("--source", type=str, default="continual", help="continual|longctx")
parser.add_argument("--phase", type=str, default="32k", help="32k|64k|128k — for naming and LR defaults")
parser.add_argument("--max-seq-len", type=int, default=32768)
parser.add_argument("--device-batch-size", type=int, default=2)
parser.add_argument("--total-batch-size", type=int, default=7_864_320)
parser.add_argument("--num-tokens", type=int, default=10_000_000_000)
parser.add_argument("--embedding-lr", type=float, default=3e-4)
parser.add_argument("--matrix-lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--rope-theta", type=float, default=100000.0, help="RoPE base for long context")
parser.add_argument("--gradient-checkpointing", action="store_true")
parser.add_argument("--warmup-steps", type=int, default=500)
parser.add_argument("--final-lr-frac", type=float, default=0.5, help="Final LR as fraction of initial")
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--run", type=str, default="dummy")
# Code data (for code-specialized training)
parser.add_argument("--code-data-dir", type=str, default=None, help="path to code data parquets")
parser.add_argument("--code-weight", type=float, default=0.7, help="fraction of tokens from code corpus")
parser.add_argument("--fim-rate", type=float, default=0.5, help="fraction of code docs to apply FIM")
parser.add_argument("--spm-rate", type=float, default=0.5, help="fraction of FIM docs using SPM format")
args = parser.parse_args()

# ---- Setup ----
device_type = args.device_type or autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# ---- Load model ----
print0(f"Loading {args.source} checkpoint...")
model, tokenizer, meta_data = load_model(args.source, device, phase="train",
                                          model_tag=args.model_tag, step=args.model_step)
orig_model = model

# Update config for long context
model.config.rope_theta = args.rope_theta
model.config.use_gradient_checkpointing = args.gradient_checkpointing
model.config.sequence_len = args.max_seq_len
# Recompute rotary embeddings with new theta and longer sequence
head_dim = model.config.n_embd // model.config.n_head
model.rotary_seq_len = args.max_seq_len * 2  # 2x safety margin
cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim)
model.cos, model.sin = cos, sin
# Recompute window sizes for new sequence length
model.window_sizes = model._compute_window_sizes(model.config)
for i in range(model.config.n_layer):
    if model.attn_types[i] == "linear":
        model.window_sizes[i] = (-1, 0)

# Enable sparse attention (InfLLM-V2 block selection on sparse layers)
model.disable_sparse = False
print0(f"Sparse attention ENABLED")
print0(f"RoPE theta: {args.rope_theta}")
print0(f"Gradient checkpointing: {args.gradient_checkpointing}")
print0(f"Sequence length: {args.max_seq_len}")

# All parameters trainable
for param in model.parameters():
    param.requires_grad = True

# ---- Optimizer ----
optimizers = model.setup_optimizers(
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# ---- Training setup ----
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
if args.total_batch_size % world_tokens_per_fwdbwd != 0:
    grad_accum_steps = max(1, grad_accum_steps)
    actual_batch = grad_accum_steps * world_tokens_per_fwdbwd
    print0(f"Adjusted total_batch_size from {args.total_batch_size} to {actual_batch}")
    args.total_batch_size = actual_batch
num_iterations = args.num_tokens // args.total_batch_size

print0(f"Phase: {args.phase}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Total iterations: {num_iterations:,}")
print0(f"Total tokens: {num_iterations * args.total_batch_size:,}")

# ---- Data ----
if args.code_data_dir:
    from nanochat.fim import make_fim_transform
    from nanochat.dataset import DATA_DIR
    fim_fn = make_fim_transform(tokenizer, fim_rate=args.fim_rate, spm_rate=args.spm_rate)
    data_sources = [
        (args.code_data_dir, args.code_weight, fim_fn),
        (DATA_DIR, 1.0 - args.code_weight, None),
    ]
    print0(f"Mixed data: {args.code_weight:.0%} code (FIM rate={args.fim_rate}) + {1-args.code_weight:.0%} text")
    train_loader = mixed_tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        data_sources=data_sources, device=device,
    )
else:
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        "train", tokenizer, args.device_batch_size, args.max_seq_len, ddp_rank, ddp_world_size, device
    )
x, y, _ = next(train_loader)

# ---- Compile ----
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

# ---- Training loop ----
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

print0(f"Starting long-context training ({args.phase})...")
for step in range(num_iterations):
    t0 = time.time()

    # LR: warmup then linear decay to final_lr_frac
    if step < args.warmup_steps:
        lrm = (step + 1) / args.warmup_steps
    else:
        progress = (step - args.warmup_steps) / max(1, num_iterations - args.warmup_steps)
        lrm = 1.0 - (1.0 - args.final_lr_frac) * progress
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Forward + backward
    model.train()
    for opt in optimizers:
        opt.zero_grad()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, _ = next(train_loader)
    for opt in optimizers:
        opt.step()

    dt = time.time() - t0
    tok_per_sec = args.total_batch_size / dt

    if step % 10 == 0:
        print0(f"step {step:06d}/{num_iterations} | loss: {train_loss.item():.4f} | lrm: {lrm:.4f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,.0f}")

    # Save checkpoint
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "longctx_checkpoints", args.model_tag or "default")
        config_to_save = dict(meta_data["model_config"])
        config_to_save["rope_theta"] = args.rope_theta
        config_to_save["sequence_len"] = args.max_seq_len
        save_checkpoint(output_dir, step, orig_model.state_dict(), None,
                       {"model_config": config_to_save, "step": step}, rank=ddp_rank)

# ---- Final save ----
base_dir = get_base_dir()
output_dir = os.path.join(base_dir, "longctx_checkpoints", args.model_tag or "default")
config_to_save = dict(meta_data["model_config"])
config_to_save["rope_theta"] = args.rope_theta
config_to_save["sequence_len"] = args.max_seq_len
save_checkpoint(output_dir, step + 1, orig_model.state_dict(), None,
               {"model_config": config_to_save, "step": step + 1,
                "user_config": {"source": "longctx_train", "phase": args.phase, "num_tokens": args.num_tokens}},
               rank=ddp_rank)
print0(f"Saved long-context checkpoint to: {output_dir}")

compute_cleanup()

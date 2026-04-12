"""
Continual Stable-Training: Train all parameters of the hybrid model with
sparse attention disabled. Linear layers use SimpleGLA, sparse layers use
standard dense FlashAttention with the existing window pattern.

Usage:
    torchrun --nproc_per_node=8 -m scripts.continual_train -- \
        --model-tag d32 --max-seq-len 4096 --num-tokens 30000000000
"""

import os
import argparse
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.loss_eval import evaluate_bpb

from contextlib import nullcontext

print_banner()

parser = argparse.ArgumentParser(description="Continual stable-training for SALA hybrid model")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--model-step", type=int, default=None)
parser.add_argument("--source", type=str, default="halo", help="Checkpoint source: halo|continual")
parser.add_argument("--max-seq-len", type=int, default=4096)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--total-batch-size", type=int, default=7_864_320, help="~7.8M tokens")
parser.add_argument("--num-tokens", type=int, default=30_000_000_000)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--warmup-steps", type=int, default=2000)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--save-every", type=int, default=-1, help="-1 = save only at end")
parser.add_argument("--run", type=str, default="dummy")
args = parser.parse_args()

# ---- Setup ----
device_type = args.device_type or autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# ---- Load model ----
print0(f"Loading {args.source} checkpoint...")
model, tokenizer, meta_data = load_model(args.source, device, phase="train",
                                          model_tag=args.model_tag, step=args.model_step)
orig_model = model  # keep reference for saving

# All parameters trainable (unfreeze everything)
for param in model.parameters():
    param.requires_grad = True

# Disable sparse attention (sparse layers use standard dense FlashAttention)
model.disable_sparse = True
print0("Sparse attention DISABLED (sparse layers use dense FlashAttention)")

# ---- Optimizer ----
optimizers = model.setup_optimizers(
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# ---- Training setup ----
max_seq_len = args.max_seq_len
tokens_per_fwdbwd = args.device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
num_iterations = args.num_tokens // args.total_batch_size

print0(f"Sequence length: {max_seq_len}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Total iterations: {num_iterations:,}")
print0(f"Total tokens: {num_iterations * args.total_batch_size:,}")

# ---- Data ----
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    "train", tokenizer, args.device_batch_size, max_seq_len, ddp_rank, ddp_world_size, device
)
x, y, _ = next(train_loader)

# ---- Compile ----
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

# ---- Training loop ----
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

print0("Starting continual stable-training...")
for step in range(num_iterations):
    t0 = time.time()

    # LR: warmup then constant
    if step < args.warmup_steps:
        lrm = (step + 1) / args.warmup_steps
    else:
        lrm = 1.0
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
        output_dir = os.path.join(base_dir, "cont_checkpoints", args.model_tag or "default")
        save_checkpoint(output_dir, step, orig_model.state_dict(), None,
                       {"model_config": meta_data["model_config"], "step": step}, rank=ddp_rank)

# ---- Final save ----
base_dir = get_base_dir()
output_dir = os.path.join(base_dir, "cont_checkpoints", args.model_tag or "default")
save_checkpoint(output_dir, step + 1, orig_model.state_dict(), None,
               {"model_config": meta_data["model_config"], "step": step + 1,
                "user_config": {"source": "continual_train", "num_tokens": args.num_tokens}},
               rank=ddp_rank)
print0(f"Saved continual training checkpoint to: {output_dir}")

compute_cleanup()

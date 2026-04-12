"""
HALO Fine-tune: Train only linear attention layers after HALO conversion.

Freezes everything except LightningAttention modules and trains on
pretraining data for a short period (1.3B tokens at 512 context).

Usage:
    torchrun --nproc_per_node=8 -m scripts.halo_train -- --model-tag d32 --max-seq-len 512 --num-tokens 1300000000
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
from nanochat.linear_attention import LightningAttention
from contextlib import nullcontext

print_banner()

parser = argparse.ArgumentParser(description="HALO fine-tune: train linear attention layers")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--model-step", type=int, default=None)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--num-tokens", type=int, default=1_300_000_000, help="Total tokens to train on")
parser.add_argument("--lr", type=float, default=7.5e-3, help="Learning rate")
parser.add_argument("--warmup-steps", type=int, default=2000)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--run", type=str, default="dummy")
args = parser.parse_args()

# ---- Setup ----
device_type = args.device_type or autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# ---- Load HALO-converted model ----
print0("Loading HALO-converted model...")
model, tokenizer, meta_data = load_model("halo", device, phase="train",
                                          model_tag=args.model_tag, step=args.model_step)

# ---- Freeze everything except linear attention ----
n_frozen = 0
n_trainable = 0
for name, param in model.named_parameters():
    is_linear_attn = False
    # Check if this parameter belongs to a LightningAttention module
    for block_idx, block in enumerate(model.transformer.h):
        if isinstance(block.attn, LightningAttention):
            for pname, p in block.attn.named_parameters():
                if p is param:
                    is_linear_attn = True
                    break
        if is_linear_attn:
            break
    param.requires_grad = is_linear_attn
    if is_linear_attn:
        n_trainable += param.numel()
    else:
        n_frozen += param.numel()

print0(f"Frozen: {n_frozen:,} params | Trainable: {n_trainable:,} params")

# Disable sparse attention (sparse layers use standard dense FlashAttention)
model.disable_sparse = True

# ---- DDP wrapping ----
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
orig_model = model.module if ddp else model

# ---- Optimizer (AdamW only, since we're only training linear attention) ----
from functools import partial
from nanochat.adamw import DistAdamW
trainable_params = [p for p in model.parameters() if p.requires_grad]
AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
optimizer = AdamWFactory([dict(params=trainable_params, lr=args.lr)], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# ---- Training setup ----
max_seq_len = args.max_seq_len
tokens_per_fwdbwd = args.device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
num_iterations = args.num_tokens // args.total_batch_size

print0(f"Sequence length: {max_seq_len}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Total iterations: {num_iterations}")
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

print0("Starting HALO fine-tuning...")
for step in range(num_iterations):
    t0 = time.time()

    # LR warmup
    if step < args.warmup_steps:
        lr_mult = (step + 1) / args.warmup_steps
    else:
        lr_mult = 1.0
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_mult

    # Forward + backward
    model.train()
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, _ = next(train_loader)

    optimizer.step()

    dt = time.time() - t0
    tok_per_sec = args.total_batch_size / dt
    if step % 10 == 0:
        print0(f"step {step:05d}/{num_iterations} | loss: {train_loss.item():.4f} | lr: {lr_mult:.4f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,.0f}")

# ---- Save ----
base_dir = get_base_dir()
output_dir = os.path.join(base_dir, "halo_checkpoints", args.model_tag or "default")
orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
save_checkpoint(
    output_dir, step + 1,
    orig_model.state_dict(),
    None,
    {"model_config": meta_data["model_config"], "step": step + 1,
     "user_config": {"source": "halo_train", "num_tokens": args.num_tokens}},
    rank=ddp_rank,
)
print0(f"Saved HALO fine-tuned checkpoint to: {output_dir}")

compute_cleanup()

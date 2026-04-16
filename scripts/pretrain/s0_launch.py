"""
S0 dense pretraining launcher (single-node, multi-GPU).

Builds the 608M dry-run model from dryrun_608m.py and trains it on the
HF streaming data mix defined in data_s0.py.

Usage:
    # Single GPU:
    python -m scripts.pretrain.s0_launch --run s0-608m-v1

    # Multi-GPU (8x H100), Muon+AdamW (default, uses DDP):
    torchrun --nproc_per_node=8 -m scripts.pretrain.s0_launch --run s0-608m-v1

    # Multi-GPU, pure AdamW (uses FSDP2):
    torchrun --nproc_per_node=8 -m scripts.pretrain.s0_launch --run s0-608m-v1 --optimizer adamw

    # Smoke test (CPU, random data, 100 steps):
    python -m scripts.pretrain.s0_launch --smoke --run dummy

    # With placeholder tokenizer:
    python -m scripts.pretrain.s0_launch --smoke --run dummy --allow-placeholder-tokenizer
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import glob
import time
import argparse
from contextlib import nullcontext

import torch
import wandb

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    autodetect_device_type, get_peak_flops,
)
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step
from scripts.configs.dryrun_608m import get_model_config
from scripts.pretrain.data_s0 import (
    get_s0_tokenizer, build_data_iterator, smoke_data_iterator,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="S0 Dense Pretraining (608M)")
# Training horizon
parser.add_argument("--total-tokens", type=int, default=10_000_000_000,
                    help="Total training tokens (default: 10B for dry-run)")
# Output
parser.add_argument("--out-dir", type=str, default="out/s0_608m",
                    help="Checkpoint output directory")
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb)")
# Model
parser.add_argument("--seq-len", type=int, default=2048,
                    help="Sequence length (default: 2048 for dry-run)")
# Optimization
parser.add_argument("--optimizer", type=str, choices=["muon_adamw", "adamw"],
                    default="muon_adamw",
                    help="Optimizer: muon_adamw (default) or adamw")
parser.add_argument("--global-batch-size", type=int, default=512,
                    help="Global batch size in sequences")
parser.add_argument("--lr-peak", type=float, default=3e-4,
                    help="Peak learning rate (used for --optimizer adamw)")
parser.add_argument("--warmup-steps", type=int, default=2000,
                    help="Linear warmup steps")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in --out-dir")
parser.add_argument("--smoke", action="store_true",
                    help="Smoke test: 100 steps on random data")
parser.add_argument("--allow-placeholder-tokenizer", action="store_true",
                    help="Allow Qwen2.5-0.5B fallback tokenizer")
# Checkpointing
parser.add_argument("--save-every", type=int, default=2000,
                    help="Save checkpoint every N steps")
parser.add_argument("--keep-last", type=int, default=3,
                    help="Number of recent checkpoints to keep")
# Logging
parser.add_argument("--log-every", type=int, default=10,
                    help="Log metrics every N steps")
args = parser.parse_args()
user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Compute init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                if device_type == "cuda" else nullcontext())
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tokenizer, vocab_size, is_placeholder = get_s0_tokenizer(args.allow_placeholder_tokenizer)
tokenizer_path = "Qwen/Qwen2.5-0.5B" if is_placeholder else "nanochat"
print0(f"Tokenizer: {tokenizer_path} | vocab_size: {vocab_size:,}")
if is_placeholder:
    print0("Placeholder tokenizer active — checkpoints NOT transferable.")

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
# The tokenizer tells the config what vocab_size to use (single source of truth)
model_config_kwargs = get_model_config(vocab_size=vocab_size)
model_config_kwargs["sequence_len"] = args.seq_len

print0(f"Model config: {model_config_kwargs}")

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model parameters: {num_params:,}")

# ---------------------------------------------------------------------------
# Distributed wrapping
# ---------------------------------------------------------------------------
use_fsdp2 = ddp and args.optimizer == "adamw"
use_ddp = ddp and args.optimizer == "muon_adamw"

if use_fsdp2:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    # Per-block sharding for memory efficiency
    for block in model.transformer.h:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    print0(f"FSDP2 enabled (world_size={ddp_world_size})")
elif use_ddp:
    # DDP — used with Muon+AdamW (DistMuon handles dist comms internally)
    print0(f"DDP mode (world_size={ddp_world_size})")

# Store original (uncompiled) model for checkpointing
orig_model = model
if not use_fsdp2:
    # torch.compile with FSDP2 is handled differently; skip for now
    model = torch.compile(model, dynamic=False)

# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------
if args.optimizer == "muon_adamw":
    optimizers = model.setup_optimizers()
    print0("Optimizer: Muon + AdamW (nanochat defaults)")
else:
    # Pure AdamW
    # Separate params into decayed and non-decayed
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr_peak,
        betas=(0.9, 0.95),
    )
    optimizers = [optimizer]
    print0(f"Optimizer: AdamW (lr={args.lr_peak}, betas=(0.9,0.95), wd=0.1)")

# Record initial LRs
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# ---------------------------------------------------------------------------
# Batch size and gradient accumulation
# ---------------------------------------------------------------------------
global_batch_tokens = args.global_batch_size * args.seq_len

# Auto micro-batch tuning (try largest first, fall back on OOM)
micro_batch_candidates = [16, 8, 4, 2, 1]
if args.smoke or device_type != "cuda":
    # For smoke test or CPU, just use 2; also shrink global batch for smoke
    device_batch_size = min(2, args.global_batch_size)
    if args.smoke:
        args.global_batch_size = device_batch_size
        global_batch_tokens = args.global_batch_size * args.seq_len
else:
    device_batch_size = None
    for mbs in micro_batch_candidates:
        try:
            dummy_x = torch.randint(0, vocab_size, (mbs, args.seq_len), device=device)
            dummy_y = torch.randint(0, vocab_size, (mbs, args.seq_len), device=device)
            with autocast_ctx:
                loss = model(dummy_x, dummy_y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            device_batch_size = mbs
            print0(f"Auto micro-batch: {mbs} fits in memory")
            break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print0(f"Auto micro-batch: {mbs} OOM, trying smaller")
    if device_batch_size is None:
        print("FATAL: Cannot fit even micro-batch=1 in GPU memory", file=sys.stderr)
        sys.exit(1)

tokens_per_micro = device_batch_size * args.seq_len * ddp_world_size
# Adjust global batch size to be divisible by tokens_per_micro
if global_batch_tokens % tokens_per_micro != 0:
    old_gbs = args.global_batch_size
    args.global_batch_size = ((global_batch_tokens // tokens_per_micro) * tokens_per_micro) // args.seq_len
    global_batch_tokens = args.global_batch_size * args.seq_len
    print0(f"Adjusted global_batch_size {old_gbs} -> {args.global_batch_size} for clean division")

grad_accum_steps = global_batch_tokens // tokens_per_micro
print0(f"Micro-batch: {device_batch_size} | Grad accum steps: {grad_accum_steps}")
print0(f"Global batch: {args.global_batch_size} seqs = {global_batch_tokens:,} tokens/step")

# ---------------------------------------------------------------------------
# Training iterations
# ---------------------------------------------------------------------------
if args.smoke:
    num_iterations = 100
    print0("Smoke test mode: 100 iterations on random data")
else:
    num_iterations = args.total_tokens // global_batch_tokens
    print0(f"Training for {num_iterations:,} iterations ({args.total_tokens:,} tokens)")

# ---------------------------------------------------------------------------
# Data iterator
# ---------------------------------------------------------------------------
if args.smoke:
    data_iter = smoke_data_iterator(
        vocab_size, device_batch_size, args.seq_len,
        num_batches=num_iterations * grad_accum_steps + 10,
    )
else:
    data_iter = build_data_iterator(
        tokenizer, device_batch_size, args.seq_len,
        seed=42 + ddp_rank,
    )

# ---------------------------------------------------------------------------
# LR schedule: WSD with warmup only (S0 = no decay)
# ---------------------------------------------------------------------------
def get_lr_multiplier(step):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    return 1.0  # constant after warmup — no decay in S0


# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------
use_wandb = args.run != "dummy" and master_process
wandb_run = (
    wandb.init(project="llm-v-s0", name=args.run, config=user_config)
    if use_wandb else DummyWandb()
)

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
start_step = 0
best_loss = float("inf")
if args.resume and os.path.exists(args.out_dir):
    try:
        resume_step = find_last_step(args.out_dir)
        model_data, optimizer_data, meta_data = load_checkpoint(
            args.out_dir, resume_step, device, load_optimizer=True, rank=ddp_rank)
        # Verify tokenizer compatibility
        saved_tokenizer = meta_data.get("tokenizer_path", "unknown")
        if saved_tokenizer != tokenizer_path:
            print0(f"WARNING: checkpoint tokenizer '{saved_tokenizer}' != current '{tokenizer_path}'")
            print0("This may cause incorrect training. Proceeding anyway.")
        orig_model.load_state_dict(model_data, strict=True, assign=True)
        for opt, opt_state in zip(optimizers, optimizer_data):
            opt.load_state_dict(opt_state)
        start_step = meta_data["step"] + 1
        best_loss = meta_data.get("best_loss", float("inf"))
        print0(f"Resumed from step {resume_step}, continuing at step {start_step}")
        del model_data, optimizer_data
    except FileNotFoundError:
        print0("No checkpoint found in --out-dir, starting from scratch")

# ---------------------------------------------------------------------------
# Checkpoint rotation utilities
# ---------------------------------------------------------------------------
def _rotate_checkpoints(checkpoint_dir, keep_last):
    """Delete old checkpoints, keeping only the last `keep_last`."""
    if ddp_rank != 0:
        return
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pt")))
    if len(model_files) <= keep_last:
        return
    for old_model in model_files[:-keep_last]:
        step_str = os.path.basename(old_model).split("_")[-1].split(".")[0]
        # Don't delete "best" checkpoint
        best_meta = os.path.join(checkpoint_dir, "best_step.txt")
        if os.path.exists(best_meta):
            with open(best_meta) as f:
                best_step_str = f.read().strip()
            if step_str == best_step_str:
                continue
        for pattern in [f"model_{step_str}.pt",
                        f"meta_{step_str}.json",
                        f"optim_{step_str}_rank*.pt"]:
            for f in glob.glob(os.path.join(checkpoint_dir, pattern)):
                os.remove(f)


def _save_best_copy(checkpoint_dir, step):
    """Copy current checkpoint as the best-loss checkpoint."""
    if ddp_rank != 0:
        return
    step_str = f"{step:06d}"
    # Write a marker so rotation doesn't delete this step
    with open(os.path.join(checkpoint_dir, "best_step.txt"), "w") as f:
        f.write(step_str)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nStarting training: step {start_step} -> {num_iterations}")
print0(f"Checkpoint dir: {args.out_dir}")
print0(f"Checkpoint every {args.save_every} steps, keep last {args.keep_last}")
print0("")

model.train()
smooth_loss = 0.0
total_training_time = 0.0

for step in range(start_step, num_iterations):
    synchronize()
    t0 = time.time()

    # --- Forward + backward with gradient accumulation ---
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        batch = next(data_iter)
        x = batch["input_ids"].to(device=device, non_blocking=True)
        y = batch["labels"].to(device=device, non_blocking=True)
        with autocast_ctx:
            loss = model(x, y)
        total_loss += loss.detach().item()
        loss = loss / grad_accum_steps
        loss.backward()

    avg_loss = total_loss / grad_accum_steps

    # --- Gradient clipping (adamw only) ---
    grad_norm = None
    if args.optimizer == "adamw":
        if use_fsdp2:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

    # --- LR update ---
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # --- Optimizer step ---
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # --- Logging ---
    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * avg_loss
    debiased_loss = smooth_loss / (1 - ema_beta ** (step - start_step + 1))

    if step > start_step + 5:
        total_training_time += dt

    tok_per_sec = int(global_batch_tokens / dt) if dt > 0 else 0
    flops_per_sec = (orig_model.estimate_flops() * global_batch_tokens / dt
                     if dt > 0 else 0)
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    if step % args.log_every == 0:
        # Console
        tokens_so_far = step * global_batch_tokens
        pct = 100 * step / num_iterations if num_iterations > 0 else 0
        grad_str = f" | grad_norm: {grad_norm:.4f}" if grad_norm is not None else ""
        print0(
            f"step {step:06d}/{num_iterations:06d} ({pct:.1f}%) | "
            f"loss: {debiased_loss:.4f}{grad_str} | "
            f"lr: {lrm * (args.lr_peak if args.optimizer == 'adamw' else 0.02):.6f} | "
            f"dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}%"
        )
        # Wandb
        log_data = {
            "step": step,
            "tokens_seen": tokens_so_far,
            "train/loss": debiased_loss,
            "train/raw_loss": avg_loss,
            "train/lr_multiplier": lrm,
            "train/tokens_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/dt_ms": dt * 1000,
        }
        if grad_norm is not None:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # --- NaN/Inf check ---
    if not torch.isfinite(torch.tensor(avg_loss)):
        print0(f"FATAL: loss is {avg_loss} at step {step}. Aborting.")
        break

    # --- Checkpointing ---
    if (step > 0 and step % args.save_every == 0) or step == num_iterations - 1:
        meta_data = {
            "step": step,
            "model_config": model_config_kwargs,
            "tokenizer_path": tokenizer_path,
            "vocab_size": vocab_size,
            "is_placeholder_tokenizer": is_placeholder,
            "optimizer": args.optimizer,
            "total_tokens_target": args.total_tokens,
            "tokens_seen": step * global_batch_tokens,
            "loss": avg_loss,
            "best_loss": best_loss,
            "user_config": user_config,
        }
        save_checkpoint(
            args.out_dir, step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            meta_data,
            rank=ddp_rank,
        )
        _rotate_checkpoints(args.out_dir, args.keep_last)

        # Best-loss tracking
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_best_copy(args.out_dir, step)
            print0(f"New best loss: {best_loss:.4f} at step {step}")

# ---------------------------------------------------------------------------
# Final stats
# ---------------------------------------------------------------------------
print0(f"\nTraining complete.")
print0(f"Total training time: {total_training_time/60:.1f} min")
print0(f"Peak memory: {get_max_memory() / 1024**2:.0f} MiB")
print0(f"Best loss: {best_loss:.4f}")
print0(f"Checkpoints in: {args.out_dir}")

wandb_run.finish()
compute_cleanup()

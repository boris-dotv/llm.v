"""
Lightweight eval harness for S0 checkpoints.

Evaluates:
  1. Perplexity: WikiText-103 test, held-out FineWeb-Edu slice
  2. Zero-shot accuracy: HellaSwag, PIQA, ARC-Easy (via CORE benchmark)
  3. HumanEval pass@1 (code generation baseline)

Usage:
    # Evaluate latest checkpoint:
    python -m scripts.pretrain.eval_s0 --checkpoint-dir out/s0_608m

    # Evaluate specific step:
    python -m scripts.pretrain.eval_s0 --checkpoint-dir out/s0_608m --step 4000

    # Skip slow evals:
    python -m scripts.pretrain.eval_s0 --checkpoint-dir out/s0_608m --skip-humaneval
"""

import os
import json
import math
import argparse
from contextlib import nullcontext

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import (
    compute_init, compute_cleanup, print0,
    autodetect_device_type,
)
from nanochat.checkpoint_manager import (
    load_checkpoint, find_last_step, _patch_missing_config_keys,
)
from scripts.pretrain.data_s0 import get_s0_tokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="S0 Eval Harness")
parser.add_argument("--checkpoint-dir", type=str, required=True,
                    help="Path to checkpoint directory (e.g. out/s0_608m)")
parser.add_argument("--step", type=int, default=None,
                    help="Checkpoint step to evaluate (default: latest)")
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--skip-humaneval", action="store_true",
                    help="Skip HumanEval pass@1 (slow + needs code execution)")
parser.add_argument("--skip-core", action="store_true",
                    help="Skip CORE benchmark (HellaSwag/PIQA/ARC-Easy)")
parser.add_argument("--ppl-max-tokens", type=int, default=500_000,
                    help="Max tokens for perplexity evaluation")
parser.add_argument("--fineweb-heldout-tokens", type=int, default=10_000_000,
                    help="Tokens for FineWeb-Edu held-out perplexity")
parser.add_argument("--allow-placeholder-tokenizer", action="store_true",
                    help="Allow Qwen2.5 fallback tokenizer for eval")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                if device_type == "cuda" else nullcontext())

# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------
step = args.step if args.step is not None else find_last_step(args.checkpoint_dir)
print0(f"Loading checkpoint from {args.checkpoint_dir} step {step}")

model_data, _, meta_data = load_checkpoint(
    args.checkpoint_dir, step, device, load_optimizer=False, rank=ddp_rank)

# Build model from saved config
model_config_kwargs = meta_data["model_config"]
_patch_missing_config_keys(model_config_kwargs)
model_config = GPTConfig(**model_config_kwargs)

with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# Strip torch.compile prefix if present
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
if device_type in {"cpu", "mps"}:
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v
                  for k, v in model_data.items()}
model.load_state_dict(model_data, strict=True, assign=True)
del model_data
model.eval()

# Load tokenizer (must match what was used during training)
saved_tokenizer_path = meta_data.get("tokenizer_path", "nanochat")
is_placeholder = saved_tokenizer_path != "nanochat"
tokenizer, vocab_size, _ = get_s0_tokenizer(
    allow_placeholder=args.allow_placeholder_tokenizer or is_placeholder)
seq_len = model_config.sequence_len

print0(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
print0(f"Tokenizer: {saved_tokenizer_path} (vocab={vocab_size:,})")
print0(f"Sequence length: {seq_len}")

# ---------------------------------------------------------------------------
# 1. Perplexity: WikiText-103
# ---------------------------------------------------------------------------
results = {}


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, texts, seq_len, max_tokens, device, label):
    """Compute perplexity = exp(avg cross-entropy loss) over chunked text."""
    print0(f"Evaluating perplexity on {label}...", end=" ")

    # Tokenize all text into one long sequence
    bos_id = tokenizer.get_bos_token_id()
    all_ids = []
    tokens_collected = 0
    for text in texts:
        if tokens_collected >= max_tokens:
            break
        ids = tokenizer.encode(text, prepend=bos_id)
        all_ids.extend(ids)
        tokens_collected = len(all_ids)

    if len(all_ids) < seq_len + 1:
        print0(f"SKIP (only {len(all_ids)} tokens, need >= {seq_len + 1})")
        return None

    # Chunk into non-overlapping blocks of seq_len
    num_chunks = (len(all_ids) - 1) // seq_len
    total_loss = 0.0
    total_tokens = 0

    for i in range(num_chunks):
        start = i * seq_len
        chunk = all_ids[start:start + seq_len + 1]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        with autocast_ctx:
            loss = model(x, y)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss)
    print0(f"loss={avg_loss:.4f}, PPL={ppl:.2f} ({total_tokens:,} tokens)")
    return ppl


# WikiText-103
try:
    from datasets import load_dataset
    wt_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    wt_texts = [row["text"] for row in wt_ds if row["text"].strip()]
    wt_ppl = evaluate_perplexity(model, tokenizer, wt_texts, seq_len,
                                  args.ppl_max_tokens, device, "WikiText-103")
    results["perplexity_wikitext103"] = wt_ppl
except Exception as e:
    print0(f"WikiText-103 eval failed: {e}")
    results["perplexity_wikitext103"] = None

# FineWeb-Edu held-out
try:
    from datasets import load_dataset
    # Use a slice that's unlikely to overlap with training data
    # (streaming means we can skip ahead)
    fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                         streaming=True, trust_remote_code=True)
    fw_texts = []
    fw_tokens_approx = 0
    # Skip first 1M examples to get "held-out" data
    for i, example in enumerate(fw_ds):
        if i < 1_000_000:
            continue
        fw_texts.append(example["text"])
        fw_tokens_approx += len(example["text"]) // 4  # rough char->token ratio
        if fw_tokens_approx >= args.fineweb_heldout_tokens:
            break

    fw_ppl = evaluate_perplexity(model, tokenizer, fw_texts, seq_len,
                                  args.fineweb_heldout_tokens, device,
                                  "FineWeb-Edu held-out")
    results["perplexity_fineweb_heldout"] = fw_ppl
except Exception as e:
    print0(f"FineWeb-Edu held-out eval failed: {e}")
    results["perplexity_fineweb_heldout"] = None

# ---------------------------------------------------------------------------
# 2. Zero-shot accuracy: HellaSwag, PIQA, ARC-Easy (via CORE)
# ---------------------------------------------------------------------------
if not args.skip_core:
    try:
        from scripts.base_eval import evaluate_model
        print0("Running CORE benchmark (HellaSwag, PIQA, ARC-Easy, ...)...")
        with autocast_ctx:
            core_results = evaluate_model(model, tokenizer, device, max_per_task=500)

        results["core_metric"] = core_results["core_metric"]
        results["zero_shot"] = {}
        # Extract individual task results
        for task_name, accuracy in core_results["results"].items():
            results["zero_shot"][task_name] = accuracy
            # Highlight the three we care about
            task_lower = task_name.lower()
            if "hellaswag" in task_lower:
                results["hellaswag"] = accuracy
            elif "piqa" in task_lower:
                results["piqa"] = accuracy
            elif "arc" in task_lower and "easy" in task_lower:
                results["arc_easy"] = accuracy

        print0(f"CORE metric: {core_results['core_metric']:.4f}")
    except Exception as e:
        print0(f"CORE benchmark failed: {e}")
        results["core_metric"] = None
else:
    print0("Skipping CORE benchmark (--skip-core)")

# ---------------------------------------------------------------------------
# 3. HumanEval pass@1
# ---------------------------------------------------------------------------
if not args.skip_humaneval:
    try:
        from datasets import load_dataset
        from nanochat.engine import Engine

        print0("Running HumanEval pass@1...")
        he_ds = load_dataset("openai/openai_humaneval", split="test")

        engine = Engine(model, tokenizer)
        passed = 0
        total = 0

        for example in he_ds:
            prompt = example["prompt"]
            tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())

            # Generate completion
            with autocast_ctx:
                try:
                    completions, _ = engine.generate_batch(
                        tokens, num_samples=1, max_tokens=256, temperature=0)
                    completion = tokenizer.decode(completions[0])
                except Exception:
                    completion = ""

            # Extract function body (stop at first unindented line or "def")
            lines = completion.split("\n")
            body_lines = []
            for line in lines:
                if line.strip().startswith("def ") and body_lines:
                    break
                body_lines.append(line)
            full_code = prompt + "\n".join(body_lines)

            # Test execution
            test_code = full_code + "\n" + example["test"] + "\n" + f"check({example['entry_point']})"
            try:
                exec_globals = {}
                exec(test_code, exec_globals)
                passed += 1
            except Exception:
                pass
            total += 1

        pass_rate = passed / total if total > 0 else 0.0
        results["humaneval_pass1"] = pass_rate
        print0(f"HumanEval pass@1: {pass_rate:.4f} ({passed}/{total})")
    except Exception as e:
        print0(f"HumanEval eval failed: {e}")
        results["humaneval_pass1"] = None
else:
    print0("Skipping HumanEval (--skip-humaneval)")

# ---------------------------------------------------------------------------
# Output eval.json
# ---------------------------------------------------------------------------
results["checkpoint_dir"] = args.checkpoint_dir
results["step"] = step
results["model_params"] = sum(p.numel() for p in model.parameters())
results["tokenizer_path"] = saved_tokenizer_path
results["vocab_size"] = vocab_size

output_path = os.path.join(args.checkpoint_dir, "eval.json")
if ddp_rank == 0:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print0(f"\nResults written to {output_path}")

# Print summary
print0("\n" + "=" * 60)
print0("EVAL SUMMARY")
print0("=" * 60)
for k, v in results.items():
    if isinstance(v, float):
        print0(f"  {k}: {v:.4f}")
    elif isinstance(v, dict):
        print0(f"  {k}:")
        for k2, v2 in v.items():
            print0(f"    {k2}: {v2:.4f}" if isinstance(v2, float) else f"    {k2}: {v2}")
    else:
        print0(f"  {k}: {v}")

compute_cleanup()

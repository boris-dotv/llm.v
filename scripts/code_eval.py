"""
Evaluate the Code model on HumanEval and MBPP benchmarks.

Example runs:
python -m scripts.code_eval -i sft
python -m scripts.code_eval -i rl --model-tag d12 --max-problems 50
torchrun --nproc_per_node=8 -m scripts.code_eval -- -i rl
"""

import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mbpp import MBPP

# -----------------------------------------------------------------------------
# Generative evaluation loop for code tasks

def run_code_eval(task_object, tokenizer, model, engine, num_samples, max_tokens, temperature, top_k, max_problems=None):
    """
    Evaluate a code task and return pass@1.
    For each problem we generate num_samples completions; the problem passes if any of them
    is correct (i.e. standard pass@k with k=num_samples, reported as pass@1).
    In a distributed setting all ranks cooperate; results are aggregated before returning.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None or max_problems < 0 else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt up to (but not including) the assistant completion
        encoded_prompt = tokenizer.render_for_completion(conversation)
        prefix_length = len(encoded_prompt)

        # Generate completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode completions and evaluate
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)

        # Overwrite progress line in place
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    return num_passed / total

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the code model on HumanEval and MBPP")
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: mid|sft|rl")
    parser.add_argument('--model-tag', type=str, default=None, help="Model tag to load")
    parser.add_argument('--max-problems', type=int, default=-1, help="Max problems to evaluate (-1 = all)")
    parser.add_argument('--num-samples', type=int, default=1, help="Samples per problem for pass@k")
    parser.add_argument('--max-tokens', type=int, default=512, help="Max tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature (0.0 = greedy)")
    parser.add_argument('--top-k', type=int, default=50, help="Top-k sampling (0 = disabled)")
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps', ''],
                        help="Device type: cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)

    max_problems = None if args.max_problems < 0 else args.max_problems

    # Shared kwargs for the eval loop
    eval_kwargs = dict(
        tokenizer=tokenizer,
        model=model,
        engine=engine,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        max_problems=max_problems,
    )

    results = {}

    # HumanEval
    print0("=" * 50)
    print0("Running HumanEval ...")
    humaneval_task = HumanEval()
    with autocast_ctx:
        humaneval_pass1 = run_code_eval(humaneval_task, **eval_kwargs)
    results['HumanEval'] = humaneval_pass1
    print0(f"HumanEval pass@1: {100 * humaneval_pass1:.2f}%")

    # MBPP
    print0("=" * 50)
    print0("Running MBPP ...")
    mbpp_task = MBPP()
    with autocast_ctx:
        mbpp_pass1 = run_code_eval(mbpp_task, **eval_kwargs)
    results['MBPP'] = mbpp_pass1
    print0(f"MBPP pass@1: {100 * mbpp_pass1:.2f}%")

    # Summary
    print0("=" * 50)
    print0("Summary:")
    for task_name, pass1 in results.items():
        print0(f"  {task_name}: {100 * pass1:.2f}%")

    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Code evaluation " + args.source, data=[
        vars(args), # CLI args
        results,
    ])

    compute_cleanup()

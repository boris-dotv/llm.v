"""
Generate text from a pre-trained base model.
python -m scripts.base_gen -p "Let's calculate what is x when x * 5 = 15" -s 76500 -n 128

"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prompt', type=str, default='The capital of France is')
parser.add_argument('-s', '--step', type=int, default=None)
parser.add_argument('-m', '--model-tag', type=str, default=None)
parser.add_argument('-n', '--num-tokens', type=int, default=128)
parser.add_argument('-t', '--temperature', type=float, default=0.8)
parser.add_argument('-k', '--top-k', type=int, default=50)
parser.add_argument('--device-type', type=str, default='')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16')
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

if ddp_rank == 0:
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    bos_token_id = tokenizer.get_bos_token_id()

    print(f"Model: step {meta['step']}, depth {meta['model_config']['n_layer']}")
    print(f"Prompt: {args.prompt}\n")

    prompt_tokens = tokenizer.encode(args.prompt, prepend=bos_token_id)

    with autocast_ctx:
        for token_column, _ in engine.generate(prompt_tokens, num_samples=1, max_tokens=args.num_tokens, temperature=args.temperature, top_k=args.top_k):
            token = token_column[0]
            print(tokenizer.decode([token]), end="", flush=True)
    print()

print("Done!")
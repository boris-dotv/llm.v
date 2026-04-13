"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--code-data-dir', type=str, default=None, help='Path to code data parquets (enables code+text mixed training)')
parser.add_argument('--code-weight', type=float, default=0.7, help='Fraction of code data in tokenizer training (default: 0.7)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
if args.code_data_dir:
    print(f"code_data_dir: {args.code_data_dir}")
    print(f"code_weight: {args.code_weight}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    4) If code_data_dir is set, interleave code and text data with weighted round-robin
    """
    nchars = 0

    if args.code_data_dir:
        # Mixed code + text mode
        from nanochat.dataset import list_parquet_files
        import pyarrow.parquet as pq

        code_parquets = list_parquet_files(data_dir=args.code_data_dir)
        text_batches = parquets_iter_batched(split="train")

        def code_batches():
            while True:
                for fp in code_parquets:
                    pf = pq.ParquetFile(fp)
                    for rg_idx in range(pf.num_row_groups):
                        rg = pf.read_row_group(rg_idx)
                        yield rg.column('text').to_pylist()

        code_iter = code_batches()
        # weighted round-robin: e.g. 7 code batches per 3 text batches for 0.7 weight
        code_count = int(args.code_weight * 10)
        text_count = 10 - code_count
        step = 0
        while nchars <= args.max_chars:
            if step % 10 < code_count:
                batch = next(code_iter)
            else:
                batch = next(text_batches)
            step += 1
            for doc in batch:
                doc_text = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return
    else:
        # Original text-only mode
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])

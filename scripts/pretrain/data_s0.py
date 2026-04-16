"""
S0 streaming data pipeline for dense pretraining.

Streams from HuggingFace datasets, tokenizes on the fly, and packs sequences
using BOS-aligned bestfit packing (same algorithm as nanochat/dataloader.py).

Data mix:
  - FineWeb-Edu (English)           70%
  - The Stack v2 dedup (code)       15%
  - OpenWebMath (math)              10%
  - FineWeb-2 Chinese subset (zh)    5%

Usage:
    # In s0_launch.py:
    from scripts.pretrain.data_s0 import build_data_iterator, smoke_data_iterator

    # Real streaming (requires internet):
    data_iter = build_data_iterator(tokenizer, batch_size=4, seq_len=2048)

    # Smoke test (random data, no internet):
    data_iter = smoke_data_iterator(vocab_size=65536, batch_size=4, seq_len=2048)
"""

import sys
import random

import torch

# ---------------------------------------------------------------------------
# Dataset source configurations
# ---------------------------------------------------------------------------
DATASET_SOURCES = [
    {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "default",
        "split": "train",
        "text_field": "text",
        "weight": 0.70,
        "label": "FineWeb-Edu",
    },
    {
        "path": "bigcode/the-stack-v2-dedup",
        "name": None,
        "split": "train",
        "text_field": "content",
        "weight": 0.15,
        "label": "Stack-v2",
    },
    {
        "path": "open-web-math/open-web-math",
        "name": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.10,
        "label": "OpenWebMath",
    },
    {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "zho_Hans",
        "split": "train",
        "text_field": "text",
        "weight": 0.05,
        "label": "FineWeb-2-zh",
    },
]


# ---------------------------------------------------------------------------
# Placeholder tokenizer adapter (wraps HuggingFace AutoTokenizer)
# ---------------------------------------------------------------------------
class HFTokenizerAdapter:
    """Adapts a HuggingFace AutoTokenizer to the nanochat tokenizer interface."""

    def __init__(self, hf_tokenizer):
        self.hf_tok = hf_tokenizer
        # Resolve BOS token id
        if hf_tokenizer.bos_token_id is not None:
            self._bos_id = hf_tokenizer.bos_token_id
        elif hasattr(hf_tokenizer, 'encode'):
            # Qwen models use <|endoftext|> as the boundary token
            self._bos_id = hf_tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if self._bos_id == hf_tokenizer.unk_token_id:
                self._bos_id = 0  # last resort
        else:
            self._bos_id = 0

    def get_vocab_size(self):
        return len(self.hf_tok)

    def get_bos_token_id(self):
        return self._bos_id

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, str):
            ids = self.hf_tok.encode(text, add_special_tokens=False)
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.get_bos_token_id()
                ids = [prepend_id] + ids
            return ids
        elif isinstance(text, list):
            results = []
            for t in text:
                ids = self.hf_tok.encode(t, add_special_tokens=False)
                if prepend is not None:
                    prepend_id = prepend if isinstance(prepend, int) else self.get_bos_token_id()
                    ids = [prepend_id] + ids
                results.append(ids)
            return results
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.hf_tok.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Tokenizer loading with fallback
# ---------------------------------------------------------------------------
_PLACEHOLDER_WARNING = """
\033[91m========================================================
WARNING: USING PLACEHOLDER TOKENIZER (Qwen2.5)
This run is for plumbing validation ONLY.
Weights produced here are NOT transferable to the real
bilingual tokenizer. Do not use for real S0 training.
========================================================\033[0m
"""


def get_s0_tokenizer(allow_placeholder=False):
    """Load the nanochat tokenizer, or fall back to Qwen2.5 placeholder.

    Returns:
        (tokenizer, vocab_size, is_placeholder)
    """
    try:
        from nanochat.tokenizer import get_tokenizer
        tok = get_tokenizer()
        return tok, tok.get_vocab_size(), False
    except Exception:
        pass

    if not allow_placeholder:
        print(
            "\033[91mERROR: nanochat tokenizer not found.\033[0m\n"
            "The bilingual tokenizer (Track B) has not been trained yet.\n"
            "To proceed with a placeholder tokenizer for plumbing validation,\n"
            "pass --allow-placeholder-tokenizer.\n"
            "\n"
            "WARNING: checkpoints produced with the placeholder tokenizer\n"
            "are NOT compatible with the real tokenizer.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fall back to Qwen2.5-0.5B tokenizer
    print(_PLACEHOLDER_WARNING, file=sys.stderr)
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    adapter = HFTokenizerAdapter(hf_tok)
    return adapter, adapter.get_vocab_size(), True


# ---------------------------------------------------------------------------
# HuggingFace streaming document sources
# ---------------------------------------------------------------------------
def _stream_documents(source_cfg, seed=42):
    """Infinite iterator over text documents from a single HF streaming dataset."""
    from datasets import load_dataset

    kwargs = dict(
        path=source_cfg["path"],
        split=source_cfg["split"],
        streaming=True,
        trust_remote_code=True,
    )
    if source_cfg["name"] is not None:
        kwargs["name"] = source_cfg["name"]

    text_field = source_cfg["text_field"]

    while True:  # infinite cycling
        ds = load_dataset(**kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for example in ds:
            text = example.get(text_field, "")
            if text and len(text.strip()) > 0:
                yield text
        seed += 1  # vary shuffle on each epoch


def _weighted_document_iterator(sources, seed=42):
    """Weighted round-robin over multiple document streams.

    Same pattern as nanochat/dataloader.py _mixed_document_batches.
    """
    rng = random.Random(seed)
    iterators = [_stream_documents(src, seed=seed + i) for i, src in enumerate(sources)]
    weights = [src["weight"] for src in sources]

    while True:
        idx = rng.choices(range(len(iterators)), weights=weights, k=1)[0]
        yield next(iterators[idx])


# ---------------------------------------------------------------------------
# Tokenized document iterator
# ---------------------------------------------------------------------------
def _tokenized_document_iterator(tokenizer, sources, seed=42, batch_size=128):
    """Yield individual tokenized documents (list[int]) from the weighted mix."""
    bos_token_id = tokenizer.get_bos_token_id()
    doc_iter = _weighted_document_iterator(sources, seed=seed)

    while True:
        # Accumulate a batch of raw texts
        texts = [next(doc_iter) for _ in range(batch_size)]
        # Tokenize the batch, each prepended with BOS
        token_lists = tokenizer.encode(texts, prepend=bos_token_id)
        for tokens in token_lists:
            if len(tokens) > 1:  # skip empty docs (just BOS)
                yield tokens


# ---------------------------------------------------------------------------
# BOS-aligned bestfit sequence packing
# ---------------------------------------------------------------------------
def _pack_sequences(tokenized_doc_iter, batch_size, seq_len, buffer_size=1000):
    """BOS-aligned bestfit packing, yielding dict(input_ids, labels).

    Replicates the algorithm from nanochat/dataloader.py lines 124-201:
    - Every row starts with BOS
    - 100% utilization (no padding)
    - Best-fit: pick largest doc that fits; crop shortest when nothing fits
    """
    row_capacity = seq_len + 1  # +1 for target at last position
    doc_buffer = []

    while True:
        rows = []
        for _ in range(batch_size):
            row = []
            while len(row) < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    doc_buffer.append(next(tokenized_doc_iter))

                remaining = row_capacity - len(row)

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)),
                                       key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])

            rows.append(row[:row_capacity])

        batch_tensor = torch.tensor(rows, dtype=torch.long)
        input_ids = batch_tensor[:, :-1]  # (B, T)
        labels = batch_tensor[:, 1:]      # (B, T)
        yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_data_iterator(tokenizer, batch_size, seq_len, seed=42,
                        sources=None, buffer_size=1000):
    """Build the S0 streaming data iterator.

    Args:
        tokenizer: A nanochat-compatible tokenizer (or HFTokenizerAdapter).
        batch_size: Micro-batch size (per device).
        seq_len: Sequence length.
        seed: Random seed for shuffling and source selection.
        sources: Optional override of DATASET_SOURCES.
        buffer_size: Document buffer size for bestfit packing.

    Yields:
        dict with "input_ids" and "labels", both LongTensor[B, T].
    """
    if sources is None:
        sources = DATASET_SOURCES

    tok_doc_iter = _tokenized_document_iterator(tokenizer, sources, seed=seed)
    return _pack_sequences(tok_doc_iter, batch_size, seq_len,
                           buffer_size=buffer_size)


def smoke_data_iterator(vocab_size, batch_size, seq_len, num_batches=100):
    """Yield random batches for smoke testing (no network, no tokenizer).

    Args:
        vocab_size: Vocabulary size for random token generation.
        batch_size: Micro-batch size.
        seq_len: Sequence length.
        num_batches: Number of batches to yield.

    Yields:
        dict with "input_ids" and "labels", both LongTensor[B, T].
    """
    rng = torch.Generator()
    rng.manual_seed(42)
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len),
                                  generator=rng)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len),
                               generator=rng)
        yield {"input_ids": input_ids, "labels": labels}

"""
Download code pretraining data from HuggingFace.

Supports two sources:
  - bigcode/starcoderdata (default, ungated, ~1T tokens, curated)
  - bigcode/the-stack-v2-dedup (gated, ~3T tokens, requires HF login)

Streams the dataset, applies quality filters, and writes parquet shards with a
`text` column to a configurable output directory. Shards are named
shard_NNNNN.parquet, matching the FineWeb convention used by the dataloader.

Usage:
    python scripts/download_code_data.py
    python scripts/download_code_data.py --max-shards 5
    python scripts/download_code_data.py --source the-stack-v2
    python scripts/download_code_data.py --languages python,javascript,typescript
"""

import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset
from nanochat.common import get_base_dir

# ---------------------------------------------------------------------------
# Defaults

DEFAULT_LANGUAGES = [
    "python", "javascript", "typescript", "java", "c", "cpp",
    "go", "rust", "kotlin", "swift", "scala", "ruby", "php", "shell",
]
DEFAULT_SHARD_SIZE = 100_000

# Language name mapping: our names -> dataset config names
STARCODERDATA_LANG_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "rust": "rust",
    "kotlin": "kotlin",
    "swift": "swift",
    "scala": "scala",
    "ruby": "ruby",
    "php": "php",
    "shell": "shell",
}

# ---------------------------------------------------------------------------
# Quality filter

def passes_quality_filter(row):
    """Return True if a row passes all quality filters."""
    size = row.get("size") or row.get("blob_size") or 0
    if size >= 100 * 1024:
        return False
    avg_line_length = row.get("avg_line_length")
    if avg_line_length is not None and avg_line_length >= 200:
        return False
    alphanum_fraction = row.get("alphanum_fraction")
    if alphanum_fraction is not None and alphanum_fraction <= 0.25:
        return False
    return True


def build_shard_path(output_dir, shard_idx):
    return os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")


def find_next_shard_idx(output_dir):
    idx = 0
    while os.path.exists(build_shard_path(output_dir, idx)):
        idx += 1
    return idx


def write_shard(rows, output_dir, shard_idx):
    path = build_shard_path(output_dir, shard_idx)
    tmp_path = path + ".tmp"
    texts = [r["text"] for r in rows]
    langs = [r["lang"] for r in rows]
    table = pa.table({"text": texts, "lang": langs})
    pq.write_table(table, tmp_path)
    os.rename(tmp_path, path)
    return path


def stream_the_stack_v1(lang):
    """Stream from bigcode/the-stack-dedup. Ungated, native parquet, ~358B tokens."""
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{lang}",
        split="train",
        streaming=True,
    )
    for row in ds:
        content = row.get("content", "")
        if not content:
            continue
        yield {"text": content, "lang": lang, **{k: row.get(k) for k in
               ("size", "avg_line_length", "alphanum_fraction") if k in row}}


def main():
    parser = argparse.ArgumentParser(
        description="Download code data and write filtered parquet shards."
    )
    parser.add_argument("--source", default="the-stack-v1",
                        choices=["the-stack-v1"],
                        help="Dataset source (default: the-stack-v1 = bigcode/the-stack-dedup)")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                        help="Comma-separated list of languages")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for parquet shards (default: $NANOCHAT_BASE_DIR/code_data/)")
    parser.add_argument("--max-shards", type=int, default=-1,
                        help="Max shards to write (-1 = unlimited)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help=f"Rows per shard (default: {DEFAULT_SHARD_SIZE})")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(get_base_dir(), "code_data")
    os.makedirs(output_dir, exist_ok=True)

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    print(f"Source       : {args.source}")
    print(f"Output dir   : {output_dir}")
    print(f"Languages    : {languages}")
    print(f"Shard size   : {args.shard_size:,} rows")
    print(f"Max shards   : {'unlimited' if args.max_shards == -1 else args.max_shards}")
    print()

    shard_idx = find_next_shard_idx(output_dir)
    if shard_idx > 0:
        print(f"Resuming — skipping {shard_idx} already-complete shard(s).")
        print()

    stream_fn = stream_the_stack_v1
    rows_buf = []
    total_seen = 0
    total_kept = 0
    shards_written = 0

    for lang in languages:
        if args.max_shards != -1 and shards_written >= args.max_shards:
            break

        print(f"[{lang}] Streaming from {args.source} ...")
        try:
            for row in stream_fn(lang):
                total_seen += 1

                if not passes_quality_filter(row):
                    continue

                rows_buf.append({"text": row["text"], "lang": lang})
                total_kept += 1

                if len(rows_buf) >= args.shard_size:
                    path = write_shard(rows_buf, output_dir, shard_idx)
                    rows_buf = []
                    shards_written += 1
                    print(
                        f"  Wrote shard {shard_idx:05d} ({args.shard_size:,} rows) -> {path}"
                        f"  [seen={total_seen:,} kept={total_kept:,}]"
                    )
                    shard_idx += 1

                    if args.max_shards != -1 and shards_written >= args.max_shards:
                        print(f"Reached --max-shards={args.max_shards}, stopping.")
                        break
        except Exception as e:
            print(f"  WARNING: Could not load language '{lang}': {e}")
            continue

    if rows_buf and (args.max_shards == -1 or shards_written < args.max_shards):
        path = write_shard(rows_buf, output_dir, shard_idx)
        shards_written += 1
        print(f"  Wrote shard {shard_idx:05d} ({len(rows_buf):,} rows, partial) -> {path}")

    print()
    print(f"Done. Shards written: {shards_written}  |  Rows kept: {total_kept:,}  |  Rows seen: {total_seen:,}")


if __name__ == "__main__":
    main()

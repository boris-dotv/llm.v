"""
Download code pretraining data from bigcode/the-stack-v2-dedup on HuggingFace.

Streams the dataset, applies quality filters, and writes parquet shards with a
`text` column (renamed from `content`) to a configurable output directory.
Shards are named shard_NNNNN.parquet, matching the FineWeb convention used by
the existing dataloader.

Usage:
    python scripts/download_code_data.py
    python scripts/download_code_data.py --max-shards 5
    python scripts/download_code_data.py --output-dir /path/to/output --shard-size 50000
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

# ---------------------------------------------------------------------------
# Quality filter

def passes_quality_filter(row):
    """Return True if a row passes all quality filters."""
    # File size < 100 KB (size field is in bytes)
    size = row.get("size") or row.get("blob_size")
    if size is not None and size >= 100 * 1024:
        return False
    # avg_line_length < 200
    avg_line_length = row.get("avg_line_length")
    if avg_line_length is not None and avg_line_length >= 200:
        return False
    # alphanum_fraction > 0.25
    alphanum_fraction = row.get("alphanum_fraction")
    if alphanum_fraction is not None and alphanum_fraction <= 0.25:
        return False
    return True


def build_shard_path(output_dir, shard_idx):
    return os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")


def find_next_shard_idx(output_dir):
    """Return the index of the first shard that doesn't exist on disk."""
    idx = 0
    while os.path.exists(build_shard_path(output_dir, idx)):
        idx += 1
    return idx


def write_shard(rows, output_dir, shard_idx):
    path = build_shard_path(output_dir, shard_idx)
    tmp_path = path + ".tmp"
    # rows is a list of dicts; we only need the `text` column for the dataloader,
    # but carrying language along is useful for inspection.
    texts = [r["text"] for r in rows]
    langs = [r["lang"] for r in rows]
    table = pa.table({"text": texts, "lang": langs})
    pq.write_table(table, tmp_path)
    os.rename(tmp_path, path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download The Stack v2 dedup and write filtered parquet shards."
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated list of languages to include (default: all 14 preset languages)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write parquet shards (default: $NANOCHAT_BASE_DIR/code_data/)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=-1,
        help="Maximum number of shards to write. -1 = unlimited (default: -1)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=f"Number of rows per shard (default: {DEFAULT_SHARD_SIZE})",
    )
    args = parser.parse_args()

    # Resolve output directory
    output_dir = args.output_dir or os.path.join(get_base_dir(), "code_data")
    os.makedirs(output_dir, exist_ok=True)

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    print(f"Output dir   : {output_dir}")
    print(f"Languages    : {languages}")
    print(f"Shard size   : {args.shard_size:,} rows")
    print(f"Max shards   : {'unlimited' if args.max_shards == -1 else args.max_shards}")
    print()

    # Resume: skip shards that already exist
    shard_idx = find_next_shard_idx(output_dir)
    if shard_idx > 0:
        print(f"Resuming — skipping {shard_idx} already-complete shard(s).")
        print()

    # Stream one language subset at a time (avoids interleaving complexity)
    rows_buf = []
    total_seen = 0
    total_kept = 0
    shards_written = 0

    for lang in languages:
        if args.max_shards != -1 and shards_written >= args.max_shards:
            break

        print(f"[{lang}] Streaming ...")
        try:
            ds = load_dataset(
                "bigcode/the-stack-v2-dedup",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"  WARNING: Could not load language '{lang}': {e}")
            continue

        for row in ds:
            total_seen += 1

            # Quality filter
            if not passes_quality_filter(row):
                continue

            # Rename content -> text, keep lang tag
            content = row.get("content", "")
            if not content:
                continue
            rows_buf.append({"text": content, "lang": lang})
            total_kept += 1

            # Flush a shard when buffer is full
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

    # Write any leftover rows as a partial final shard (skip if empty)
    if rows_buf and (args.max_shards == -1 or shards_written < args.max_shards):
        path = write_shard(rows_buf, output_dir, shard_idx)
        shards_written += 1
        print(
            f"  Wrote shard {shard_idx:05d} ({len(rows_buf):,} rows, partial) -> {path}"
        )

    print()
    print(f"Done. Shards written: {shards_written}  |  Rows kept: {total_kept:,}  |  Rows seen: {total_seen:,}")


if __name__ == "__main__":
    main()

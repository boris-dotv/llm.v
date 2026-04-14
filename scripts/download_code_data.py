"""
Download code pretraining data from HuggingFace.

Downloads raw parquet files directly via huggingface_hub, bypassing the
datasets library entirely (avoids "loading scripts no longer supported"
errors). Reads each parquet, applies quality filters, renames content->text,
and writes output shards.

Default source: bigcode/the-stack-dedup (permissive licenses, ~358B tokens)

Usage:
    python scripts/download_code_data.py
    python scripts/download_code_data.py --max-shards 5
    python scripts/download_code_data.py --languages python,javascript
"""

import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from nanochat.common import get_base_dir

DEFAULT_LANGUAGES = [
    "python", "javascript", "typescript", "java", "c", "cpp",
    "go", "rust", "kotlin", "swift", "scala", "ruby", "php", "shell",
]
DEFAULT_SHARD_SIZE = 100_000
REPO_ID = "bigcode/starcoderdata"


def passes_quality_filter(content, avg_line_length=None, alphanum_fraction=None):
    if len(content.encode("utf-8", errors="ignore")) >= 100 * 1024:
        return False
    if avg_line_length is not None and avg_line_length >= 200:
        return False
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


def list_parquet_files_for_lang(fs, lang):
    """List all parquet files for a language in the-stack-dedup repo."""
    path = f"datasets/{REPO_ID}/data/{lang}"
    try:
        files = fs.ls(path, detail=False)
        return [f for f in files if f.endswith(".parquet")]
    except FileNotFoundError:
        return []


def stream_parquet_from_hf(fs, hf_path):
    """Read a single parquet file from HuggingFace and yield rows."""
    with fs.open(hf_path, "rb") as f:
        table = pq.read_table(f)
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        yield row


def main():
    parser = argparse.ArgumentParser(
        description="Download code data and write filtered parquet shards."
    )
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                        help="Comma-separated list of languages")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for parquet shards")
    parser.add_argument("--max-shards", type=int, default=-1,
                        help="Max shards to write (-1 = unlimited)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help=f"Rows per shard (default: {DEFAULT_SHARD_SIZE})")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(get_base_dir(), "code_data")
    os.makedirs(output_dir, exist_ok=True)

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    print(f"Repo         : {REPO_ID}")
    print(f"Output dir   : {output_dir}")
    print(f"Languages    : {languages}")
    print(f"Shard size   : {args.shard_size:,} rows")
    print(f"Max shards   : {'unlimited' if args.max_shards == -1 else args.max_shards}")
    print(flush=True)

    fs = HfFileSystem()

    shard_idx = find_next_shard_idx(output_dir)
    if shard_idx > 0:
        print(f"Resuming — skipping {shard_idx} already-complete shard(s).")
        print(flush=True)

    rows_buf = []
    total_seen = 0
    total_kept = 0
    shards_written = 0

    for lang in languages:
        if args.max_shards != -1 and shards_written >= args.max_shards:
            break

        print(f"[{lang}] Listing parquet files ...", flush=True)
        pq_files = list_parquet_files_for_lang(fs, lang)
        if not pq_files:
            print(f"  No parquet files found for '{lang}', skipping.", flush=True)
            continue
        print(f"  Found {len(pq_files)} parquet files", flush=True)

        for pq_file in pq_files:
            if args.max_shards != -1 and shards_written >= args.max_shards:
                break

            fname = os.path.basename(pq_file)
            print(f"  Reading {fname} ...", end=" ", flush=True)
            try:
                for row in stream_parquet_from_hf(fs, pq_file):
                    total_seen += 1
                    content = row.get("content", "")
                    if not content:
                        continue
                    avg_ll = row.get("avg_line_length")
                    alpha = row.get("alphanum_fraction")
                    if not passes_quality_filter(content, avg_ll, alpha):
                        continue
                    rows_buf.append({"text": content, "lang": lang})
                    total_kept += 1

                    if len(rows_buf) >= args.shard_size:
                        path = write_shard(rows_buf, output_dir, shard_idx)
                        rows_buf = []
                        shards_written += 1
                        print(
                            f"\n  Wrote shard {shard_idx:05d} ({args.shard_size:,} rows)"
                            f"  [seen={total_seen:,} kept={total_kept:,}]",
                            flush=True,
                        )
                        shard_idx += 1
                print(f"done (total kept so far: {total_kept:,})", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                continue

    if rows_buf and (args.max_shards == -1 or shards_written < args.max_shards):
        path = write_shard(rows_buf, output_dir, shard_idx)
        shards_written += 1
        print(f"  Wrote shard {shard_idx:05d} ({len(rows_buf):,} rows, partial)", flush=True)

    print()
    print(f"Done. Shards written: {shards_written}  |  Rows kept: {total_kept:,}  |  Rows seen: {total_seen:,}")


if __name__ == "__main__":
    main()

"""
Download SFT and RL datasets for the code training pipeline.

Each dataset is downloaded from HuggingFace and saved to disk under
$NANOCHAT_BASE_DIR/ using datasets.save_to_disk(). Already-downloaded
directories are skipped automatically.

Usage:
    python scripts/download_post_data.py
    python scripts/download_post_data.py --datasets magicoder_oss,sc2_exec
"""

import os
import argparse

from datasets import load_dataset
from nanochat.common import get_base_dir

# ---------------------------------------------------------------------------
# Dataset registry
#
# Each entry is a tuple of:
#   (save_dir_name, hf_id, split_or_splits)
#
# split_or_splits is either a string (single split -> saved directly) or a
# dict mapping {save_dir_suffix: split_name} for datasets with named splits.
# ---------------------------------------------------------------------------

DATASETS = {
    "magicoder_oss": (
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "train",
    ),
    "magicoder_evol": (
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "train",
    ),
    "sc2_exec": (
        "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "train",
    ),
    "code_feedback": (
        "m-a-p/CodeFeedback-Filtered-Instruction",
        "train",
    ),
    "open_code_interp": (
        "m-a-p/OpenCodeInterpreter-68K",
        "train",
    ),
    "taco_train": (
        "BAAI/TACO",
        "train",
    ),
    # APPS has train and test splits
    "apps": (
        "codeparrot/apps",
        {"apps_train": "train", "apps_test": "test"},
    ),
    # CodeContests has train and test splits
    "code_contests": (
        "deepmind/code_contests",
        {"code_contests_train": "train", "code_contests_test": "test"},
    ),
    "mbpp": (
        "google-research-datasets/mbpp",
        "train",
    ),
}


def download_dataset(key, base_dir, hf_id, splits):
    """
    Download one entry from the registry.

    `splits` is either a str (single split) or a dict {save_dir: split_name}.
    Returns the number of splits actually downloaded (0 if all were skipped).
    """
    if isinstance(splits, str):
        # Single split: save_dir == key
        save_dir = os.path.join(base_dir, key)
        if os.path.exists(save_dir):
            print(f"  [skip] {key} already exists at {save_dir}")
            return 0
        print(f"  Downloading {hf_id} (split={splits}) ...")
        ds = load_dataset(hf_id, split=splits)
        ds.save_to_disk(save_dir)
        print(f"  Saved {len(ds):,} rows -> {save_dir}")
        return 1
    else:
        # Multiple splits: dict {save_dir_name: split_name}
        downloaded = 0
        for save_name, split_name in splits.items():
            save_dir = os.path.join(base_dir, save_name)
            if os.path.exists(save_dir):
                print(f"  [skip] {save_name} already exists at {save_dir}")
                continue
            print(f"  Downloading {hf_id} (split={split_name}) ...")
            ds = load_dataset(hf_id, split=split_name)
            ds.save_to_disk(save_dir)
            print(f"  Saved {len(ds):,} rows -> {save_dir}")
            downloaded += 1
        return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download code SFT/RL datasets and save to disk."
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated list of dataset keys to download "
            f"(default: all). Available: {', '.join(DATASETS)}"
        ),
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    print(f"Base dir: {base_dir}")
    print()

    if args.datasets:
        keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
        unknown = [k for k in keys if k not in DATASETS]
        if unknown:
            parser.error(f"Unknown dataset key(s): {', '.join(unknown)}. Available: {', '.join(DATASETS)}")
    else:
        keys = list(DATASETS.keys())

    total_downloaded = 0
    total_skipped = 0

    for key in keys:
        hf_id, splits = DATASETS[key]
        print(f"=== {key} ({hf_id}) ===")
        try:
            n = download_dataset(key, base_dir, hf_id, splits)
            if isinstance(splits, dict):
                expected = len(splits)
            else:
                expected = 1
            total_downloaded += n
            total_skipped += expected - n
        except Exception as e:
            print(f"  ERROR downloading {key}: {e}")
        print()

    print(f"Done. Downloaded: {total_downloaded}  Skipped (already exist): {total_skipped}")


if __name__ == "__main__":
    main()

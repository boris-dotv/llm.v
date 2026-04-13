"""
Magicoder datasets for code SFT training.
https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K
"""

import os
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir

BASE_DIR = get_base_dir()


class MagicoderOSS(Task):
    """Magicoder OSS-Instruct dataset. ~75K coding instruction pairs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds_path = os.path.join(BASE_DIR, "magicoder_oss")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ]
        return {"messages": messages}


class MagicoderEvol(Task):
    """Magicoder Evol-Instruct dataset. ~110K evolved coding instruction pairs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds_path = os.path.join(BASE_DIR, "magicoder_evol")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
        return {"messages": messages}

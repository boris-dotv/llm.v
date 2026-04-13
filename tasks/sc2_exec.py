"""
StarCoder2-Self-Alignment execution-filtered dataset for code SFT training.
https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k
"""

import os
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir

BASE_DIR = get_base_dir()


class SC2ExecFiltered(Task):
    """SC2 self-alignment execution-filtered dataset. ~50K code instruction pairs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds_path = os.path.join(BASE_DIR, "sc2_exec")
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

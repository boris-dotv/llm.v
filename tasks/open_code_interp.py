"""
OpenCodeInterpreter multi-turn dataset for code SFT training.
https://huggingface.co/datasets/m-a-p/OpenCodeInterpreter-DS-1.3B
"""

import os
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir

BASE_DIR = get_base_dir()


class OpenCodeInterpreter(Task):
    """OpenCodeInterpreter multi-turn code conversation dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds_path = os.path.join(BASE_DIR, "open_code_interp")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = []
        for turn in row["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        return {"messages": messages}

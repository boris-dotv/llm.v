"""
CodeFeedback dataset for code SFT training.
https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction
"""

import os
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir

BASE_DIR = get_base_dir()


class CodeFeedback(Task):
    """CodeFeedback-Filtered-Instruction dataset. ~66K code instruction pairs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds_path = os.path.join(BASE_DIR, "code_feedback")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        return {"messages": messages}

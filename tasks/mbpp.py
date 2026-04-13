"""
MBPP (Mostly Basic Python Problems) evaluation dataset.
https://huggingface.co/datasets/google-research-datasets/mbpp
"""

import os
import re
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir
from nanochat.execution import execute_code

BASE_DIR = get_base_dir()


def extract_program(completion):
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


class MBPP(Task):
    """MBPP sanitized evaluation dataset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds_path = os.path.join(BASE_DIR, "mbpp")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": row["code"]},
        ]
        return {
            "messages": messages,
            "test_list": row["test_list"],
        }

    def evaluate(self, conversation, completion):
        """Return 1 if the completion passes all test assertions, 0 otherwise."""
        code = extract_program(completion)
        test_list = conversation.get("test_list", [])
        program = code + "\n" + "\n".join(test_list)
        result = execute_code(program)
        return 1 if result.success else 0

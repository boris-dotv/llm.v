"""
APPS (Automated Programming Progress Standard) dataset for RL training.
https://huggingface.co/datasets/codeparrot/apps
"""

import os
import json
import re
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir
from nanochat.code_execution import execution_reward

BASE_DIR = get_base_dir()


def extract_program(completion):
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


class APPSTask(Task):
    """APPS coding benchmark dataset for RL training."""

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "APPSTask split must be train|test"
        if split == "train":
            ds_path = os.path.join(BASE_DIR, "apps_train")
        else:
            ds_path = os.path.join(BASE_DIR, "apps_test")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]

        # Parse solutions; fall back to placeholder if invalid
        try:
            solutions = json.loads(row["solutions"])
            assistant_content = solutions[0] if solutions else "# no solution available"
        except (json.JSONDecodeError, IndexError, TypeError):
            assistant_content = "# no solution available"

        # Parse test cases
        test_cases = []
        try:
            io_data = json.loads(row["input_output"])
            inputs = io_data.get("inputs", [])
            outputs = io_data.get("outputs", [])
            test_cases = [
                {"input": str(inp), "output": str(out)}
                for inp, out in zip(inputs, outputs)
            ]
        except (json.JSONDecodeError, TypeError):
            pass

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        return {
            "messages": messages,
            "_test_cases": test_cases,
        }

    def reward(self, conversation, assistant_response):
        code = extract_program(assistant_response)
        test_cases = conversation.get("_test_cases", [])
        return execution_reward(code, test_cases)

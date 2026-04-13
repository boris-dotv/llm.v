"""
CodeContests dataset for RL training.
https://huggingface.co/datasets/deepmind/code_contests
"""

import os
import re
from datasets import load_from_disk
from tasks.common import Task
from nanochat.common import get_base_dir
from nanochat.code_execution import execution_reward

BASE_DIR = get_base_dir()

# Language ID for Python 3 in the CodeContests dataset
_PYTHON3_LANGUAGE = 3


def extract_program(completion):
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


def _extract_python_solution(solutions):
    """
    Extract the first Python 3 solution from a solutions dict.
    solutions is a dict with 'language' (list of ints) and 'solution' (list of str).
    Returns the solution string or None.
    """
    if not solutions:
        return None
    languages = solutions.get("language", [])
    sol_texts = solutions.get("solution", [])
    for lang, sol in zip(languages, sol_texts):
        if lang == _PYTHON3_LANGUAGE:
            return sol
    # Fall back to first solution if no Python found
    if sol_texts:
        return sol_texts[0]
    return None


class CodeContests(Task):
    """DeepMind CodeContests dataset for RL training."""

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "CodeContests split must be train|test"
        if split == "train":
            ds_path = os.path.join(BASE_DIR, "code_contests_train")
        else:
            ds_path = os.path.join(BASE_DIR, "code_contests_test")
        self.ds = load_from_disk(ds_path)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        description = row["description"]

        # Extract a Python solution (prefer solutions, fall back to correct_solutions)
        solution = _extract_python_solution(row.get("solutions"))
        if solution is None:
            solution = _extract_python_solution(row.get("correct_solutions"))
        if solution is None:
            solution = "# no solution available"

        # Use public_tests for reward signal
        test_cases = []
        public_tests = row.get("public_tests", {})
        inputs = public_tests.get("input", [])
        outputs = public_tests.get("output", [])
        test_cases = [
            {"input": str(inp), "output": str(out)}
            for inp, out in zip(inputs, outputs)
        ]

        messages = [
            {"role": "user", "content": description},
            {"role": "assistant", "content": solution},
        ]
        return {
            "messages": messages,
            "_test_cases": test_cases,
        }

    def reward(self, conversation, assistant_response):
        code = extract_program(assistant_response)
        test_cases = conversation.get("_test_cases", [])
        return execution_reward(code, test_cases)

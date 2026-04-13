"""
Code execution utilities for RL reward computation.

Provides subprocess-based code execution with stdin/stdout support,
suitable for competitive programming problems (APPS, TACO, CodeContests)
where solutions read from stdin and write to stdout.

Uses subprocess.run with pipes rather than exec() to support input().
"""

import subprocess
import tempfile
import os


def execute_with_stdin(code: str, stdin_input: str, expected_output: str,
                       timeout: float = 10.0) -> bool:
    """
    Execute Python code with given stdin, compare stdout to expected output.
    Returns True if output matches after stripping trailing whitespace per line.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False
        # Compare output: strip trailing whitespace per line, ignore trailing newlines
        actual = result.stdout.rstrip()
        expected = expected_output.rstrip()
        return actual == expected
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def execute_with_tests(code: str, test_code: str, timeout: float = 10.0) -> bool:
    """
    Execute code followed by test assertions (HumanEval/MBPP style).
    Returns True if all assertions pass.
    """
    program = code + "\n" + test_code
    try:
        result = subprocess.run(
            ["python3", "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def execution_reward(code: str, test_cases: list[dict],
                     max_tests: int = 5, timeout_per_test: float = 10.0) -> float:
    """
    Run code against test_cases, return fraction passing as reward in [0, 1].

    Args:
        code: Python code to execute
        test_cases: list of {"input": str, "output": str} dicts
        max_tests: maximum number of test cases to run (for speed)
        timeout_per_test: timeout per test case in seconds

    Returns:
        Fraction of tests passing (e.g. 3/5 = 0.6). Returns 0.0 if no test cases.
    """
    if not test_cases:
        return 0.0

    tests_to_run = test_cases[:max_tests]
    passed = 0
    for tc in tests_to_run:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        if execute_with_stdin(code, inp, out, timeout=timeout_per_test):
            passed += 1

    return passed / len(tests_to_run)

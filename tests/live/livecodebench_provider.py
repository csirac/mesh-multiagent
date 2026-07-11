"""
LiveCodeBench providers for code generation benchmarks.

Loads problems from HuggingFace, formats prompts, and evaluates solutions
using sandboxed subprocess execution.

Supports two problem types:
1. Functional (LeetCode) - class Solution with method stubs
2. Stdin/stdout (AtCoder/CodeForces) - read stdin, write stdout

Reuses MeshDecoder and DirectOpenAIDecoder from evalplus_provider for code
generation. This module handles problem loading, prompt formatting, and
local evaluation.

Usage:
    from tests.live.livecodebench_provider import (
        load_problems, build_prompt, evaluate_solution,
        evaluate_solutions_batch,
    )

    problems = load_problems(version="release_v6", max_problems=50)
    prompt = build_prompt(problems[0])
    result = evaluate_solution(problems[0], code)
"""

from __future__ import annotations

import ast
import json
import logging
import multiprocessing
import os
import pickle
import re
import signal
import sys
import time
import zlib
import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch, mock_open

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Version → JSONL files mapping (from LCB's loading script)
_VERSION_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


# ---------------------------------------------------------------------------
# Data model (mirrors LiveCodeBench's CodeGenerationProblem)
# ---------------------------------------------------------------------------

class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class TestCase:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        if isinstance(self.testtype, str):
            self.testtype = TestType(self.testtype)


@dataclass
class LCBProblem:
    """A LiveCodeBench code generation problem."""
    question_id: str
    question_title: str
    question_content: str
    platform: Platform
    difficulty: Difficulty
    contest_id: str
    contest_date: datetime
    starter_code: str
    public_test_cases: list[TestCase]
    private_test_cases: list[TestCase]
    metadata: dict

    @property
    def is_functional(self) -> bool:
        """LeetCode problems use class Solution with function calls."""
        return bool(self.starter_code)

    @property
    def func_name(self) -> str | None:
        """Function name for call-based evaluation."""
        return self.metadata.get("func_name")

    def get_evaluation_sample(self) -> dict:
        """Format for testing_util.run_test compatibility."""
        all_tests = self.public_test_cases + self.private_test_cases
        return {
            "input_output": json.dumps({
                "inputs": [t.input for t in all_tests],
                "outputs": [t.output for t in all_tests],
                "fn_name": self.func_name,
            }),
        }

    def get_public_evaluation_sample(self) -> dict:
        """Format using only public test cases (for agent self-testing)."""
        return {
            "input_output": json.dumps({
                "inputs": [t.input for t in self.public_test_cases],
                "outputs": [t.output for t in self.public_test_cases],
                "fn_name": self.func_name,
            }),
        }


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_problems(
    version: str = "release_v6",
    max_problems: int | None = None,
    difficulty: str | None = None,
    platform: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[LCBProblem]:
    """
    Load LiveCodeBench problems from HuggingFace.

    Args:
        version: Dataset version tag (e.g., "release_v6")
        max_problems: Limit number of problems loaded
        difficulty: Filter by difficulty ("easy", "medium", "hard")
        platform: Filter by platform ("leetcode", "codeforces", "atcoder")
        start_date: Filter problems after this date (YYYY-MM-DD)
        end_date: Filter problems before this date (YYYY-MM-DD)

    Returns:
        List of LCBProblem objects
    """
    logger.info(f"Loading LiveCodeBench dataset (version={version})...")

    if version not in _VERSION_FILES:
        raise ValueError(f"Unknown version {version}. Available: {list(_VERSION_FILES.keys())}")

    # Download JSONL files from HuggingFace Hub
    rows = []
    for filename in _VERSION_FILES[version]:
        local_path = hf_hub_download(
            "livecodebench/code_generation_lite",
            filename,
            repo_type="dataset",
        )
        with open(local_path) as f:
            for line in f:
                rows.append(json.loads(line))

    logger.info(f"Loaded {len(rows)} raw rows from {len(_VERSION_FILES[version])} files")

    problems = []
    for row in rows:
        # Parse test cases
        pub_tests = json.loads(row["public_test_cases"])
        pub_tests = [TestCase(**t) for t in pub_tests]

        try:
            priv_tests = json.loads(row["private_test_cases"])
        except (json.JSONDecodeError, TypeError):
            # Compressed format
            priv_tests = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(row["private_test_cases"].encode("utf-8"))
                    )
                )
            )
        priv_tests = [TestCase(**t) for t in priv_tests]

        metadata = json.loads(row["metadata"])
        contest_date = datetime.fromisoformat(row["contest_date"])

        prob = LCBProblem(
            question_id=row["question_id"],
            question_title=row["question_title"],
            question_content=row["question_content"],
            platform=Platform(row["platform"]),
            difficulty=Difficulty(row["difficulty"]),
            contest_id=row["contest_id"],
            contest_date=contest_date,
            starter_code=row["starter_code"],
            public_test_cases=pub_tests,
            private_test_cases=priv_tests,
            metadata=metadata,
        )
        problems.append(prob)

    # Apply filters
    if difficulty:
        diff = Difficulty(difficulty.lower())
        problems = [p for p in problems if p.difficulty == diff]

    if platform:
        plat = Platform(platform.lower())
        problems = [p for p in problems if p.platform == plat]

    if start_date:
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date >= dt]

    if end_date:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date <= dt]

    if max_problems:
        problems = problems[:max_problems]

    logger.info(f"Loaded {len(problems)} problems")
    return problems


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters."
)

FORMATTING_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer "
    "to stdout (do not directly test on the sample inputs). Enclose your "
    "code within delimiters as follows. Ensure that when the python program "
    "runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def build_prompt(problem: LCBProblem, include_tests: bool = False) -> str:
    """
    Build a prompt for a LiveCodeBench problem.

    Args:
        problem: The problem to format
        include_tests: Whether to include public test cases in the prompt

    Returns:
        Formatted prompt string
    """
    prompt = f"{SYSTEM_MESSAGE}\n\n"
    prompt += f"### Question:\n{problem.question_content}\n\n"

    if problem.is_functional:
        prompt += f"### Format: {FORMATTING_WITH_STARTER}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_STDIN}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"

    if include_tests and problem.public_test_cases:
        prompt += "### Public Test Cases:\n"
        for i, tc in enumerate(problem.public_test_cases):
            prompt += f"\n**Test {i + 1}:**\n"
            prompt += f"Input:\n```\n{tc.input}\n```\n"
            prompt += f"Expected Output:\n```\n{tc.output}\n```\n"
        prompt += "\n"

    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def build_agentic_prompt(problem: LCBProblem) -> str:
    """
    Build an agentic prompt that includes public test cases and
    encourages the agent to test its code.
    """
    prompt = "IGNORE ALL PREVIOUS CONTEXT. This is a new, independent problem.\n\n"
    prompt += f"{SYSTEM_MESSAGE}\n\n"
    prompt += f"### Question:\n{problem.question_content}\n\n"

    if problem.is_functional:
        prompt += f"### Format: {FORMATTING_WITH_STARTER}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_STDIN}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"

    # Include public test cases for the agent to use
    if problem.public_test_cases:
        prompt += "### Public Test Cases:\n"
        for i, tc in enumerate(problem.public_test_cases):
            prompt += f"\n**Test {i + 1}:**\n"
            prompt += f"Input:\n```\n{tc.input}\n```\n"
            prompt += f"Expected Output:\n```\n{tc.output}\n```\n"
        prompt += "\n"

    prompt += """You have access to tools for testing your code. Please:
1. Write your implementation
2. Use bash_exec to test it against the public test cases above
3. If tests fail, read the error and fix your code
4. Iterate until your code works correctly

IMPORTANT:
- Include ALL imports needed
- Do NOT reference any files or code from previous problems

When you're confident your code is correct, respond with your final implementation in a ```python code block.
"""
    return prompt


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(response: str, problem: LCBProblem) -> str:
    """
    Extract Python code from an LLM response.

    For functional problems (LeetCode), extracts code containing 'class Solution'.
    For stdin problems, extracts the last fenced code block.

    Returns empty string if extraction fails.
    """
    # Find all fenced code blocks
    code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)

    if not code_blocks:
        return ""

    if problem.is_functional:
        # Prefer block containing 'class Solution'
        for block in code_blocks:
            if "class Solution" in block:
                return _clean_code(block.strip())
        # Fall back to block containing the function name
        if problem.func_name:
            for block in code_blocks:
                if f"def {problem.func_name}" in block:
                    return _clean_code(block.strip())
    else:
        # For stdin problems, use the last non-empty code block
        # (LLMs often show the problem first, then the solution)
        pass

    # Fall back to last non-empty block
    for block in reversed(code_blocks):
        if block.strip():
            return _clean_code(block.strip())

    return ""


def _clean_code(code: str) -> str:
    """Remove trailing test/debug code from extracted code."""
    lines = code.split('\n')
    clean_lines = []

    for line in lines:
        stripped = line.strip()
        # Stop at standalone print/assert at module level (test code)
        # but only if not inside class/function
        if not line[0:1].isspace() and stripped.startswith(('print(', 'assert ')):
            # Allow print() inside stdin solutions
            if 'class Solution' in '\n'.join(clean_lines):
                break
            # For stdin solutions, print() is part of the solution
            clean_lines.append(line)
        else:
            clean_lines.append(line)

    # Remove trailing empty lines
    while clean_lines and not clean_lines[-1].strip():
        clean_lines.pop()

    return '\n'.join(clean_lines)


# ---------------------------------------------------------------------------
# Evaluation (sandboxed subprocess execution)
# ---------------------------------------------------------------------------

# Standard imports prepended to all code for execution
_IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\n"
    "import heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)


def _run_test_worker(sample: dict, code: str, timeout: int, result_queue):
    """
    Worker function for subprocess-based test execution.
    Runs in a separate process for isolation.
    """
    # Set up signal handler for timeout
    def timeout_handler(signum, frame):
        raise TimeoutError("Test execution timed out")

    signal.signal(signal.SIGALRM, timeout_handler)

    # Apply security restrictions
    _reliability_guard()

    try:
        in_outs = json.loads(sample["input_output"])
        fn_name = in_outs.get("fn_name")
        inputs = in_outs["inputs"]
        outputs = in_outs["outputs"]

        if fn_name:
            results, metadata = _grade_call_based(code, inputs, outputs, fn_name, timeout)
        else:
            results, metadata = _grade_stdio(code, inputs, outputs, timeout)

        result_queue.put(("ok", results, metadata))
    except Exception as e:
        result_queue.put(("error", [-4], {"error": repr(e), "error_code": -4}))


def _grade_call_based(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    """Grade a call-based (functional) solution."""
    code = _IMPORT_STRING + "\n\n" + code

    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if hasattr(tmp_sol, "Solution"):
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
    finally:
        signal.alarm(0)

    method = getattr(compiled_sol, fn_name, None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": f"Function {fn_name} not found"}

    all_inputs_parsed = [
        [json.loads(line) for line in inp.split("\n")] for inp in all_inputs
    ]
    all_outputs_parsed = [json.loads(out) for out in all_outputs]

    all_results = []
    total_time = 0
    for inp, expected in zip(all_inputs_parsed, all_outputs_parsed):
        signal.alarm(timeout)
        try:
            start = time.time()
            prediction = method(*inp)
            total_time += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            passed = prediction == expected
            all_results.append(passed)

            if not passed:
                return all_results, {
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except TimeoutError:
            signal.alarm(0)
            all_results.append(-3)
            return all_results, {"error_code": -3, "error_message": "Time Limit Exceeded"}
        except Exception as e:
            signal.alarm(0)
            all_results.append(-4)
            return all_results, {"error_code": -4, "error_message": f"Runtime Error: {e}"}
        finally:
            signal.alarm(0)

    return all_results, {"execution_time": total_time}


def _clean_if_name(code: str) -> str:
    """Remove if __name__ == '__main__' wrapper."""
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1])
                    + "\n"
                    + ast.unparse(last_block.body)
                )
    except Exception:
        pass
    return code


def _make_function(code: str) -> str:
    """Wrap stdin-style code in a function for isolated execution."""
    try:
        import_stmts = []
        other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)

        func_def = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return (
            _IMPORT_STRING + "\n"
            + ast.unparse(import_stmts) + "\n"
            + ast.unparse(func_def)
        )
    except Exception:
        return code


class _MockStdinWithBuffer:
    """Mock for sys.stdin that supports .buffer attribute."""
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = type('MockBuffer', (), {
            'read': lambda self_, *a: inputs.encode('utf-8'),
            'readline': lambda self_, *a: inputs.encode('utf-8').split(b'\n')[0] + b'\n',
        })()

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class _Capturing(list):
    """Capture stdout output."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda: None
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


def _call_method(method, inputs):
    """Call a stdin-style function with mocked input."""
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = _MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner(m):
        try:
            return m()
        except SystemExit:
            pass

    return _inner(method)


def _get_stripped_lines(val: str) -> list[str]:
    """Strip and split output into lines."""
    val = val.strip()
    return [line.strip() for line in val.split("\n")]


def _grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    """Grade a stdin/stdout solution."""
    from decimal import Decimal

    code = _clean_if_name(code)
    code = _make_function(code)

    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        compiled_sol = tmp_sol
    finally:
        signal.alarm(0)

    method = getattr(compiled_sol, "wrapped_function", None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": "Could not compile code"}

    all_results = []
    total_time = 0
    for inp, expected_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        with _Capturing() as captured:
            try:
                start = time.time()
                _call_method(method, inp)
                total_time += time.time() - start
                signal.alarm(0)
            except TimeoutError:
                signal.alarm(0)
                all_results.append(-3)
                return all_results, {"error_code": -3, "error_message": "Time Limit Exceeded"}
            except Exception as e:
                signal.alarm(0)
                all_results.append(-4)
                return all_results, {"error_code": -4, "error_message": f"Runtime Error: {e}"}
            finally:
                signal.alarm(0)

        prediction = captured[0]
        pred_lines = _get_stripped_lines(prediction)
        expected_lines = _get_stripped_lines(expected_out)

        if len(pred_lines) != len(expected_lines):
            all_results.append(-2)
            return all_results, {"error_code": -2, "error_message": "Wrong answer: mismatched output length"}

        line_match = True
        for pred_line, exp_line in zip(pred_lines, expected_lines):
            if pred_line == exp_line:
                continue
            # Try decimal comparison for floating point
            try:
                pred_decimals = [Decimal(x) for x in pred_line.split()]
                exp_decimals = [Decimal(x) for x in exp_line.split()]
                if pred_decimals == exp_decimals:
                    continue
            except Exception:
                pass
            line_match = False
            break

        if not line_match:
            all_results.append(-2)
            return all_results, {"error_code": -2, "error_message": "Wrong Answer"}

        all_results.append(True)

    return all_results, {"execution_time": total_time}


def _reliability_guard():
    """Disable destructive operations in the evaluation subprocess."""
    import builtins
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def evaluate_solution(
    problem: LCBProblem,
    code: str,
    timeout: int = 10,
    use_private_tests: bool = True,
) -> dict:
    """
    Evaluate a code solution against test cases in a sandboxed subprocess.

    Args:
        problem: The problem to evaluate against
        code: The generated Python code
        timeout: Per-test-case timeout in seconds
        use_private_tests: If True, use all tests; if False, only public tests

    Returns:
        Dict with keys:
        - passed: bool (all tests passed)
        - results: list of per-test results (True or error code)
        - metadata: dict with error details if any
        - num_tests: total test cases
        - num_passed: number passing
    """
    if not code.strip():
        return {
            "passed": False,
            "results": [],
            "metadata": {"error": "Empty code submission"},
            "num_tests": 0,
            "num_passed": 0,
        }

    if use_private_tests:
        sample = problem.get_evaluation_sample()
    else:
        sample = problem.get_public_evaluation_sample()

    in_outs = json.loads(sample["input_output"])
    num_tests = len(in_outs["inputs"])

    # Run in subprocess for isolation
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    p = ctx.Process(
        target=_run_test_worker,
        args=(sample, code, timeout, result_queue),
    )
    p.start()
    # Global timeout: per-test timeout * num_tests + overhead
    global_timeout = (timeout + 1) * num_tests + 10
    p.join(timeout=global_timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=5)
        return {
            "passed": False,
            "results": [-1] * num_tests,
            "metadata": {"error_code": -1, "error_message": "Global timeout"},
            "num_tests": num_tests,
            "num_passed": 0,
        }

    if result_queue.empty():
        return {
            "passed": False,
            "results": [-1] * num_tests,
            "metadata": {"error_code": -1, "error_message": "Worker process crashed"},
            "num_tests": num_tests,
            "num_passed": 0,
        }

    status, results, metadata = result_queue.get()

    num_passed = sum(1 for r in results if r is True or (isinstance(r, (int, float)) and r > 0))
    passed = all(r is True or (isinstance(r, (int, float)) and r > 0) for r in results)

    return {
        "passed": passed,
        "results": results,
        "metadata": metadata,
        "num_tests": num_tests,
        "num_passed": num_passed,
    }


def evaluate_solutions_batch(
    problems: list[LCBProblem],
    codes: list[str],
    timeout: int = 10,
    num_workers: int = 4,
) -> list[dict]:
    """
    Evaluate multiple solutions in parallel.

    Args:
        problems: List of problems
        codes: List of code solutions (same order as problems)
        timeout: Per-test-case timeout
        num_workers: Number of parallel evaluation workers

    Returns:
        List of evaluation result dicts
    """
    results = []
    for prob, code in zip(problems, codes):
        result = evaluate_solution(prob, code, timeout=timeout)
        results.append(result)
    return results

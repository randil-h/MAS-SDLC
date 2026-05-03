"""
Test Engineer Agent — MAS SDLC Pipeline.

Persona  : Senior QA Engineer
Input    : state["generated_code"], state["requirements"]
Output   : pytest suite + structured execution results
"""

import ast
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from langchain_ollama import OllamaLLM

from state import SDLCState
from tools.code_tools import strip_markdown_fences
from tools.file_tools import read_from_file, save_to_file, append_log
from tools.execution_tools import run_pytest


MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

CODE_PATH = "output/generated_code.py"
TEST_PATH = "output/test_suite.py"
RESULT_PATH = "output/test_results.json"

AGENT_NAME = "TestEngineerAgent"


SYSTEM_PROMPT = """
You are a senior QA Engineer.

Write a COMPLETE pytest test suite for the provided Python code.

RULES:
- Output ONLY raw Python code.
- Import module as:
    from generated_code import *
- Cover:
    1. Happy path scenarios
    2. Edge cases from requirements
    3. At least 2 negative tests
- Every test must include a docstring
- Use pytest only (no external libraries).
- ALWAYS add `import pytest` at the top when you use @pytest.fixture, pytest.raises,
  pytest.mark, or any other pytest API.
- If you use `time.sleep` or datetime helpers, include the matching imports (`import time`, etc.).
- If you reference `hashlib`, `datetime`, etc., add matching imports (`import hashlib`, etc.).
- Name tests like:
    test_<behavior>_<condition>
"""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def ensure_test_helper_imports(code: str) -> str:
    """
    Models often omit imports; inserting them avoids NameError during collection.

    Imports are inserted **after** `from generated_code import ...` when present so that
    ``import pytest`` survives after `from generated_code import *` (star-import could
    otherwise shadow pytest if it were placed above).
    """
    needs_pytest = bool(re.search(r"@pytest\b|\bpytest\.(raises|mark|fixture|parametrize)\b", code))
    needs_time = "time.sleep" in code
    needs_hashlib = bool(re.search(r"\bhashlib\.", code))

    def _has_import(pat: str) -> bool:
        return bool(re.search(pat, code, flags=re.MULTILINE))

    to_add: list[str] = []
    if needs_pytest and not _has_import(r"(?m)^(?:import\s+pytest\b|from\s+pytest\b)"):
        to_add.append("import pytest")
    if needs_time and not _has_import(r"(?m)^(?:import\s+time\b|from\s+time\b)"):
        to_add.append("import time")
    if needs_hashlib and not _has_import(r"(?m)^(?:import\s+hashlib\b|from\s+hashlib\b)"):
        to_add.append("import hashlib")

    if not to_add:
        return code

    lines = code.replace("\r\n", "\n").split("\n")
    anchor = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*from\s+generated_code\s+import\s+", line):
            anchor = i

    if anchor >= 0:
        merged = [*lines[: anchor + 1], *to_add, *lines[anchor + 1 :]]
        out = "\n".join(merged)
        return out + "\n" if code.endswith("\n") else out

    return "\n".join(to_add) + "\n" + code


def build_prompt(code: str, requirements: dict) -> str:
    requirements_json = json.dumps(requirements, indent=2)
    return f"""
{SYSTEM_PROMPT}

REQUIREMENTS:
{requirements_json}

CODE:
{code}
"""


def load_generated_code(state: SDLCState):
    try:
        code = read_from_file(CODE_PATH)
        return code, ["read_from_file generated_code.py"]
    except FileNotFoundError:
        return state.get("generated_code", ""), ["used state generated_code"]


def load_requirements(state: SDLCState):
    try:
        raw = read_from_file("output/requirements.json")
        return json.loads(raw), ["read requirements.json"]
    except Exception:
        return state.get("requirements", {}), ["used state requirements"]


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def parse_pytest_output(output: str):
    results = []
    failures = []

    for line in output.splitlines():
        if "::" in line and ("PASSED" in line or "FAILED" in line):
            parts = line.split("::")
            test_name = parts[-1].split()[0]
            status = "PASSED" if "PASSED" in line else "FAILED"

            results.append({
                "test": test_name,
                "status": status
            })

        if "AssertionError" in line or "ValueError" in line:
            failures.append(line.strip())

    return results, failures


def map_test_to_requirement(test_name: str, requirements: dict) -> str:
    for req in requirements.get("functional_requirements", []):
        if any(word in test_name.lower() for word in req.lower().split()):
            return req
    return "General behavior"


# -------------------------------------------------------------------
# Main Agent
# -------------------------------------------------------------------

def test_engineer_node(state: SDLCState) -> SDLCState:

    errors = list(state.get("errors", []))
    tool_calls = []

    test_results = {}

    try:
        model = os.environ.get("OLLAMA_MODEL", "phi3:mini")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # Load inputs
        code, calls = load_generated_code(state)
        tool_calls.extend(calls)

        if not code:
            errors.append("No generated code found")
            return {**state, "errors": errors}

        requirements, calls = load_requirements(state)
        tool_calls.extend(calls)

        # LLM
        prompt = build_prompt(code, requirements)
        llm = OllamaLLM(model=model, base_url=base_url)
        tool_calls.append("Ollama invoked")

        raw_tests = llm.invoke(prompt)

        # Clean
        generated_tests = strip_markdown_fences(raw_tests)

        if "from generated_code import" in generated_tests:
            generated_tests = generated_tests.replace(
                "from generated_code import *",
                "import sys\nsys.path.append('output')\nfrom generated_code import *"
            )

        generated_tests = ensure_test_helper_imports(generated_tests)

        if not is_valid_python(generated_tests):
            raise Exception("Generated test code is not valid Python")

        # Save tests
        save_to_file(TEST_PATH, generated_tests)
        tool_calls.append("saved test file")

        # Run pytest
        raw_results = run_pytest(TEST_PATH)
        tool_calls.append("pytest executed")

        parsed_tests, failure_lines = parse_pytest_output(raw_results["output"])

        # Coverage
        coverage = [
            {
                "test": t["test"],
                "status": t["status"],
                "mapped_requirement": map_test_to_requirement(t["test"], requirements)
            }
            for t in parsed_tests
        ]

        # Failures
        structured_failures = [
            {
                "issue": line,
                "impact": "Mismatch between expected and actual behavior"
            }
            for line in failure_lines[:5]
        ]

        # Analysis
        analysis = {
            "root_causes": [],
            "test_quality_issues": []
        }

        rc = raw_results.get("exit_code")

        if raw_results["errors"] > 0 or (
            isinstance(rc, int) and rc not in (0, 5) and raw_results["passed"] == 0
        ):
            analysis["root_causes"].append(
                "pytest aborted during collection/setup or crashed — inspect raw_output"
            )

        if raw_results["failed"] > 0:
            analysis["root_causes"].append("Core functionality not aligned with requirements")

        if "assert False" in raw_results["output"]:
            analysis["test_quality_issues"].append("Generated test contains 'assert False'")

        if "ValueError" in raw_results["output"]:
            analysis["root_causes"].append("Exception handling mismatch")

        failed_summary = raw_results["failed"] > 0 or raw_results["errors"] > 0
        if isinstance(rc, int) and rc != 0:
            failed_summary = True

        summary_status = "FAILED" if failed_summary else "PASSED"

        # Final structured result
        test_results = {
            "summary": {
                "total_tests": len(parsed_tests),
                "passed": raw_results["passed"],
                "failed": raw_results["failed"],
                "errors": raw_results["errors"],
                "exit_code": rc,
                "status": summary_status,
            },
            "coverage": coverage,
            "failures": structured_failures,
            "analysis": analysis,
            "raw_output": raw_results["output"]
        }

        # Save results
        save_to_file(RESULT_PATH, json.dumps(test_results, indent=2))
        tool_calls.append("saved results")

    except Exception:
        full_error = traceback.format_exc()
        errors.append(f"{AGENT_NAME} error:\n{full_error}")
        print(full_error, file=sys.stderr)

    # Logging
    append_log(
        state["log_path"],
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": AGENT_NAME,
            "tool_calls": tool_calls,
            "output": test_results,
        },
    )

    return {
        **state,
        "test_results": test_results,
        "errors": errors,
    }
"""
Test Engineer Agent — MAS SDLC Pipeline.

Persona  : Senior QA Engineer
Input    : state["generated_code"], state["requirements"]
Output   : pytest suite + structured execution results
"""

import json
import os
import sys
import ast
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
- Use pytest only (no external libraries)
- Name tests like:
    test_<behavior>_<condition>
"""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

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

        if raw_results["failed"] > 0:
            analysis["root_causes"].append("Core functionality not aligned with requirements")

        if "assert False" in raw_results["output"]:
            analysis["test_quality_issues"].append("Generated test contains 'assert False'")

        if "ValueError" in raw_results["output"]:
            analysis["root_causes"].append("Exception handling mismatch")

        # Final structured result
        test_results = {
            "summary": {
                "total_tests": len(parsed_tests),
                "passed": raw_results["passed"],
                "failed": raw_results["failed"],
                "errors": raw_results["errors"],
                "status": "FAILED" if raw_results["failed"] > 0 else "PASSED"
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
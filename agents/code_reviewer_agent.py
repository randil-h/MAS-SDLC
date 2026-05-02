"""
Code Reviewer Agent — MAS SDLC Pipeline.

Agent persona : Staff Engineer
Input         : state["generated_code"], state["test_results"], state["requirements"]
Output        : state["review_report"] (str)

The agent runs flake8 static analysis on the generated code, then calls a
locally-hosted Ollama LLM to produce a structured Markdown code review report
with 6 sections: Summary, Requirements Coverage, Code Quality, Security & Edge
Cases, Test Coverage Assessment, and Actionable Suggestions.

Tools used    : run_static_analysis, read_from_file,
                save_to_file → output/review_report.md, append_log
"""

import json
import os
import sys
from datetime import datetime, timezone

from langchain_ollama import OllamaLLM as Ollama

from state import SDLCState
from tools.analysis_tools import run_static_analysis
from tools.file_tools import append_log, read_from_file, save_to_file
from tools.review_tools import parse_review_sections

# ---------------------------------------------------------------------------
# Constants — overridable via environment variables so that runtime config
# passed through the API layer is honoured without restarting the process.
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_CODE_PATH = "output/generated_code.py"
_REQUIREMENTS_PATH = "output/requirements.json"
_REVIEW_OUTPUT_PATH = "output/review_report.md"
_AGENT_NAME = "CodeReviewerAgent"

# GPU / memory tuning — same defaults as the other agents.
_NUM_GPU = 99
_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Staff Engineer conducting a thorough code review.
You will be given:
  1. A structured requirements document (JSON)
  2. Generated Python source code
  3. Test execution results (JSON)
  4. Static analysis findings from flake8

Your task is to produce a structured Markdown code review report.

CRITICAL RULES:
1. Output ONLY valid Markdown. No extra prose before the first heading.
2. The report MUST contain exactly these 6 sections in this order:

## Summary
A 2-4 sentence overall verdict. State clearly whether the code PASSES, FAILS,
or CONDITIONALLY PASSES the review, and give the single most important reason.

## Requirements Coverage
For each functional requirement listed in the requirements document, state
whether the code implements it (YES / PARTIAL / NO) and a one-sentence
justification.

## Code Quality
Evaluate readability, type hints, docstrings, naming conventions, and overall
structure. Reference specific line numbers or function names where relevant.

## Security & Edge Cases
Identify missing input validation, unhandled exceptions, potential security
issues (injection, path traversal, etc.), and any edge cases from the
requirements that are not handled.

## Test Coverage Assessment
Based on the test results and the test output provided, assess whether the
test suite adequately exercises the code. Note any obvious gaps.

## Actionable Suggestions
A numbered list of concrete, specific improvements the developer should make.
Each item must be actionable (e.g. "Add a ValueError check for negative input
in calculate_discount() at line 42") rather than generic advice.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_generated_code(state: SDLCState) -> tuple[str, list[str]]:
    """
    Load generated Python code from disk, falling back to state.

    Returns
    -------
    tuple[str, list[str]]
        (code_string, tool_calls_log)
    """
    tool_calls: list[str] = []

    try:
        code = read_from_file(_CODE_PATH)
        tool_calls.append(f"read_from_file('{_CODE_PATH}') -> success")
        return code, tool_calls
    except FileNotFoundError:
        tool_calls.append(
            f"read_from_file('{_CODE_PATH}') -> FileNotFoundError; "
            "falling back to state['generated_code']"
        )

    code = state.get("generated_code") or ""
    if not code:
        tool_calls.append("state['generated_code'] -> empty")
    else:
        tool_calls.append("state['generated_code'] -> success (fallback)")

    return code, tool_calls


def _load_requirements(state: SDLCState) -> tuple[dict, list[str]]:
    """
    Load requirements from disk, falling back to state.

    Returns
    -------
    tuple[dict, list[str]]
        (requirements_dict, tool_calls_log)
    """
    tool_calls: list[str] = []

    try:
        raw = read_from_file(_REQUIREMENTS_PATH)
        requirements = json.loads(raw)
        tool_calls.append(f"read_from_file('{_REQUIREMENTS_PATH}') -> success")
        return requirements, tool_calls
    except FileNotFoundError:
        tool_calls.append(
            f"read_from_file('{_REQUIREMENTS_PATH}') -> FileNotFoundError; "
            "falling back to state['requirements']"
        )
    except json.JSONDecodeError as exc:
        tool_calls.append(
            f"read_from_file('{_REQUIREMENTS_PATH}') -> JSONDecodeError: {exc}; "
            "falling back to state['requirements']"
        )

    requirements = state.get("requirements") or {}
    if not requirements:
        tool_calls.append("state['requirements'] -> empty; using empty dict as last resort")
    else:
        tool_calls.append("state['requirements'] -> success (fallback)")

    return requirements, tool_calls


def _build_prompt(
    requirements: dict,
    generated_code: str,
    test_results: dict,
    static_analysis: dict,
) -> str:
    """
    Assemble the full review prompt from its four input sources.

    Parameters
    ----------
    requirements : dict
        Structured requirements document from the Requirements Agent.
    generated_code : str
        Python source code produced by the Code Generator Agent.
    test_results : dict
        pytest execution summary from the Test Engineer Agent.
    static_analysis : dict
        flake8 findings from run_static_analysis().

    Returns
    -------
    str
        Complete prompt string to send to the LLM.
    """
    requirements_block = json.dumps(requirements, indent=2) if requirements else "(not available)"

    test_summary = (
        f"Passed: {test_results.get('passed', 'N/A')}, "
        f"Failed: {test_results.get('failed', 'N/A')}, "
        f"Errors: {test_results.get('errors', 'N/A')}"
    )
    test_output_excerpt = str(test_results.get("output", "(no output)"))[:1000]

    flake8_issues = static_analysis.get("issues", [])
    if flake8_issues:
        flake8_block = "\n".join(flake8_issues)
    else:
        flake8_block = "No issues found."

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        "---\n\n"
        "### Requirements Document\n\n"
        f"```json\n{requirements_block}\n```\n\n"
        "### Generated Code\n\n"
        f"```python\n{generated_code}\n```\n\n"
        "### Test Results\n\n"
        f"{test_summary}\n\n"
        f"Test output (first 1000 chars):\n```\n{test_output_excerpt}\n```\n\n"
        "### Static Analysis (flake8)\n\n"
        f"```\n{flake8_block}\n```\n\n"
        "---\n\n"
        "Now write the code review report following the 6-section structure above."
    )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def code_reviewer_node(state: SDLCState) -> SDLCState:
    """
    LangGraph node: review generated code and produce a structured Markdown report.

    Workflow
    --------
    1. Load generated code from ``output/generated_code.py`` (falls back to state).
    2. Load requirements from ``output/requirements.json`` (falls back to state).
    3. Read test results from state.
    4. Run flake8 static analysis via ``run_static_analysis``.
    5. Build a prompt combining all four inputs.
    6. Invoke the Ollama LLM to generate the review report.
    7. Save the report to ``output/review_report.md``.
    8. Append a structured log entry to the run log.
    9. Return the updated state with ``review_report`` populated.

    Parameters
    ----------
    state : SDLCState
        Current pipeline state. Reads ``generated_code``, ``test_results``,
        and ``requirements``.

    Returns
    -------
    SDLCState
        Updated state with ``review_report`` set to the Markdown report string.
        On failure, ``errors`` is updated and ``review_report`` is None.
    """
    errors: list[str] = list(state.get("errors") or [])
    tool_calls: list[str] = []
    review_report: str = ""

    try:
        # Step 1 — Load generated code
        generated_code, code_tool_calls = _load_generated_code(state)
        tool_calls.extend(code_tool_calls)

        if not generated_code:
            error_msg = (
                f"[{_AGENT_NAME}] No generated code available from file or state. "
                "Cannot conduct review."
            )
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            return {**state, "review_report": None, "errors": errors}

        # Step 2 — Load requirements
        requirements, req_tool_calls = _load_requirements(state)
        tool_calls.extend(req_tool_calls)

        # Step 3 — Read test results from state (produced by Test Engineer)
        test_results: dict = state.get("test_results") or {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "output": "(test results not available)",
        }
        tool_calls.append(
            f"state['test_results'] -> "
            f"passed={test_results.get('passed')}, "
            f"failed={test_results.get('failed')}, "
            f"errors={test_results.get('errors')}"
        )

        # Step 4 — Run flake8 static analysis
        static_analysis = run_static_analysis(_CODE_PATH)
        tool_calls.append(
            f"run_static_analysis('{_CODE_PATH}') -> "
            f"issue_count={static_analysis.get('issue_count', 0)}"
        )

        # Step 5 — Build prompt
        prompt = _build_prompt(requirements, generated_code, test_results, static_analysis)

        # Step 6 — Instantiate LLM (read env at call-time to honour API config)
        model = os.environ.get("OLLAMA_MODEL", _MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", _BASE_URL)
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", str(_NUM_CTX)))

        llm = Ollama(
            model=model,
            base_url=base_url,
            num_gpu=_NUM_GPU,
            num_ctx=num_ctx,
        )
        tool_calls.append(
            f"Ollama.invoke(model='{model}', base_url='{base_url}', "
            f"num_gpu={_NUM_GPU}, num_ctx={num_ctx})"
        )

        review_report = llm.invoke(prompt)
        tool_calls.append(f"Ollama response received — length={len(review_report)} chars")

        parsed = parse_review_sections(review_report)
        tool_calls.append(
            f"parse_review_sections() -> is_complete={parsed.is_complete}, "
            f"verdict='{parsed.verdict}', "
            f"missing={parsed.missing_sections}"
        )
        if not parsed.is_complete:
            errors.append(
                f"[{_AGENT_NAME}] Review report is missing sections: "
                f"{parsed.missing_sections}"
            )

        # Step 7 — Save report to disk
        save_success = save_to_file(_REVIEW_OUTPUT_PATH, review_report)
        tool_calls.append(
            f"save_to_file('{_REVIEW_OUTPUT_PATH}') -> "
            f"{'success' if save_success else 'FAILED'}"
        )
        if not save_success:
            errors.append(
                f"[{_AGENT_NAME}] Failed to save review report to '{_REVIEW_OUTPUT_PATH}'."
            )

    except Exception as exc:
        error_msg = f"[{_AGENT_NAME}] Unexpected error during code review: {exc}"
        print(error_msg, file=sys.stderr)
        errors.append(error_msg)
        tool_calls.append(f"EXCEPTION: {exc}")

    # Step 8 — Observability log
    append_log(
        state.get("log_path", "logs/run.json"),
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": _AGENT_NAME,
            "model": os.environ.get("OLLAMA_MODEL", _MODEL),
            "base_url": os.environ.get("OLLAMA_BASE_URL", _BASE_URL),
            "input": {
                "has_generated_code": bool(state.get("generated_code")),
                "has_requirements": bool(state.get("requirements")),
                "has_test_results": bool(state.get("test_results")),
            },
            "tool_calls": tool_calls,
            "output": {
                "report_length": len(review_report),
                "saved_to": _REVIEW_OUTPUT_PATH,
                "errors": errors,
            },
        },
    )

    # Step 9 — Return updated state
    return {
        **state,
        "review_report": review_report if review_report else None,
        "errors": errors,
    }

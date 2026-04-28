"""
Code Generator Agent — MAS SDLC Pipeline.

This agent receives a structured requirements document and uses a locally-hosted
Ollama LLM to produce clean, type-hinted, well-documented Python source code that
satisfies every listed requirement and edge case.

Agent persona : Expert Python Developer
Input         : state["requirements"] (dict)
Output        : state["generated_code"] (str)
Tools used    : read_from_file, save_to_file, append_log

IT22240088
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_ollama import OllamaLLM

from state import SDLCState
from tools.code_tools import ValidationResult, strip_markdown_fences, validate_python_code
from tools.file_tools import append_log, read_from_file, save_to_file

# ---------------------------------------------------------------------------
# Constants — can be overridden via environment variables so API-provided
# runtime configuration is honored without restarting the process.
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_REQUIREMENTS_PATH = "output/requirements.json"
_CODE_OUTPUT_PATH = "output/generated_code.py"
_AGENT_NAME = "CodeGeneratorAgent"

# GPU / memory tuning
# num_gpu=99  → send all layers to Metal GPU (Apple Silicon) / CUDA (NVIDIA).
#               Falls back to CPU automatically if the GPU can't fit the model.
# num_ctx     → context window in tokens.  Smaller = less VRAM/RAM pressure.
#               Default Ollama value is 2048; override via OLLAMA_NUM_CTX.
_NUM_GPU = 99
_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
# Base token budget; each retry multiplies this by _RETRY_PREDICT_MULTIPLIER.
# 2048 avoids the most common single-function truncation; retries escalate to
# 4096 and 8192 for larger generated files.
_NUM_PREDICT_BASE = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))
_RETRY_PREDICT_MULTIPLIER = 2   # double budget on every retry
_MAX_RETRIES = 2                # up to 3 total attempts (2048 → 4096 → 8192)

_SYSTEM_PROMPT = """You are an expert Python developer. You will receive a structured requirements document and must write clean, working Python code that fulfills all requirements.

STRICT RULES — follow every one, no exceptions:
1. Output ONLY raw Python source code. Absolutely no markdown, no ```python fences, no explanations, no comments outside the code itself.
2. Use type hints on ALL function and method signatures (arguments AND return type).
3. Write a docstring for every function and class.
4. Handle all edge cases listed in the requirements.
5. Do NOT use any external libraries — only Python's standard library.
6. NEVER place bare function calls or any executable statements at the top level of the module. All executable code MUST be inside a function/class definition or inside an `if __name__ == "__main__":` block. Violating this rule is a critical error.
7. Prefer clear, readable code over clever one-liners.
8. Validate inputs at the start of each public function and raise appropriate built-in exceptions (ValueError, TypeError, etc.) with descriptive messages.
9. Finish writing the COMPLETE code before stopping — do not truncate mid-string, mid-function, or mid-class.

SYNTAX RULES — these are the most common mistakes; avoid them:
- NEVER use f-strings with complex nested calls. Instead assign intermediate values to variables first.
  WRONG:  cursor.execute(f"... {datetime.strptime(s, fmt).isoformat()}")
  RIGHT:  ts = datetime.strptime(s, fmt).isoformat(); cursor.execute(f"... {ts}")
- NEVER mix SQL 'AS' keyword syntax inside Python expressions.
  WRONG:  len(db_file as sqlite3)
  RIGHT:  with sqlite3.connect(db_file) as conn:
- NEVER leave a string literal open at the end of a line. Every opening quote must have a matching closing quote on the same line (or use triple-quotes for multi-line strings).
- Count your parentheses. Every '(' must have a matching ')' before the line ends."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_requirements(state: SDLCState) -> tuple[dict, list[str]]:
    """
    Load the requirements dict from disk, falling back to state if the file is absent.

    Parameters
    ----------
    state : SDLCState
        Current pipeline state, used as a fallback source for requirements.

    Returns
    -------
    tuple[dict, list[str]]
        A tuple of (requirements_dict, tool_calls_log) where tool_calls_log
        records which data source was actually used.
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

    # Fallback: use what is already in state
    requirements = state.get("requirements") or {}
    if not requirements:
        tool_calls.append("state['requirements'] -> empty; using empty dict as last resort")
    else:
        tool_calls.append("state['requirements'] -> success (fallback)")

    return requirements, tool_calls


def _build_prompt(requirements: dict) -> str:
    """
    Construct the full prompt string to send to the LLM.

    Parameters
    ----------
    requirements : dict
        Structured requirements document from the Requirements Agent.

    Returns
    -------
    str
        The complete prompt combining system instructions and requirements JSON.
    """
    requirements_json = json.dumps(requirements, indent=2)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        "Implement the following requirements as a single valid Python module.\n\n"
        f"{requirements_json}"
    )


# ---------------------------------------------------------------------------
# Truncation repair helper
# ---------------------------------------------------------------------------


def _repair_truncated_code(code: str, error_lineno: int) -> str:
    """
    Salvage the longest syntactically valid prefix of LLM-generated code.

    When the model runs out of its token budget it stops mid-statement,
    leaving the tail of the source broken.  This function walks backward
    from the reported error line, trying progressively shorter prefixes
    until ``ast.parse`` succeeds.

    The repaired prefix is incomplete but valid Python — the pipeline can
    still save it, the reviewer agent will flag the missing implementation,
    and the operator can re-run with a larger ``OLLAMA_NUM_PREDICT`` budget.

    Parameters
    ----------
    code : str
        The full (broken) source string from the LLM.
    error_lineno : int
        1-based line number of the syntax error, as reported by Python's
        ``ast.parse``.

    Returns
    -------
    str
        The longest syntactically valid prefix, or an empty string when no
        valid prefix could be found.
    """
    import ast as _ast

    lines = code.splitlines()
    # Start just above the error line and walk backward
    start = min(error_lineno - 1, len(lines))
    for cutoff in range(start, -1, -1):
        candidate = "\n".join(lines[:cutoff]).strip()
        if not candidate:
            continue
        try:
            _ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return ""


# ---------------------------------------------------------------------------
# Post-processing helper
# ---------------------------------------------------------------------------


def _wrap_top_level_calls(code: str) -> str:
    """
    Move stray top-level executable expression statements into an
    ``if __name__ == "__main__":`` guard.

    The LLM occasionally emits bare function calls (e.g. ``run()``,
    ``main()``) at module scope despite instructions not to.  This helper
    detects them via the AST and rewrites the source so those lines are
    indented under the guard, preventing import-time side-effects and the
    associated ``validate_python_code`` warnings.

    If the code cannot be parsed (a syntax error already exists) the
    original string is returned unchanged so the caller can handle the
    failure gracefully.

    Parameters
    ----------
    code : str
        Raw Python source code, possibly containing top-level calls.

    Returns
    -------
    str
        Rewritten source with offending top-level calls moved into the
        ``__main__`` guard, or the original string if no fix was needed /
        possible.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code  # cannot fix; let validation surface the error

    lines = code.splitlines()
    call_linenos: set[int] = set()

    for node in tree.body:
        if (
            isinstance(node, _ast.Expr)
            and not isinstance(node.value, _ast.Constant)
        ):
            # Collect every source line that belongs to this expression
            end = getattr(node, "end_lineno", node.lineno)
            for ln in range(node.lineno, end + 1):
                call_linenos.add(ln)

    if not call_linenos:
        return code  # nothing to fix

    kept: list[str] = []
    extracted: list[str] = []

    for i, line in enumerate(lines, start=1):
        if i in call_linenos:
            extracted.append("    " + line)
        else:
            kept.append(line)

    # Append the __main__ guard (or merge into existing one if present)
    main_guard = 'if __name__ == "__main__":'
    if any(main_guard in l for l in kept):
        # Find insertion point — just before the existing guard's body ends
        # Simple approach: append to the very end of the file
        kept.append("")
        kept.extend(extracted)
    else:
        kept.append("")
        kept.append(main_guard)
        kept.extend(extracted)

    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def code_generator_node(state: SDLCState) -> SDLCState:
    """
    LangGraph node: generate Python source code from structured requirements.

    Workflow
    --------
    1. Load requirements from ``output/requirements.json`` (falls back to state).
    2. Build a detailed prompt combining the system persona and requirements JSON.
    3. Invoke the Ollama LLM to generate Python code.
    4. Strip accidental markdown fences via ``strip_markdown_fences``.
    5. Validate syntax and quality via ``validate_python_code``.
    6. Save the code to ``output/generated_code.py`` (saved even if invalid so reviewers can inspect it).
    7. Append a structured log entry to the run log.
    8. Return the updated state with ``generated_code`` populated.

    Parameters
    ----------
    state : SDLCState
        Current pipeline state. Reads ``requirements`` and ``log_path``.

    Returns
    -------
    SDLCState
        Updated state with ``generated_code`` set to the generated Python source.
    """
    errors: list[str] = list(state.get("errors") or [])
    tool_calls: list[str] = []
    generated_code: str = ""
    requirements: dict = {}  # initialised early so append_log can always reference it

    try:
        # Step 1 — Load requirements
        requirements, load_tool_calls = _load_requirements(state)
        tool_calls.extend(load_tool_calls)

        if not requirements:
            error_msg = (
                f"[{_AGENT_NAME}] No requirements available from file or state. "
                "Cannot generate code."
            )
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            return {**state, "errors": errors}

        # Step 2 — Build prompt
        prompt = _build_prompt(requirements)

        # Step 3 — Call LLM with retry on syntax failure (truncation recovery)
        model    = os.environ.get("OLLAMA_MODEL",   _MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", _BASE_URL)
        num_ctx  = int(os.environ.get("OLLAMA_NUM_CTX", str(_NUM_CTX)))

        validation: ValidationResult = ValidationResult()
        for attempt in range(_MAX_RETRIES + 1):
            num_predict = _NUM_PREDICT_BASE * (_RETRY_PREDICT_MULTIPLIER ** attempt)
            llm = OllamaLLM(
                model=model,
                base_url=base_url,
                num_gpu=_NUM_GPU,
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=0,
            )
            tool_calls.append(
                f"Ollama.invoke(attempt={attempt + 1}, model='{model}', "
                f"base_url='{base_url}', num_gpu={_NUM_GPU}, "
                f"num_ctx={num_ctx}, num_predict={num_predict})"
            )

            raw_response: str = llm.invoke(prompt)

            # Step 4 — Strip markdown fences the LLM may have added
            generated_code = strip_markdown_fences(raw_response)
            tool_calls.append(f"strip_markdown_fences() -> applied (attempt {attempt + 1})")

            # Auto-wrap any stray top-level calls into an __main__ guard
            generated_code = _wrap_top_level_calls(generated_code)

            # Step 5 — Validate syntax and code quality
            validation = validate_python_code(generated_code)
            tool_calls.append(
                f"validate_python_code() -> valid={validation.is_valid}, "
                f"functions={validation.function_count}, "
                f"type_hints={validation.has_type_hints}, "
                f"warnings={len(validation.warnings)} (attempt {attempt + 1})"
            )

            if validation.is_valid:
                break  # success — no need to retry

            # Syntax failure: log and retry with a larger token budget
            retry_msg = (
                f"[{_AGENT_NAME}] Attempt {attempt + 1} failed syntax validation "
                f"(num_predict={num_predict}): {validation.syntax_error}"
            )
            print(retry_msg, file=sys.stderr)
            tool_calls.append(retry_msg)

        if not validation.is_valid:
            # Last-resort: try to salvage a syntactically valid prefix so the
            # pipeline can continue with partial (but parseable) code.
            repaired = _repair_truncated_code(generated_code, validation.syntax_error_line)
            if repaired:
                repaired_validation = validate_python_code(repaired)
                if repaired_validation.is_valid:
                    generated_code = repaired
                    validation = repaired_validation
                    tool_calls.append(
                        f"_repair_truncated_code() -> salvaged {len(repaired)} chars "
                        f"(truncated at line {validation.syntax_error_line})"
                    )
                    errors.append(
                        f"[{_AGENT_NAME}] WARNING: Code was truncated by the LLM and "
                        "salvaged as a partial implementation. Re-run with a larger "
                        "OLLAMA_NUM_PREDICT value for a complete result."
                    )

            if not validation.is_valid:
                error_msg = (
                    f"[{_AGENT_NAME}] LLM output failed syntax validation after "
                    f"{_MAX_RETRIES + 1} attempt(s) and repair: {validation.syntax_error}"
                )
                print(error_msg, file=sys.stderr)
                errors.append(error_msg)
                # Save the raw (invalid) output so the reviewer can inspect it
                save_to_file(_CODE_OUTPUT_PATH, generated_code)

        if validation.is_valid:
            # Demote top-level-call warnings to informational; they were auto-fixed above
            quality_warnings = [
                w for w in validation.warnings
                if "top-level expression statement" not in w
                and "__init__' is missing a docstring" not in w
                and "class '" not in w
                and "function 'main' is missing a docstring" not in w
            ]
            for w in quality_warnings:
                errors.append(f"[{_AGENT_NAME}] Code quality warning: {w}")

            # Step 6 — Persist validated code to disk
            save_success = save_to_file(_CODE_OUTPUT_PATH, generated_code)
            tool_calls.append(
                f"save_to_file('{_CODE_OUTPUT_PATH}') -> {'success' if save_success else 'FAILED'}"
            )
            if not save_success:
                errors.append(
                    f"[{_AGENT_NAME}] Failed to save generated code to '{_CODE_OUTPUT_PATH}'."
                )

    except Exception as exc:
        error_msg = f"[{_AGENT_NAME}] Unexpected error during code generation: {exc}"
        print(error_msg, file=sys.stderr)
        errors.append(error_msg)
        tool_calls.append(f"EXCEPTION: {exc}")

    # Step 7 — Observability log
    append_log(
        state.get("log_path", "logs/run.json"),
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": _AGENT_NAME,
            "model": os.environ.get("OLLAMA_MODEL", _MODEL),
            "base_url": os.environ.get("OLLAMA_BASE_URL", _BASE_URL),
            "input": {
                "requirements_keys": list(requirements.keys()) if requirements else [],
                "requirements_source": (
                    "file" if Path(_REQUIREMENTS_PATH).exists() else "state_fallback"
                ),
            },
            "tool_calls": tool_calls,
            "output": {
                "code_length": len(generated_code),
                "saved_to": _CODE_OUTPUT_PATH,
                "errors": errors,
            },
        },
    )

    # Step 8 — Return updated state
    return {
        **state,
        "generated_code": generated_code,
        "errors": errors,
    }

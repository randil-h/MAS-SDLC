"""
Code Generator Agent — MAS SDLC Pipeline.

This agent receives a structured requirements document and uses a locally-hosted
Ollama LLM to produce clean, type-hinted, well-documented Python source code that
satisfies every listed requirement and edge case.

Agent persona : Expert Python Developer
Input         : state["requirements"] (dict)
Output        : state["generated_code"] (str)
Tools used    : read_from_file, save_to_file, append_log
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.llms import Ollama

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

_SYSTEM_PROMPT = """You are an expert Python developer. You will receive a structured requirements document and must write clean, working Python code that fulfills all requirements.

Rules:
- Output ONLY raw Python code. No markdown. No ```python fences. No explanations.
- Use type hints on all functions and method signatures.
- Write a docstring for every function and class.
- Handle all edge cases listed in the requirements.
- Do not use any external libraries unless they are part of Python's standard library.
- Structure the code as a module (functions/classes only, no top-level executable code except an optional `if __name__ == "__main__"` demo block).
- Prefer clear, readable code over clever one-liners.
- Validate inputs at the start of each public function and raise appropriate built-in exceptions (ValueError, TypeError, etc.) with descriptive messages."""


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
        The complete prompt combining system instructions and the requirements JSON.
    """
    requirements_json = json.dumps(requirements, indent=2)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        "Here is the requirements document you must implement:\n\n"
        f"{requirements_json}"
    )


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

        # Step 3 — Call LLM (read env at invocation time to honor API request config)
        model    = os.environ.get("OLLAMA_MODEL",   _MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", _BASE_URL)
        num_ctx  = int(os.environ.get("OLLAMA_NUM_CTX", str(_NUM_CTX)))
        llm = Ollama(
            model=model,
            base_url=base_url,
            num_gpu=_NUM_GPU,   # push all transformer layers onto Metal / CUDA
            num_ctx=num_ctx,    # keep context window small to reduce memory pressure
        )
        tool_calls.append(
            f"Ollama.invoke(model='{model}', base_url='{base_url}', "
            f"num_gpu={_NUM_GPU}, num_ctx={num_ctx})"
        )

        raw_response: str = llm.invoke(prompt)

        # Step 4 — Strip markdown fences the LLM may have added
        generated_code = strip_markdown_fences(raw_response)
        tool_calls.append("strip_markdown_fences() -> applied")

        # Step 5 — Validate syntax and code quality before saving
        validation: ValidationResult = validate_python_code(generated_code)
        tool_calls.append(
            f"validate_python_code() -> valid={validation.is_valid}, "
            f"functions={validation.function_count}, "
            f"type_hints={validation.has_type_hints}, "
            f"warnings={len(validation.warnings)}"
        )

        if not validation.is_valid:
            error_msg = (
                f"[{_AGENT_NAME}] LLM output failed syntax validation: "
                f"{validation.syntax_error}"
            )
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            # Still save the raw (invalid) output so the reviewer can inspect it
            save_to_file(_CODE_OUTPUT_PATH, generated_code)
        else:
            if validation.warnings:
                for w in validation.warnings:
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

    # Step 6 — Observability log
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

    # Step 7 — Return updated state
    return {
        **state,
        "generated_code": generated_code,
        "errors": errors,
    }

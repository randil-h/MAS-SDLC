"""
Requirements Analyst Agent — MAS SDLC Pipeline.

Agent persona : Senior Business Analyst + Software Architect
Input         : state["user_prompt"] (str)
Output        : state["requirements"] (dict)

The agent calls a locally-hosted Ollama LLM to transform a natural-language
feature request into a structured JSON requirements document with 7 fields:
  feature_name, description, functional_requirements, edge_cases,
  constraints, input_spec, output_spec.

Tools used    : save_to_file → output/requirements.json, append_log
"""

import json
import os
import sys
from datetime import datetime, timezone

from langchain_ollama import OllamaLLM

from state import SDLCState
from tools.code_tools import strip_markdown_fences
from tools.file_tools import append_log, save_to_file

# ---------------------------------------------------------------------------
# Constants — overridable via environment variables so that runtime config
# passed through the API layer is honoured without restarting the process.
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_REQUIREMENTS_OUTPUT_PATH = "output/requirements.json"
_AGENT_NAME = "RequirementsAgent"
_NUM_GPU = 99
_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))

# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to act as a Senior Business Analyst and
# return ONLY a strict JSON object. No prose, no markdown fences.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Senior Business Analyst and Software Architect.
Your task is to analyse a natural-language feature request and produce a structured requirements document.

CRITICAL RULES:
1. Respond with ONLY a single valid JSON object. No prose, no markdown, no code fences.
2. Do NOT include any text before or after the JSON object.
3. The JSON object MUST contain exactly these 7 keys:

{
  "feature_name": "<short name for the feature, max 60 chars>",
  "description": "<2-4 sentence description of what the module must do>",
  "functional_requirements": [
    "<requirement 1>",
    "<requirement 2>",
    "<requirement 3 — provide at least 4 items>"
  ],
  "edge_cases": [
    "<edge case 1>",
    "<edge case 2 — provide at least 3 items>"
  ],
  "constraints": [
    "<technical constraint 1>",
    "<technical constraint 2 — provide at least 3 items>"
  ],
  "input_spec": "<description of the expected inputs, their types, and validation rules>",
  "output_spec": "<description of the expected output format and return type>"
}

Be specific, technical, and precise. Base all requirements directly on the user's feature request.\
"""


# ---------------------------------------------------------------------------
# Helper — build the user-facing prompt
# ---------------------------------------------------------------------------


def _build_prompt(user_prompt: str) -> str:
    """
    Combine the system instructions with the user's feature request.

    Parameters
    ----------
    user_prompt : str
        The raw natural-language feature request from the user.

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        "Here is the feature request you must analyse:\n\n"
        f"{user_prompt}"
    )


# ---------------------------------------------------------------------------
# Helper — parse LLM response into a requirements dict
# ---------------------------------------------------------------------------


def _parse_requirements(raw_response: str) -> tuple[dict, str | None]:
    """
    Parse the LLM's raw string response into a validated requirements dict.

    Strips markdown code fences first (in case the model wraps in ```json...```),
    then attempts JSON parsing. Validates that all 7 required keys are present.

    Parameters
    ----------
    raw_response : str
        The raw string returned by the Ollama LLM.

    Returns
    -------
    tuple[dict, str | None]
        A tuple of (requirements_dict, error_message).
        On success, error_message is None.
        On failure, requirements_dict is empty and error_message describes the problem.
    """
    required_keys = {
        "feature_name",
        "description",
        "functional_requirements",
        "edge_cases",
        "constraints",
        "input_spec",
        "output_spec",
    }

    # Strip any markdown fences the LLM may have added
    cleaned = strip_markdown_fences(raw_response).strip()

    # Attempt JSON parsing
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {}, f"JSON parse error: {exc}. Raw response (first 500 chars): {cleaned[:500]}"

    if not isinstance(parsed, dict):
        return {}, f"Expected a JSON object, got {type(parsed).__name__}."

    missing_keys = required_keys - set(parsed.keys())
    if missing_keys:
        return {}, f"LLM response is missing required keys: {missing_keys}."

    return parsed, None


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def requirements_node(state: SDLCState) -> SDLCState:
    """
    LangGraph node: analyse user prompt and produce a structured requirements document.

    Workflow
    --------
    1. Read ``user_prompt`` from state.
    2. Build a prompt combining the Business Analyst system instructions and the prompt.
    3. Invoke the Ollama LLM to generate a JSON requirements document.
    4. Strip markdown fences and parse the JSON response.
    5. Validate that all 7 required keys are present.
    6. Save the requirements to ``output/requirements.json``.
    7. Append a structured log entry.
    8. Return the updated state with ``requirements`` populated.

    Parameters
    ----------
    state : SDLCState
        Current pipeline state. Reads ``user_prompt`` and ``log_path``.

    Returns
    -------
    SDLCState
        Updated state with ``requirements`` set to the parsed requirements dict.
        On LLM or parse failure, ``errors`` is updated and ``requirements`` is None.
    """
    errors: list[str] = list(state.get("errors") or [])
    tool_calls: list[str] = []
    requirements: dict = {}
    user_prompt: str = state.get("user_prompt", "")

    try:
        if not user_prompt:
            error_msg = f"[{_AGENT_NAME}] No user_prompt found in state. Cannot generate requirements."
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            return {**state, "requirements": None, "errors": errors}

        # Step 1 — Build prompt
        prompt = _build_prompt(user_prompt)

        # Step 2 — Instantiate LLM (read env at call-time to honour API config)
        model    = os.environ.get("OLLAMA_MODEL",    _MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", _BASE_URL)
        num_ctx  = int(os.environ.get("OLLAMA_NUM_CTX", str(_NUM_CTX)))

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

        # Step 3 — Call LLM
        raw_response: str = llm.invoke(prompt)
        tool_calls.append(f"Ollama response received — length={len(raw_response)} chars")

        # Step 4 — Parse and validate JSON response
        requirements, parse_error = _parse_requirements(raw_response)
        tool_calls.append(
            f"_parse_requirements() -> "
            f"{'success' if not parse_error else f'FAILED: {parse_error}'}"
        )

        if parse_error:
            error_msg = f"[{_AGENT_NAME}] Failed to parse LLM response: {parse_error}"
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            # Return without saving — the code generator will receive an empty requirements
            return {**state, "requirements": None, "errors": errors}

        # Step 5 — Save requirements to disk
        save_success = save_to_file(
            _REQUIREMENTS_OUTPUT_PATH,
            json.dumps(requirements, indent=2)
        )
        tool_calls.append(
            f"save_to_file('{_REQUIREMENTS_OUTPUT_PATH}') -> "
            f"{'success' if save_success else 'FAILED'}"
        )
        if not save_success:
            errors.append(
                f"[{_AGENT_NAME}] Failed to save requirements to '{_REQUIREMENTS_OUTPUT_PATH}'."
            )

    except Exception as exc:
        error_msg = f"[{_AGENT_NAME}] Unexpected error during requirements analysis: {exc}"
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
                "user_prompt": user_prompt,
            },
            "tool_calls": tool_calls,
            "output": {
                "requirements_keys": list(requirements.keys()) if requirements else [],
                "saved_to": _REQUIREMENTS_OUTPUT_PATH,
                "errors": errors,
            },
        },
    )

    # Step 7 — Return updated state
    return {
        **state,
        "requirements": requirements if requirements else None,
        "errors": errors,
    }

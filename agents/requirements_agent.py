"""
Requirements Analyst Agent — MAS SDLC Pipeline.

TODO (team member): Replace the DEV STUB below with the real implementation.
  - Persona  : Senior Business Analyst + Software Architect
  - Input    : state["user_prompt"]
  - LLM call : system prompt → strict JSON with 7 fields
  - Tools    : save_to_file → output/requirements.json, append_log
  - Output   : state["requirements"] (dict)
"""

import json

from state import SDLCState
from tools.file_tools import append_log, save_to_file


def requirements_node(state: SDLCState) -> SDLCState:
    """
    Analyse the user prompt and produce a structured requirements document.

    Parameters
    ----------
    state : SDLCState
        Current pipeline state. Reads ``user_prompt``.

    Returns
    -------
    SDLCState
        Updated state with ``requirements`` populated.
    """
    # ------------------------------------------------------------------ #
    # DEV STUB — returns dummy requirements derived from the user prompt.
    # Replace this entire function body with the real LLM implementation.
    # ------------------------------------------------------------------ #
    prompt = state.get("user_prompt", "generic feature")

    requirements = {
        "feature_name": prompt[:60],
        "description": (
            f"Implement a Python module that fulfils the following request: {prompt}. "
            "The module should be well-structured, handle edge cases, and follow "
            "Python best practices."
        ),
        "functional_requirements": [
            "Accept the primary input as described in the feature request.",
            "Validate all inputs before processing and raise appropriate exceptions.",
            "Return a well-defined output as described in the feature request.",
            "Handle all listed edge cases gracefully.",
            "Provide clear error messages for invalid inputs.",
        ],
        "edge_cases": [
            "Empty or None input should raise a ValueError or TypeError.",
            "Inputs at boundary values (e.g. zero, empty string) must be handled.",
            "Unexpected input types must raise TypeError with a descriptive message.",
        ],
        "constraints": [
            "Use only the Python standard library — no third-party packages.",
            "All functions must include type hints and docstrings.",
            "No top-level executable code outside an optional __main__ guard.",
        ],
        "input_spec": "Inputs as described in the feature request, with type validation.",
        "output_spec": "A well-typed return value or raised exception as appropriate.",
    }

    save_to_file("output/requirements.json", json.dumps(requirements, indent=2))
    append_log(
        state.get("log_path", "logs/run.json"),
        {
            "agent": "RequirementsAgent",
            "note": "DEV STUB — dummy requirements used",
            "input": {"user_prompt": prompt},
            "tool_calls": ["save_to_file('output/requirements.json')"],
            "output": requirements,
        },
    )

    return {**state, "requirements": requirements}

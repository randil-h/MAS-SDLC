"""
Code Reviewer Agent — MAS SDLC Pipeline.

TODO (team member): Implement code_reviewer_node.
  - Persona  : Staff Engineer
  - Input    : state["generated_code"], state["test_results"], state["requirements"]
  - LLM call : system prompt + code + requirements + test results + static analysis
               → Markdown report with 6 sections
  - Tools    : run_static_analysis, read_from_file,
               save_to_file → output/review_report.md, append_log
  - Output   : state["review_report"] (str)
"""

from state import SDLCState


def code_reviewer_node(state: SDLCState) -> SDLCState:
    """
    Conduct a thorough code review and produce a structured Markdown report.

    Parameters
    ----------
    state : SDLCState
        Current pipeline state. Reads ``generated_code``, ``test_results``,
        and ``requirements``.

    Returns
    -------
    SDLCState
        Updated state with ``review_report`` populated.
    """
    # ------------------------------------------------------------------ #
    # DEV STUB — returns a placeholder report so the pipeline can complete.
    # Replace this entire function body with the real LLM implementation.
    # ------------------------------------------------------------------ #
    review_report = (
        "## Summary\n\n"
        "> **DEV STUB** — Code Reviewer not yet implemented.\n\n"
        "## Requirements Coverage\n\n_Pending implementation._\n\n"
        "## Code Quality\n\n_Pending implementation._\n\n"
        "## Security & Edge Cases\n\n_Pending implementation._\n\n"
        "## Test Coverage Assessment\n\n_Pending implementation._\n\n"
        "## Actionable Suggestions\n\n1. Implement this agent.\n"
    )
    return {**state, "review_report": review_report}

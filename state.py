"""
Shared state definition for the MAS SDLC LangGraph pipeline.

Every agent node reads from and writes to this TypedDict, which LangGraph
passes through the sequential graph as a single source of truth.
"""

from typing import Optional, TypedDict


class SDLCState(TypedDict):
    """
    Global state shared across all agents in the SDLC pipeline.

    Fields
    ------
    user_prompt : str
        The original natural-language feature request supplied by the user.
    requirements : Optional[dict]
        Structured requirements document produced by the Requirements Agent.
        Contains keys: feature_name, description, functional_requirements,
        edge_cases, constraints, input_spec, output_spec.
    generated_code : Optional[str]
        Raw Python source code produced by the Code Generator Agent.
    test_results : Optional[dict]
        pytest execution results produced by the Test Engineer Agent.
        Contains keys: passed, failed, errors, output.
    review_report : Optional[str]
        Markdown-formatted code review report produced by the Code Reviewer Agent.
    log_path : str
        Relative path to the JSON log file for this pipeline run.
    errors : list[str]
        Accumulated non-fatal error messages from any agent in the pipeline.
    """

    user_prompt: str
    requirements: Optional[dict]
    generated_code: Optional[str]
    test_results: Optional[dict]
    review_report: Optional[str]
    log_path: str
    errors: list[str]

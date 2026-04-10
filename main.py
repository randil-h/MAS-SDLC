"""
MAS SDLC — CLI Entry Point.

Usage
-----
    python main.py "Build a login system with email and password validation"

If no argument is supplied the system defaults to a built-in demonstration prompt.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from graph import build_graph


def main() -> None:
    """
    Run the full SDLC pipeline from the command line.

    Reads the feature request from CLI arguments, initialises the pipeline
    state, invokes the LangGraph graph, and prints a summary of all outputs.
    """
    user_prompt: str = (
        " ".join(sys.argv[1:])
        or "Build a user registration module with input validation"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{timestamp}.json"

    # Ensure runtime directories exist
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    initial_state = {
        "user_prompt": user_prompt,
        "requirements": None,
        "generated_code": None,
        "test_results": None,
        "review_report": None,
        "log_path": log_path,
        "errors": [],
    }

    print(f"\nStarting SDLC Pipeline")
    print(f"   Prompt : {user_prompt}")
    print(f"   Log    : {log_path}\n")

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print("\nSDLC Pipeline Complete")
    print(f"   Requirements  : output/requirements.json")
    print(f"   Generated Code: output/generated_code.py")
    print(f"   Test Results  : output/test_results.json")
    print(f"   Review Report : output/review_report.md")
    print(f"   Full Log      : {log_path}")

    if final_state.get("errors"):
        print(f"\nErrors encountered during the run:")
        for err in final_state["errors"]:
            print(f"   • {err}")


if __name__ == "__main__":
    main()

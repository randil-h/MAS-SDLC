"""
Static analysis utilities for the MAS SDLC pipeline.

Wraps flake8 as a subprocess so that linting results are returned as
structured data without affecting the parent process.
"""

import subprocess
import sys
from pathlib import Path


def run_static_analysis(code_file_path: str) -> dict:
    """
    Run flake8 static analysis on a Python source file.

    The analysis is performed in a subprocess so that flake8's internal
    state cannot affect the running pipeline.  All exceptions are caught
    and reported inside the returned dictionary.

    Parameters
    ----------
    code_file_path : str
        Path to the Python source file to analyse.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``issues``       (list[str]): Each flake8 finding as a string,
          in the form ``"filename:line:col: CODE message"``.
        - ``issue_count``  (int): Total number of findings.
        - ``raw_output``   (str): The complete raw stdout from flake8.

        If flake8 is not installed or the file is missing, ``issues`` will
        contain a single descriptive error string and ``issue_count`` will be 1.
    """
    try:
        code_path = Path(code_file_path)
        if not code_path.exists():
            msg = f"[analysis_tools] Source file not found: '{code_file_path}'"
            return {"issues": [msg], "issue_count": 1, "raw_output": msg}

        proc = subprocess.run(
            [
                sys.executable, "-m", "flake8",
                "--max-line-length=120",
                str(code_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        raw = proc.stdout.strip()
        issues = [line for line in raw.splitlines() if line.strip()]

        return {
            "issues": issues,
            "issue_count": len(issues),
            "raw_output": raw,
        }

    except subprocess.TimeoutExpired:
        msg = "[analysis_tools] flake8 timed out after 60 seconds."
        return {"issues": [msg], "issue_count": 1, "raw_output": msg}
    except FileNotFoundError:
        msg = (
            "[analysis_tools] flake8 is not installed or not accessible. "
            "Install it with: pip install flake8"
        )
        return {"issues": [msg], "issue_count": 1, "raw_output": msg}
    except Exception as exc:
        msg = f"[analysis_tools] Unexpected error running flake8: {exc}"
        return {"issues": [msg], "issue_count": 1, "raw_output": msg}

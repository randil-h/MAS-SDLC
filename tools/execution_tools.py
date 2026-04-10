"""
Test execution utilities for the MAS SDLC pipeline.

Provides a safe wrapper around pytest that captures all output and never
propagates exceptions to callers — failures are returned as structured data.
"""

import subprocess
import sys
from pathlib import Path


def run_pytest(test_file_path: str) -> dict:
    """
    Execute pytest on a single test file and return structured results.

    The function runs pytest as a subprocess so that even test suites that
    call sys.exit() or have import errors cannot affect the parent process.
    stdout and stderr are both captured and merged into the "output" field.

    Parameters
    ----------
    test_file_path : str
        Path to the pytest-compatible Python test file to execute.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``passed``  (int): Number of tests that passed.
        - ``failed``  (int): Number of tests that failed.
        - ``errors``  (int): Number of tests that errored (collection/fixture errors).
        - ``output``  (str): Full combined stdout + stderr from the pytest run.

        If pytest cannot be invoked (e.g. not installed, file missing), all
        counts are 0 and ``output`` contains the error description.
    """
    result: dict = {"passed": 0, "failed": 0, "errors": 0, "output": ""}

    try:
        test_path = Path(test_file_path)
        if not test_path.exists():
            result["output"] = f"[execution_tools] Test file not found: '{test_file_path}'"
            return result

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        combined_output = proc.stdout + proc.stderr
        result["output"] = combined_output

        # Parse the summary line produced by pytest, e.g.:
        # "3 passed, 1 failed, 2 errors in 0.45s"
        for line in combined_output.splitlines():
            line_lower = line.lower()
            if " passed" in line_lower or " failed" in line_lower or " error" in line_lower:
                parts = line_lower.split(",")
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        result["passed"] = _extract_count(part)
                    elif "failed" in part:
                        result["failed"] = _extract_count(part)
                    elif "error" in part:
                        result["errors"] = _extract_count(part)

    except subprocess.TimeoutExpired:
        result["output"] = "[execution_tools] pytest timed out after 120 seconds."
    except FileNotFoundError:
        result["output"] = (
            "[execution_tools] pytest is not installed or not accessible on PATH. "
            "Install it with: pip install pytest"
        )
    except Exception as exc:
        result["output"] = f"[execution_tools] Unexpected error running pytest: {exc}"

    return result


def _extract_count(text: str) -> int:
    """
    Extract the leading integer from a pytest summary fragment such as '3 passed'.

    Parameters
    ----------
    text : str
        A whitespace-stripped summary fragment.

    Returns
    -------
    int
        The extracted count, or 0 if no integer prefix is found.
    """
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return 0

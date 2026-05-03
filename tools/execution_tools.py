import re
import subprocess
import sys
from pathlib import Path


def _parse_pytest_summary_counts(text: str) -> tuple[int, int, int]:
    """Extract passed / failed / error counts (last occurrence wins — pytest summary is near EOF)."""

    def _last_int(pattern: str) -> int:
        hits = list(re.finditer(pattern, text, flags=re.I))
        return int(hits[-1].group(1)) if hits else 0

    passed = _last_int(r"\b(\d+)\s+passed\b")
    failed = _last_int(r"\b(\d+)\s+failed\b")

    errs_long = _last_int(r"\b(\d+)\s+errors\b")
    errs_short = _last_int(r"\b(\d+)\s+error\b")
    errors = max(errs_long, errs_short)

    inter = re.search(r"Interrupted:\s*(\d+)\s+error", text, re.I)
    if inter:
        errors = max(errors, int(inter.group(1)))

    return passed, failed, errors


def _reconcile_collection_signals(combined_output: str, rc: int | None, result: dict) -> None:
    """
    When pytest exits oddly or parsers miss counts, derive collection/setup failures from text.
    """
    if not (combined_output or "").strip():
        return
    low = combined_output.lower()
    bumped = False
    if "error collecting" in low:
        bumped = True
    if "errors during collection" in low:
        bumped = True
    if "interrupted:" in low and "collection" in low:
        bumped = True
    if "nameerror" in low and "pytest" in low:
        bumped = True
    if re.search(r"collected\s+0\s+items[^\n]*/[^\n]+\berrors?\b", combined_output, re.I):
        bumped = True
    # Non-zero pytest exit but nothing parsed (older parsers / edge plugins)
    if isinstance(rc, int) and rc not in (0, 5):
        summary_hits = re.findall(r"\b\d+\s+(?:passed|failed|errors?)\b", combined_output, flags=re.I)
        if rc != 5 and len(summary_hits) == 0 and result.get("passed", 0) == 0:
            bumped = True
    if bumped:
        cur = result.get("errors", 0) or 0
        try:
            cur = int(cur)
        except (TypeError, ValueError):
            cur = 0
        result["errors"] = max(cur, 1)


def run_pytest(test_file_path: str) -> dict:
    """
    Execute pytest and return structured results with parsed test details.
    """

    result = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "output": "",
        "exit_code": None,
        "tests": [],       
        "failures": []     
    }

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
        result["exit_code"] = proc.returncode

        #  Parse test results (PASSED / FAILED)
        for line in combined_output.splitlines():
            if "::" in line and ("PASSED" in line or "FAILED" in line):
                parts = line.split("::")
                test_name = parts[-1].split()[0]
                status = "PASSED" if "PASSED" in line else "FAILED"

                result["tests"].append({
                    "test": test_name,
                    "status": status
                })

            #  Capture failure signals
            if "AssertionError" in line or "ValueError" in line:
                result["failures"].append(line.strip())

        p, f, e = _parse_pytest_summary_counts(combined_output)
        result["passed"] = max(result["passed"], p)
        result["failed"] = max(result["failed"], f)
        result["errors"] = max(result["errors"], e)

        rc = proc.returncode
        total_ran = result["passed"] + result["failed"] + result["errors"]
        if rc not in (0, None) and rc != 5 and total_ran == 0:
            result["errors"] = 1

        _reconcile_collection_signals(combined_output, proc.returncode, result)

    except subprocess.TimeoutExpired:
        result["output"] = "[execution_tools] pytest timed out after 120 seconds."
        result["exit_code"] = -1

    except FileNotFoundError:
        result["output"] = (
            "[execution_tools] pytest is not installed. Run: pip install pytest"
        )

    except Exception as exc:
        result["output"] = f"[execution_tools] Unexpected error: {exc}"

    return result
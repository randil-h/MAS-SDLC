import subprocess
import sys
from pathlib import Path


def run_pytest(test_file_path: str) -> dict:
    """
    Execute pytest and return structured results with parsed test details.
    """

    result = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "output": "",
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

        #  Parse summary counts
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
            "[execution_tools] pytest is not installed. Run: pip install pytest"
        )

    except Exception as exc:
        result["output"] = f"[execution_tools] Unexpected error: {exc}"

    return result


def _extract_count(text: str) -> int:
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return 0
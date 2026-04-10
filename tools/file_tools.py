"""
File I/O utilities and structured observability logging for the MAS SDLC pipeline.

All functions are safe by design: they catch every exception internally,
log to stderr, and return a safe sentinel value rather than raising.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_to_file(path: str, content: str) -> bool:
    """
    Save string content to a file, creating parent directories as needed.

    Parameters
    ----------
    path : str
        Destination file path (relative to the caller's working directory or absolute).
    content : str
        UTF-8 string content to write.

    Returns
    -------
    bool
        True on success, False if any I/O error occurred (error is printed to stderr).
    """
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return True
    except Exception as exc:
        print(f"[file_tools] save_to_file failed for '{path}': {exc}", file=sys.stderr)
        return False


def read_from_file(path: str) -> str:
    """
    Read and return the full text content of a file.

    Parameters
    ----------
    path : str
        Path to the file to read.

    Returns
    -------
    str
        The complete UTF-8 text content of the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path, with a descriptive message.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"[file_tools] read_from_file: file not found at '{path}'. "
            "Ensure the preceding agent has written its output before this agent runs."
        )
    return source.read_text(encoding="utf-8")


def append_log(log_path: str, entry: dict[str, Any]) -> None:
    """
    Append a structured JSON entry to the run-level log file.

    The log file contains a top-level JSON array.  If the file does not exist it
    is created with an empty array first.  Each entry is expected to contain at
    minimum the keys: timestamp, agent, input, tool_calls, output.

    Parameters
    ----------
    log_path : str
        Path to the JSON log file for this pipeline run.
    entry : dict[str, Any]
        Log entry dictionary.  A "timestamp" key is injected automatically if
        it is absent so callers do not have to supply it.

    Returns
    -------
    None
        This function never raises; errors are printed to stderr.
    """
    try:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if not entry.get("timestamp"):
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        if log_file.exists():
            try:
                existing: list[dict[str, Any]] = json.loads(log_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        else:
            existing = []

        existing.append(entry)
        log_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"[file_tools] append_log failed for '{log_path}': {exc}", file=sys.stderr)

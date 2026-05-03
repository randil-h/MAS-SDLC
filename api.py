"""
REST API for MAS SDLC pipeline orchestration.

Run with:
    uvicorn api:app --port 8000
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.code_generator_agent import code_generator_node
from agents.code_reviewer_agent import code_reviewer_node
from agents.requirements_agent import requirements_node
from agents.test_engineer_agent import test_engineer_node

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEP_ORDER: list[tuple[str, str, Any]] = [
    ("requirements",  "Requirements Analyst", requirements_node),
    ("code_generator","Code Generator",       code_generator_node),
    ("test_engineer", "Test Engineer",         test_engineer_node),
    ("code_reviewer", "Code Reviewer",         code_reviewer_node),
]

# ---------------------------------------------------------------------------
# Disk persistence
#
# Every run is written to runs/<run_id>.json after each state mutation so
# a server restart (uvicorn --reload, OOM crash, etc.) doesn't lose state.
# The `initial_state` field is internal pipeline scratch-space and is never
# serialised to disk to keep files small.
# ---------------------------------------------------------------------------

_RUNS_DIR = Path("runs")
_INTERNAL_KEYS = {"initial_state"}   # strip before writing to disk


def _persist_run(run_id: str, run: dict[str, Any]) -> None:
    """Write a sanitised copy of the run dict to disk (non-blocking best-effort)."""
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in run.items() if k not in _INTERNAL_KEYS}
        (_RUNS_DIR / f"{run_id}.json").write_text(
            json.dumps(payload, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass  # persistence failures must never kill the worker thread


def _load_run_from_disk(run_id: str) -> dict[str, Any] | None:
    """Return a run dict loaded from disk, or None if not found."""
    path = _RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _recover_interrupted_runs() -> None:
    """
    On server startup scan the runs directory and mark any runs that were
    left in 'running' or 'queued' state (i.e. interrupted by a crash) as
    'failed'.  The pipeline thread is gone so they can never complete.
    """
    if not _RUNS_DIR.exists():
        return
    for path in _RUNS_DIR.glob("*.json"):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("status") in ("running", "queued"):
                run["status"] = "failed"
                run["completed_at"] = _utc_now_iso()
                run.setdefault("errors", []).append(
                    "Run was interrupted by a server restart and cannot resume."
                )
                # Mark any in-progress step as failed too
                for step in run.get("steps", []):
                    if step.get("status") in ("running", "queued", "pending"):
                        step["status"] = "failed"
                        step["completed_at"] = step.get("completed_at") or _utc_now_iso()
                path.write_text(
                    json.dumps(run, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class RunCreateRequest(BaseModel):
    user_prompt: str = Field(min_length=5, max_length=6000)
    model_name: str = Field(default="phi3:mini", max_length=200)
    ollama_base_url: str = Field(default="http://localhost:11434", max_length=500)
    num_ctx: int = Field(default=2048, ge=512, le=32768)


class RunSummary(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress_percent: int
    current_step_key: str | None
    current_step_label: str | None
    started_at: str
    completed_at: str | None = None


class RunDetails(RunSummary):
    steps: list[dict[str, Any]]
    errors: list[str]
    result: dict[str, Any] | None
    log_path: str | None


# ---------------------------------------------------------------------------
# In-memory store + lock
# ---------------------------------------------------------------------------

_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="MAS SDLC API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    """Recover any runs left interrupted by a previous crash."""
    _recover_interrupted_runs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _mutate_run(run_id: str, updates: dict[str, Any]) -> None:
    """Apply updates to the in-memory run dict and immediately persist to disk."""
    with _runs_lock:
        run = _runs[run_id]
        run.update(updates)
        _persist_run(run_id, run)


def _result_payload_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Snapshot artefacts produced so far (used after each step and on failure)."""
    return {
        "requirements": state.get("requirements"),
        "generated_code": state.get("generated_code"),
        "test_results": state.get("test_results"),
        "review_report": state.get("review_report"),
        "errors": list(state.get("errors") or []),
        "log_path": state.get("log_path"),
    }


def _new_run_state(payload: RunCreateRequest) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    steps = [
        {
            "key": key,
            "label": label,
            "status": "pending",
            "started_at": None,
            "completed_at": None,
        }
        for key, label, _ in STEP_ORDER
    ]
    return {
        "status": "queued",
        "progress_percent": 0,
        "current_step_key": None,
        "current_step_label": None,
        "started_at": _utc_now_iso(),
        "completed_at": None,
        "steps": steps,
        "errors": [],
        "result": None,
        "log_path": log_path,
        # Internal — stripped before disk persistence
        "initial_state": {
            "user_prompt": payload.user_prompt.strip(),
            "requirements": None,
            "generated_code": None,
            "test_results": None,
            "review_report": None,
            "log_path": log_path,
            "errors": [],
        },
        "model_name": payload.model_name,
        "ollama_base_url": payload.ollama_base_url,
        "num_ctx": payload.num_ctx,
    }


# ---------------------------------------------------------------------------
# Model availability + auto-pull
# ---------------------------------------------------------------------------


def _model_is_available(model: str, base_url: str) -> bool:
    try:
        url = base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data: dict[str, Any] = json.loads(resp.read())
        names: list[str] = [m.get("name", "") for m in data.get("models", [])]
        tag_prefix = model.split(":")[0] + ":"
        return any(
            n == model or n.startswith(tag_prefix) or n == model + ":latest"
            for n in names
        )
    except Exception:
        return True  # can't check → let the agent fail with a clear message


def _pull_model(model: str, base_url: str, run_id: str) -> None:
    """Pull a missing model, streaming progress into the run's current_step_label."""

    def _set_label(label: str) -> None:
        with _runs_lock:
            if run_id in _runs:
                _runs[run_id]["current_step_label"] = label
                _persist_run(run_id, _runs[run_id])

    _set_label(f"Pulling model: {model} ...")

    try:
        pull_url = base_url.rstrip("/") + "/api/pull"
        body = json.dumps({"name": model, "stream": True}).encode()
        req = urllib.request.Request(
            pull_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk: dict[str, Any] = json.loads(line)
                    status_msg: str = chunk.get("status", "")
                    completed = chunk.get("completed")
                    total = chunk.get("total")
                    if completed and total:
                        pct = int(completed / total * 100)
                        _set_label(f"Pulling {model}: {pct}%")
                    elif status_msg:
                        _set_label(f"Pulling {model}: {status_msg}")
                except json.JSONDecodeError:
                    pass
        _set_label(f"Model ready: {model}")
        return
    except urllib.error.URLError:
        pass

    _set_label(f"Pulling {model} via CLI ...")
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout
        for line in proc.stdout:
            line = line.strip()
            if line:
                _set_label(f"Pulling {model}: {line[:80]}")
        proc.wait()
    except FileNotFoundError:
        _set_label("Warning: 'ollama' CLI not found; model pull skipped")

    _set_label(f"Model ready: {model}")


# ---------------------------------------------------------------------------
# Pipeline worker
# ---------------------------------------------------------------------------


def _pipeline_worker(run_id: str) -> None:
    with _runs_lock:
        run = _runs[run_id]
        run["status"] = "running"
        _persist_run(run_id, run)

    model     = run["model_name"]
    base_url  = run["ollama_base_url"]
    num_ctx   = run.get("num_ctx", 2048)

    os.environ["OLLAMA_MODEL"]    = model
    os.environ["OLLAMA_BASE_URL"] = base_url
    os.environ["OLLAMA_NUM_CTX"]  = str(num_ctx)

    if not _model_is_available(model, base_url):
        _pull_model(model, base_url, run_id)

    state = run["initial_state"]
    total_steps = len(STEP_ORDER)

    try:
        for idx, (step_key, step_label, step_fn) in enumerate(STEP_ORDER):
            with _runs_lock:
                current_run = _runs[run_id]
                current_run["current_step_key"]           = step_key
                current_run["current_step_label"]         = step_label
                current_run["steps"][idx]["status"]       = "running"
                current_run["steps"][idx]["started_at"]   = _utc_now_iso()
                current_run["progress_percent"]           = int((idx / total_steps) * 100)
                _persist_run(run_id, current_run)

            state = step_fn(state)

            with _runs_lock:
                current_run = _runs[run_id]
                current_run["steps"][idx]["status"]         = "completed"
                current_run["steps"][idx]["completed_at"]   = _utc_now_iso()
                current_run["progress_percent"]             = int(((idx + 1) / total_steps) * 100)
                current_run["errors"]                       = list(state.get("errors") or [])
                current_run["result"]                       = _result_payload_from_state(state)
                _persist_run(run_id, current_run)

        with _runs_lock:
            current_run = _runs[run_id]
            current_run["status"]               = "completed"
            current_run["current_step_key"]     = None
            current_run["current_step_label"]   = None
            current_run["completed_at"]         = _utc_now_iso()
            current_run["result"]               = _result_payload_from_state(state)
            _persist_run(run_id, current_run)

    except Exception as exc:
        with _runs_lock:
            current_run = _runs[run_id]
            current_run["status"]       = "failed"
            current_run["completed_at"] = _utc_now_iso()
            current_run["errors"]       = list(current_run.get("errors") or []) + [
                f"Unhandled pipeline error: {exc}"
            ]
            current_run["result"]       = _result_payload_from_state(state)
            if current_run["current_step_key"]:
                for step in current_run["steps"]:
                    if step["key"] == current_run["current_step_key"] and step["status"] == "running":
                        step["status"]       = "failed"
                        step["completed_at"] = _utc_now_iso()
                        break
            _persist_run(run_id, current_run)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _to_summary(run_id: str, run: dict[str, Any]) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        status=run["status"],
        progress_percent=run["progress_percent"],
        current_step_key=run["current_step_key"],
        current_step_label=run["current_step_label"],
        started_at=run["started_at"],
        completed_at=run["completed_at"],
    )


def _to_details(run_id: str, run: dict[str, Any]) -> RunDetails:
    return RunDetails(
        **_to_summary(run_id, run).model_dump(),
        steps=run["steps"],
        errors=run["errors"],
        result=run["result"],
        log_path=run["log_path"],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunSummary)
def create_run(payload: RunCreateRequest) -> RunSummary:
    run_id    = uuid.uuid4().hex
    run_state = _new_run_state(payload)
    with _runs_lock:
        _runs[run_id] = run_state
        _persist_run(run_id, run_state)
    thread = threading.Thread(target=_pipeline_worker, args=(run_id,), daemon=True)
    thread.start()
    with _runs_lock:
        return _to_summary(run_id, _runs[run_id])


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    with _runs_lock:
        return [_to_summary(run_id, run) for run_id, run in _runs.items()]


@app.get("/api/runs/{run_id}", response_model=RunDetails)
def get_run(run_id: str) -> RunDetails:
    # Check in-memory first (fast path)
    with _runs_lock:
        run = _runs.get(run_id)

    # Fall back to disk — covers server restarts
    if run is None:
        run = _load_run_from_disk(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        # Restore into memory so subsequent polls are fast
        with _runs_lock:
            _runs[run_id] = run

    return _to_details(run_id, run)

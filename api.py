"""
REST API for MAS SDLC pipeline orchestration.

Run with:
    uvicorn api:app --reload
"""

from __future__ import annotations

import os
import threading
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

STEP_ORDER: list[tuple[str, str, Any]] = [
    (
        "requirements",
        "Requirements Analyst",
        requirements_node,
    ),
    (
        "code_generator",
        "Code Generator",
        code_generator_node,
    ),
    (
        "test_engineer",
        "Test Engineer",
        test_engineer_node,
    ),
    (
        "code_reviewer",
        "Code Reviewer",
        code_reviewer_node,
    ),
]


class RunCreateRequest(BaseModel):
    user_prompt: str = Field(min_length=5, max_length=6000)
    model_name: str = Field(default="llama3:8b", max_length=200)
    ollama_base_url: str = Field(default="http://localhost:11434", max_length=500)


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


_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()

app = FastAPI(title="MAS SDLC API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _new_run_state(payload: RunCreateRequest) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    steps = [
        {"key": key, "label": label, "status": "pending", "started_at": None, "completed_at": None}
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
    }


def _pipeline_worker(run_id: str) -> None:
    with _runs_lock:
        run = _runs[run_id]
        run["status"] = "running"

    os.environ["OLLAMA_MODEL"] = run["model_name"]
    os.environ["OLLAMA_BASE_URL"] = run["ollama_base_url"]

    state = run["initial_state"]
    total_steps = len(STEP_ORDER)

    try:
        for idx, (step_key, step_label, step_fn) in enumerate(STEP_ORDER):
            with _runs_lock:
                current_run = _runs[run_id]
                current_run["current_step_key"] = step_key
                current_run["current_step_label"] = step_label
                current_run["steps"][idx]["status"] = "running"
                current_run["steps"][idx]["started_at"] = _utc_now_iso()
                current_run["progress_percent"] = int((idx / total_steps) * 100)

            state = step_fn(state)

            with _runs_lock:
                current_run = _runs[run_id]
                current_run["steps"][idx]["status"] = "completed"
                current_run["steps"][idx]["completed_at"] = _utc_now_iso()
                current_run["progress_percent"] = int(((idx + 1) / total_steps) * 100)
                current_run["errors"] = list(state.get("errors") or [])

        with _runs_lock:
            current_run = _runs[run_id]
            current_run["status"] = "completed"
            current_run["current_step_key"] = None
            current_run["current_step_label"] = None
            current_run["completed_at"] = _utc_now_iso()
            current_run["result"] = {
                "requirements": state.get("requirements"),
                "generated_code": state.get("generated_code"),
                "test_results": state.get("test_results"),
                "review_report": state.get("review_report"),
                "errors": state.get("errors") or [],
                "log_path": state.get("log_path"),
            }
    except Exception as exc:
        with _runs_lock:
            current_run = _runs[run_id]
            current_run["status"] = "failed"
            current_run["completed_at"] = _utc_now_iso()
            current_run["errors"] = list(current_run.get("errors") or []) + [f"Unhandled API error: {exc}"]
            if current_run["current_step_key"]:
                for step in current_run["steps"]:
                    if step["key"] == current_run["current_step_key"] and step["status"] == "running":
                        step["status"] = "failed"
                        step["completed_at"] = _utc_now_iso()
                        break


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunSummary)
def create_run(payload: RunCreateRequest) -> RunSummary:
    run_id = uuid.uuid4().hex
    run_state = _new_run_state(payload)
    with _runs_lock:
        _runs[run_id] = run_state
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
    with _runs_lock:
        run = _runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return _to_details(run_id, run)

# AI Software Dev Team — Multi-Agent SDLC System

A locally-hosted Multi-Agent System (MAS) that automates the Software Development Lifecycle. Given a natural-language feature request, four specialised AI agents collaborate sequentially to produce structured requirements, working Python code, a pytest test suite, and a professional code review report — all running 100% locally via Ollama with no paid APIs.

Built with **LangGraph** for orchestration, **Ollama (llama3:8b)** as the LLM engine, and **Streamlit** for the web interface.

---

## Prerequisites

1. **Python 3.11+** installed.
2. **Ollama** installed and running — [ollama.ai](https://ollama.ai).
3. The target model pulled locally:
   ```bash
   ollama pull llama3:8b
   ```

---

## Installation

```bash
cd mas_sdlc
pip install -r requirements.txt
```

---

## Running the System

### Option A — Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  
Enter a feature request, configure the Ollama model in the sidebar, and click **Run SDLC Pipeline**.

### Option B — Command Line

```bash
python main.py "Build a password reset feature with token expiry"
```

If no argument is supplied, a default demonstration prompt is used.

---

## Output Files

All outputs are written to the `output/` directory and are gitignored (regenerated on every run).

| File | Produced by | Contents |
|------|-------------|----------|
| `output/requirements.json` | Requirements Analyst | Structured JSON: feature name, description, functional requirements, edge cases, constraints, I/O specs |
| `output/generated_code.py` | Code Generator | Clean, type-hinted Python module fulfilling all requirements |
| `output/test_suite.py` | Test Engineer | Full pytest test suite for the generated code |
| `output/test_results.json` | Test Engineer | pytest execution results: passed / failed / errors counts + raw output |
| `output/review_report.md` | Code Reviewer | Structured Markdown review: summary, requirements coverage, code quality, security, suggestions |

Observability logs are written to `logs/run_<timestamp>.json` — one JSON array per run, with one entry per agent recording its inputs, tool calls, and outputs.

---

## Running the Test Harness

```bash
pytest tests/ -v
```

The `tests/` directory contains one evaluation file per agent.  
`test_code_generator_agent.py` is fully implemented; the other three are stubs for team members to complete.

---

## Project Structure

```
mas_sdlc/
├── app.py                         # Streamlit UI
├── main.py                        # CLI entry point
├── state.py                       # SDLCState TypedDict
├── graph.py                       # LangGraph pipeline
├── agents/
│   ├── requirements_agent.py      # Agent 1 (stub)
│   ├── code_generator_agent.py    # Agent 2 — fully implemented
│   ├── test_engineer_agent.py     # Agent 3 (stub)
│   └── code_reviewer_agent.py     # Agent 4 (stub)
├── tools/
│   ├── file_tools.py              # save_to_file, read_from_file, append_log
│   ├── execution_tools.py         # run_pytest
│   └── analysis_tools.py         # run_static_analysis (flake8)
├── tests/
│   ├── test_code_generator_agent.py   # Full test suite
│   ├── test_requirements_agent.py     # Stub
│   ├── test_engineer_agent.py         # Stub
│   └── test_code_reviewer_agent.py    # Stub
├── output/                        # Runtime outputs (gitignored)
├── logs/                          # Observability logs (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Team Contributions

| Member | Agent | Tool |
|--------|-------|------|
| Member 1 | Requirements Analyst | `save_to_file` / `append_log` |
| **Member 2** | **Code Generator** | **`read_from_file`** |
| Member 3 | Test Engineer | `run_pytest` |
| Member 4 | Code Reviewer | `run_static_analysis` |

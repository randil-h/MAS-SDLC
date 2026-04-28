# AI Software Dev Team - Multi-Agent SDLC System

A locally hosted Multi-Agent System (MAS) that automates the Software Development Lifecycle. Given a natural-language feature request, four specialised AI agents collaborate sequentially to produce structured requirements, working Python code, a pytest test suite, and a code review report.

Built with **LangGraph** for orchestration, **Ollama (llama3:8b)** for LLM inference, **FastAPI** for the REST backend, and **Next.js** for a modern interactive frontend.

---

## Prerequisites

1. Python 3.11+
2. Node.js 18+
3. Ollama installed and running - [https://ollama.ai](https://ollama.ai)
4. Local model pulled:

```bash
ollama pull llama3:8b
```

---

## Installation

### 1) Backend dependencies

```bash
pip install -r requirements.txt
```

### 2) Frontend dependencies

```bash
cd frontend
npm install
```

---

## Run the Full Web App

Start backend (from project root):

```bash
uvicorn api:app --reload --port 8000
```

Start frontend (in a second terminal):

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

If your backend runs on a different URL, set:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## REST API

- `GET /health` - service health check
- `POST /api/runs` - create and start a new SDLC run
- `GET /api/runs` - list runs
- `GET /api/runs/{run_id}` - get live status, progress, step timeline, and final outputs

Example create request:

```json
{
  "user_prompt": "Build a password reset feature with token expiry",
  "model_name": "llama3:8b",
  "ollama_base_url": "http://localhost:11434"
}
```

---

## CLI Mode (Optional)

You can still run the pipeline directly:

```bash
python main.py "Build a password reset feature with token expiry"
```

---

## Output Files

All outputs are written to `output/` and regenerated on each run.

| File | Produced by | Contents |
|------|-------------|----------|
| `output/requirements.json` | Requirements Analyst | Structured JSON requirements |
| `output/generated_code.py` | Code Generator | Generated Python module |
| `output/test_suite.py` | Test Engineer | Generated pytest suite (when agent implemented) |
| `output/test_results.json` | Test Engineer | pytest execution summary |
| `output/review_report.md` | Code Reviewer | Markdown review report |

Run logs are written to `logs/run_<timestamp>_<id>.json`.

---

## Project Structure

```
MAS-SDLC/
├── api.py                         # FastAPI REST backend
├── main.py                        # CLI entry point
├── graph.py                       # LangGraph pipeline
├── state.py                       # Shared SDLCState
├── agents/                        # Pipeline agents
├── tools/                         # File, execution, and analysis tools
├── frontend/                      # Next.js frontend
│   ├── app/
│   ├── components/
│   └── lib/
├── output/                        # Runtime outputs
├── logs/                          # Observability logs
└── requirements.txt
```

To run the code generator agent tests
```bash
pytest tests/test_code_generator_agent.py -rA
```



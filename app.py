"""
MAS SDLC — Streamlit Web Interface.

Launch with:
    streamlit run app.py

The UI provides a professional, single-page experience for the full SDLC
pipeline: prompt input → live agent progress → tabbed output display.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure the mas_sdlc package is importable when app.py is
# launched from the project root (streamlit run mas_sdlc/app.py) or from
# within the package directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Change working directory to the package root so relative output/logs paths
# used by agents resolve correctly.
os.chdir(_HERE)

from graph import build_graph  # noqa: E402 — must come after sys.path setup

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Software Dev Team",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal tweaks for a polished look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Tighten sidebar spacing */
        section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

        /* Card-like metric boxes */
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 0.6rem 1rem;
        }

        /* Status badge styling */
        .badge-pending  { color: #6c757d; font-weight: 600; }
        .badge-running  { color: #f0a500; font-weight: 600; }
        .badge-done     { color: #28a745; font-weight: 600; }
        .badge-error    { color: #dc3545; font-weight: 600; }

        /* Download button consistency */
        div[data-testid="stDownloadButton"] > button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/robot-2.png",
        width=64,
    )
    st.title("AI Software Dev Team")
    st.caption("Locally-hosted Multi-Agent SDLC System")
    st.divider()

    st.subheader("Configuration")
    ollama_base_url: str = st.text_input(
        "Ollama Base URL",
        value="http://localhost:11434",
        help="URL of your locally running Ollama server.",
    )
    model_name: str = st.selectbox(
        "LLM Model",
        options=["llama3:8b", "phi3", "qwen:7b", "mistral"],
        index=0,
        help="Model must be pulled via `ollama pull <model>` before running.",
    )

    st.divider()
    with st.expander("About the Agents", expanded=False):
        st.markdown(
            """
**1 · Requirements Analyst**
Decomposes your feature request into a structured JSON requirements document with functional requirements, edge cases, and constraints.

**2 · Code Generator** *(this member's agent)*
Translates the requirements document into clean, type-hinted, fully-documented Python source code using only the standard library.

**3 · Test Engineer**
Writes a complete `pytest` test suite covering happy paths, edge cases, and negative scenarios, then executes it.

**4 · Code Reviewer**
Runs static analysis and produces a structured Markdown code review report covering quality, security, and actionable suggestions.
            """
        )

    st.divider()
    st.caption("Stack: LangGraph · Ollama · Streamlit")
    st.caption("All processing runs 100% locally.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_errors" not in st.session_state:
    st.session_state.pipeline_errors = []
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown("## AI Software Dev Team")
st.markdown(
    "Enter a natural-language feature request below. "
    "The four-agent pipeline will autonomously produce requirements, "
    "working Python code, a test suite, and a code review report — "
    "all running locally via Ollama."
)
st.divider()

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------

col_input, col_meta = st.columns([3, 1])

with col_input:
    user_prompt: str = st.text_area(
        "Feature Request",
        value="Build a password reset module with token generation and expiry validation",
        height=120,
        placeholder="Describe the feature you want to build…",
        help="Be as specific as possible for best results.",
    )

with col_meta:
    st.markdown("**Pipeline steps**")
    st.markdown("1. Requirements Analysis")
    st.markdown("2. Code Generation")
    st.markdown("3. Test Engineering")
    st.markdown("4. Code Review")
    st.markdown("")
    estimated = "~3–8 min"
    st.info(f"Est. time: {estimated}")

run_button = st.button(
    "Run SDLC Pipeline",
    type="primary",
    use_container_width=True,
    disabled=not user_prompt.strip(),
)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_button and user_prompt.strip():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{timestamp}.json"
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    initial_state = {
        "user_prompt": user_prompt.strip(),
        "requirements": None,
        "generated_code": None,
        "test_results": None,
        "review_report": None,
        "log_path": log_path,
        "errors": [],
    }

    st.divider()
    st.markdown("### Pipeline Progress")

    # We use st.status blocks to show live per-agent progress
    agent_labels = [
        ("requirements",   "Requirements Analyst",  "Analysing feature request and producing requirements…"),
        ("code_generator", "Code Generator",         "Writing Python code from the requirements…"),
        ("test_engineer",  "Test Engineer",           "Generating and running the pytest test suite…"),
        ("code_reviewer",  "Code Reviewer",           "Running static analysis and writing the review report…"),
    ]

    final_state = None
    run_error = None

    with st.container():
        progress_bar = st.progress(0, text="Starting pipeline…")

        for idx, (_, label, description) in enumerate(agent_labels):
            progress_bar.progress(
                int((idx / len(agent_labels)) * 100),
                text=f"Running: {label}",
            )

        # Run the full graph — progress is approximate since LangGraph is
        # synchronous; we update the bar before and after the invoke call.
        progress_bar.progress(5, text="Initialising LangGraph…")

        try:
            # Inject sidebar config into environment so agents can pick it up
            os.environ["OLLAMA_BASE_URL"] = ollama_base_url
            os.environ["OLLAMA_MODEL"] = model_name

            graph = build_graph()

            with st.status("Running all agents...", expanded=True) as status_block:
                st.write("**[1/4]** Requirements Analyst — analysing prompt…")
                st.write("**[2/4]** Code Generator — writing Python code…")
                st.write("**[3/4]** Test Engineer — generating & running tests…")
                st.write("**[4/4]** Code Reviewer — reviewing code quality…")

                final_state = graph.invoke(initial_state)

                if final_state.get("errors"):
                    status_block.update(
                        label="Pipeline completed with errors", state="error"
                    )
                else:
                    status_block.update(
                        label="Pipeline completed successfully!", state="complete"
                    )

        except Exception as exc:
            run_error = str(exc)
            st.error(f"Pipeline failed with an unexpected error: {exc}")

        progress_bar.progress(100, text="Done!")

    if final_state:
        st.session_state.pipeline_result = final_state
        st.session_state.pipeline_errors = final_state.get("errors", [])
        st.session_state.pipeline_ran = True

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if st.session_state.pipeline_ran and st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    errors = st.session_state.pipeline_errors

    st.divider()
    st.markdown("### Pipeline Results")

    # Summary metrics row
    test_results: dict = result.get("test_results") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Requirements", "Done" if result.get("requirements") else "Missing")
    m2.metric("Code Generated", "Done" if result.get("generated_code") else "Missing")
    m3.metric(
        "Tests Passed",
        f"{test_results.get('passed', 0)} / "
        f"{test_results.get('passed', 0) + test_results.get('failed', 0)}",
    )
    m4.metric("Review Report", "Done" if result.get("review_report") else "Missing")

    st.markdown("")

    # Four output tabs
    tab_req, tab_code, tab_tests, tab_review = st.tabs(
        ["Requirements", "Generated Code", "Test Results", "Review Report"]
    )

    # ---- Tab 1: Requirements ------------------------------------------------
    with tab_req:
        requirements = result.get("requirements")
        if requirements:
            st.markdown("#### Structured Requirements Document")
            st.json(requirements)
            req_json_str = json.dumps(requirements, indent=2)
            st.download_button(
                label="Download requirements.json",
                data=req_json_str,
                file_name="requirements.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("Requirements were not generated. Check the errors panel below.")

    # ---- Tab 2: Generated Code ----------------------------------------------
    with tab_code:
        generated_code = result.get("generated_code")
        if generated_code:
            st.markdown("#### Generated Python Module")
            line_count = len(generated_code.splitlines())
            st.caption(f"{line_count} lines of code")
            st.code(generated_code, language="python", line_numbers=True)
            st.download_button(
                label="Download generated_code.py",
                data=generated_code,
                file_name="generated_code.py",
                mime="text/x-python",
                use_container_width=True,
            )
        else:
            st.warning("Code was not generated. Check the errors panel below.")

    # ---- Tab 3: Test Results ------------------------------------------------
    with tab_tests:
        if test_results:
            st.markdown("#### Test Execution Summary")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Passed", test_results.get("passed", 0))
            tc2.metric("Failed", test_results.get("failed", 0))
            tc3.metric("Errors", test_results.get("errors", 0))

            st.markdown("#### Raw pytest Output")
            pytest_output: str = test_results.get("output", "No output captured.")
            st.code(pytest_output, language="text")

            st.download_button(
                label="Download test_results.json",
                data=json.dumps(test_results, indent=2),
                file_name="test_results.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("Test results are not available. Check the errors panel below.")

    # ---- Tab 4: Review Report -----------------------------------------------
    with tab_review:
        review_report = result.get("review_report")
        if review_report:
            st.markdown("#### Code Review Report")
            st.markdown(review_report)
            st.download_button(
                label="Download review_report.md",
                data=review_report,
                file_name="review_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.warning("Review report was not generated. Check the errors panel below.")

    # ---- Errors panel -------------------------------------------------------
    if errors:
        st.divider()
        with st.expander("Errors & Warnings", expanded=True):
            for err in errors:
                st.error(err)

    # ---- Log path info ------------------------------------------------------
    log_path_display = result.get("log_path", "logs/run_<timestamp>.json")
    st.divider()
    st.caption(f"Full observability log saved to: `{log_path_display}`")

# ---------------------------------------------------------------------------
# Empty state hint
# ---------------------------------------------------------------------------

elif not st.session_state.pipeline_ran:
    st.divider()
    st.info(
        "Enter a feature request above and click **Run SDLC Pipeline** to get started.",
    )

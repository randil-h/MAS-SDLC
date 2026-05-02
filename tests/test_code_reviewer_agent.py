"""
Evaluation harness for the Code Reviewer Agent and its tools.

Tests validate that:
  1. parse_review_sections correctly extracts all 6 Markdown sections.
  2. parse_review_sections extracts the verdict from the Summary section.
  3. run_static_analysis returns a well-structured dict and handles errors.
  4. code_reviewer_node produces a report containing all 6 required headings.
  5. code_reviewer_node creates output/review_report.md on disk.
  6. code_reviewer_node falls back gracefully when inputs are missing.
  7. The observability log is a valid JSON array with the required keys.
  8. Property-based invariants hold regardless of the input scenario.

All agent-level tests mock OllamaLLM and run_static_analysis so the suite
runs fully offline without a running Ollama instance.
"""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.code_reviewer_agent import code_reviewer_node
from state import SDLCState
from tools.analysis_tools import run_static_analysis
from tools.review_tools import ReviewSections, parse_review_sections

# Shared fixtures

_SAMPLE_REQUIREMENTS: dict = {
    "feature_name": "Password Validator",
    "description": "Validates a password string against strength rules.",
    "functional_requirements": [
        "Password must be at least 8 characters long.",
        "Password must contain at least one uppercase letter.",
        "Password must contain at least one digit.",
    ],
    "edge_cases": [
        "Empty string should be rejected.",
        "None input should raise TypeError.",
    ],
    "constraints": ["Use only Python standard library."],
}

_SAMPLE_CODE = '''\
def validate_password(password: str) -> bool:
    """Validate a password against strength rules."""
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain at least one digit.")
    return True
'''

_SAMPLE_TEST_RESULTS: dict = {
    "passed": 5,
    "failed": 0,
    "errors": 0,
    "output": "5 passed in 0.12s",
}

_SAMPLE_REVIEW_REPORT = """\
## Summary
The generated code PASSES the review. The password validator correctly implements
all functional requirements with proper type hints and input validation. No
critical issues were found.

## Requirements Coverage
- Password must be at least 8 characters: YES — len(password) < 8 check present.
- Must contain uppercase: YES — any(c.isupper() ...) check present.
- Must contain a digit: YES — any(c.isdigit() ...) check present.
- Empty string rejected: YES — handled by the length check (len < 8).
- None raises TypeError: YES — isinstance check at line 3 raises TypeError.

## Code Quality
The code is clean and readable. `validate_password` uses a full type hint
`(password: str) -> bool`. A comprehensive one-line docstring is present.
No clever one-liners; all checks use clear conditional expressions.

## Security & Edge Cases
No injection risks identified. Input validation is performed before any
string processing. The None input edge case is handled. Unicode/non-ASCII
characters are not explicitly addressed but do not cause crashes.

## Test Coverage Assessment
The test suite passes all 5 tests with 0 failures. Happy path and all
documented edge cases appear to be covered. No obvious gaps in coverage.

## Actionable Suggestions
1. Add handling for unicode/non-ASCII characters to avoid ambiguous behaviour
   with isupper() and isdigit() on multi-byte characters.
2. Consider adding a maximum password length guard to prevent potential DoS
   via extremely long input strings.
"""

_CLEAN_STATIC_ANALYSIS: dict = {
    "issues": [],
    "issue_count": 0,
    "raw_output": "",
}


def _make_state(tmp_dir: str, **overrides) -> SDLCState:
    """Build a minimal SDLCState that points logs at a temp directory."""
    base: SDLCState = SDLCState(
        user_prompt="Build a password validator",
        requirements=_SAMPLE_REQUIREMENTS,
        generated_code=_SAMPLE_CODE,
        test_results=_SAMPLE_TEST_RESULTS,
        review_report=None,
        log_path=str(Path(tmp_dir) / "logs" / "run_test.json"),
        errors=[],
    )
    return {**base, **overrides}


def _run_reviewer_node(
    tmp_dir: str,
    llm_response: str = _SAMPLE_REVIEW_REPORT,
    state_overrides: dict | None = None,
    write_code_file: bool = True,
    write_req_file: bool = True,
) -> SDLCState:
    """
    Run code_reviewer_node inside a temp directory with mocked LLM and
    static analysis.

    Patches:
      - agents.code_reviewer_agent.Ollama → returns llm_response
      - agents.code_reviewer_agent.run_static_analysis → returns _CLEAN_STATIC_ANALYSIS
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_dir)
    Path("output").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        if write_code_file:
            Path("output/generated_code.py").write_text(_SAMPLE_CODE, encoding="utf-8")
        if write_req_file:
            Path("output/requirements.json").write_text(
                json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8"
            )

        state = _make_state(tmp_dir, **(state_overrides or {}))

        with (
            patch("agents.code_reviewer_agent.Ollama") as MockOllama,
            patch("agents.code_reviewer_agent.run_static_analysis") as mock_sa,
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = llm_response
            MockOllama.return_value = mock_llm
            mock_sa.return_value = _CLEAN_STATIC_ANALYSIS

            result = code_reviewer_node(state)
    finally:
        os.chdir(original_cwd)

    return result


# 1. parse_review_sections — unit tests

class TestParseReviewSections(unittest.TestCase):
    """Unit tests for the parse_review_sections tool in review_tools.py."""

    def test_complete_report_all_sections_populated(self) -> None:
        """All six section fields are non-empty when the report is complete."""
        result = parse_review_sections(_SAMPLE_REVIEW_REPORT)
        self.assertTrue(result.summary)
        self.assertTrue(result.requirements_coverage)
        self.assertTrue(result.code_quality)
        self.assertTrue(result.security_edge_cases)
        self.assertTrue(result.test_coverage_assessment)
        self.assertTrue(result.actionable_suggestions)

    def test_complete_report_is_complete_true(self) -> None:
        """is_complete is True and missing_sections is empty for a full report."""
        result = parse_review_sections(_SAMPLE_REVIEW_REPORT)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.missing_sections, [])

    def test_partial_report_is_complete_false(self) -> None:
        """is_complete is False when one or more sections are absent."""
        partial = "## Summary\nCode PASSES.\n## Code Quality\nLooks good."
        result = parse_review_sections(partial)
        self.assertFalse(result.is_complete)

    def test_partial_report_missing_sections_recorded(self) -> None:
        """missing_sections lists the canonical headings that were not found."""
        partial = "## Summary\nCode PASSES.\n## Code Quality\nLooks good."
        result = parse_review_sections(partial)
        self.assertIn("## Requirements Coverage", result.missing_sections)
        self.assertIn("## Security & Edge Cases", result.missing_sections)
        self.assertIn("## Test Coverage Assessment", result.missing_sections)
        self.assertIn("## Actionable Suggestions", result.missing_sections)
        # Sections that ARE present must not appear in missing_sections
        self.assertNotIn("## Summary", result.missing_sections)
        self.assertNotIn("## Code Quality", result.missing_sections)

    def test_empty_string_returns_all_empty_fields(self) -> None:
        """An empty report string returns a ReviewSections with all fields empty."""
        result = parse_review_sections("")
        self.assertFalse(result.is_complete)
        self.assertEqual(result.summary, "")
        self.assertEqual(result.verdict, "UNKNOWN")
        self.assertEqual(len(result.missing_sections), 6)

    def test_whitespace_only_string_treated_as_empty(self) -> None:
        """A whitespace-only string is treated the same as an empty string."""
        result = parse_review_sections("   \n\t  \n  ")
        self.assertFalse(result.is_complete)
        self.assertEqual(len(result.missing_sections), 6)

    def test_subheadings_included_in_section_content(self) -> None:
        """### subheadings inside a section are captured as content, not boundaries."""
        report = (
            "## Summary\n"
            "Code PASSES.\n"
            "### Detail\n"
            "Some detail text.\n"
            "## Requirements Coverage\n"
            "All met.\n"
        )
        result = parse_review_sections(report)
        self.assertIn("### Detail", result.summary)
        self.assertIn("Some detail text.", result.summary)

    def test_section_content_does_not_bleed_into_next(self) -> None:
        """Content from one section does not appear in the following section."""
        result = parse_review_sections(_SAMPLE_REVIEW_REPORT)
        # "PASSES" verdict phrase is in Summary; it must not appear in Code Quality
        self.assertNotIn("PASSES the review", result.code_quality)


# 2. parse_review_sections — verdict extraction

class TestExtractVerdict(unittest.TestCase):
    """Tests for verdict extraction via parse_review_sections."""

    def _verdict(self, summary_text: str) -> str:
        report = f"## Summary\n{summary_text}\n## Requirements Coverage\nOK.\n## Code Quality\nOK.\n## Security & Edge Cases\nOK.\n## Test Coverage Assessment\nOK.\n## Actionable Suggestions\n1. None.\n"
        return parse_review_sections(report).verdict

    def test_verdict_passes(self) -> None:
        """'PASSES' in summary yields verdict PASSES."""
        self.assertEqual(self._verdict("The code PASSES the review."), "PASSES")

    def test_verdict_fails(self) -> None:
        """'FAILS' in summary yields verdict FAILS."""
        self.assertEqual(self._verdict("The code FAILS due to missing tests."), "FAILS")

    def test_verdict_conditionally_passes(self) -> None:
        """'CONDITIONALLY PASSES' is matched before the shorter 'PASSES'."""
        self.assertEqual(
            self._verdict("The code CONDITIONALLY PASSES pending fixes."),
            "CONDITIONALLY PASSES",
        )

    def test_verdict_case_insensitive(self) -> None:
        """Verdict matching is case-insensitive."""
        self.assertEqual(self._verdict("The code passes the review."), "PASSES")

    def test_verdict_unknown_when_absent(self) -> None:
        """Summary with no verdict phrase returns UNKNOWN."""
        self.assertEqual(self._verdict("The review is complete."), "UNKNOWN")

    def test_verdict_unknown_for_empty_summary(self) -> None:
        """Empty summary field returns UNKNOWN."""
        result = parse_review_sections("")
        self.assertEqual(result.verdict, "UNKNOWN")


# 3. run_static_analysis — unit tests

class TestRunStaticAnalysis(unittest.TestCase):
    """Unit tests for the run_static_analysis tool in analysis_tools.py."""

    def test_returns_required_keys(self) -> None:
        """Return dict always contains 'issues', 'issue_count', 'raw_output'."""
        result = run_static_analysis("nonexistent_file.py")
        for key in ("issues", "issue_count", "raw_output"):
            self.assertIn(key, result, f"Missing key '{key}' in result")

    def test_missing_file_returns_graceful_error(self) -> None:
        """A non-existent file path returns a structured error dict, never raises."""
        result = run_static_analysis("/tmp/this_file_does_not_exist_mas_sdlc.py")
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result["issue_count"], 1)
        self.assertTrue(result["issues"])

    def test_missing_file_does_not_raise(self) -> None:
        """Calling run_static_analysis on a missing path must never raise."""
        try:
            run_static_analysis("completely_missing_path/also_missing.py")
        except Exception as exc:
            self.fail(f"run_static_analysis raised an exception: {exc}")

    def test_clean_code_returns_zero_issues(self) -> None:
        """A syntactically clean, PEP-8-compliant file produces zero issues."""
        clean_code = (
            'def add(a: int, b: int) -> int:\n'
            '    """Return the sum of a and b."""\n'
            '    return a + b\n'
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(clean_code)
            tmp_path = f.name
        try:
            result = run_static_analysis(tmp_path)
            self.assertEqual(
                result["issue_count"], 0,
                f"Expected 0 flake8 issues, got: {result['issues']}",
            )
        finally:
            os.unlink(tmp_path)

    def test_issues_field_is_always_a_list(self) -> None:
        """The 'issues' value is always a list, even on error paths."""
        result = run_static_analysis("no_such_file.py")
        self.assertIsInstance(result["issues"], list)


# 4. code_reviewer_node — structure tests

class TestCodeReviewerNodeStructure(unittest.TestCase):
    """Integration tests for code_reviewer_node using a mocked LLM."""

    def test_all_six_headings_present_in_report(self) -> None:
        """The returned review_report must contain all 6 required ## headings."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(tmp)
        report = result.get("review_report", "")
        for heading in (
            "## Summary",
            "## Requirements Coverage",
            "## Code Quality",
            "## Security & Edge Cases",
            "## Test Coverage Assessment",
            "## Actionable Suggestions",
        ):
            self.assertIn(heading, report, f"Missing heading: '{heading}'")

    def test_output_file_is_created(self) -> None:
        """output/review_report.md must exist on disk after the node runs."""
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            Path("output").mkdir()
            Path("logs").mkdir()
            Path("output/generated_code.py").write_text(_SAMPLE_CODE, encoding="utf-8")
            Path("output/requirements.json").write_text(
                json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8"
            )
            state = _make_state(tmp)
            with (
                patch("agents.code_reviewer_agent.Ollama") as MockOllama,
                patch("agents.code_reviewer_agent.run_static_analysis") as mock_sa,
            ):
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_REVIEW_REPORT
                MockOllama.return_value = mock_llm
                mock_sa.return_value = _CLEAN_STATIC_ANALYSIS
                code_reviewer_node(state)
            self.assertTrue(
                Path("output/review_report.md").exists(),
                "output/review_report.md was not created",
            )
            os.chdir(original_cwd)

    def test_state_has_review_report_populated(self) -> None:
        """state['review_report'] must be a non-empty string after a successful run."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(tmp)
        self.assertIsInstance(result.get("review_report"), str)
        self.assertTrue(result["review_report"])

    def test_no_errors_on_valid_complete_run(self) -> None:
        """No errors are appended to state when the report contains all 6 sections."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(tmp)
        self.assertEqual(result.get("errors", []), [])

    def test_parse_review_sections_finds_complete_report(self) -> None:
        """parse_review_sections confirms the node's output is complete and has a verdict."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(tmp)
        parsed = parse_review_sections(result.get("review_report", ""))
        self.assertTrue(parsed.is_complete)
        self.assertEqual(parsed.verdict, "PASSES")


# 5. code_reviewer_node — fallback and error handling

class TestCodeReviewerNodeFallback(unittest.TestCase):
    """Tests that the node handles missing or broken inputs gracefully."""

    def test_no_generated_code_records_error_in_state(self) -> None:
        """When no code is available, an error must be recorded in state['errors']."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(
                tmp,
                write_code_file=False,
                state_overrides={"generated_code": None},
            )
        self.assertTrue(
            len(result.get("errors", [])) > 0,
            "An error should be recorded when generated_code is absent",
        )

    def test_no_generated_code_review_report_is_none(self) -> None:
        """When no code is available, review_report must be None (not an empty string)."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(
                tmp,
                write_code_file=False,
                state_overrides={"generated_code": None},
            )
        self.assertIsNone(result.get("review_report"))

    def test_missing_requirements_file_falls_back_to_state(self) -> None:
        """When requirements.json is absent the node falls back to state and still runs."""
        with tempfile.TemporaryDirectory() as tmp:
            # write_req_file=False → no file on disk; state still has requirements
            result = _run_reviewer_node(tmp, write_req_file=False)
        self.assertIsNotNone(result.get("review_report"))
        self.assertIsInstance(result.get("review_report"), str)

    def test_llm_exception_records_error_and_does_not_crash(self) -> None:
        """An exception raised by the LLM must be caught and recorded in state['errors']."""
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            Path("output").mkdir()
            Path("logs").mkdir()
            Path("output/generated_code.py").write_text(_SAMPLE_CODE, encoding="utf-8")
            Path("output/requirements.json").write_text(
                json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8"
            )
            state = _make_state(tmp)
            with (
                patch("agents.code_reviewer_agent.Ollama") as MockOllama,
                patch("agents.code_reviewer_agent.run_static_analysis") as mock_sa,
            ):
                mock_llm = MagicMock()
                mock_llm.invoke.side_effect = RuntimeError("Ollama connection refused")
                MockOllama.return_value = mock_llm
                mock_sa.return_value = _CLEAN_STATIC_ANALYSIS
                result = code_reviewer_node(state)
            os.chdir(original_cwd)

        self.assertTrue(
            any("Ollama connection refused" in e or "Unexpected error" in e
                for e in result.get("errors", [])),
            "LLM exception should be recorded in state['errors']",
        )

    def test_incomplete_llm_report_appends_warning(self) -> None:
        """A report missing sections causes a warning entry in state['errors']."""
        incomplete_report = "## Summary\nCode PASSES.\n## Code Quality\nOK."
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(tmp, llm_response=incomplete_report)
        self.assertTrue(
            any("missing sections" in e for e in result.get("errors", [])),
            "A missing-sections warning should appear in state['errors']",
        )


# 6. Observability log tests

class TestCodeReviewerNodeObservability(unittest.TestCase):
    """Tests that the agent writes a well-structured observability log entry."""

    def _run_and_get_log(self, tmp_dir: str) -> list[dict]:
        """Run the node and return the parsed log array."""
        log_path = Path(tmp_dir) / "logs" / "run_test.json"
        _run_reviewer_node(tmp_dir)
        return json.loads(log_path.read_text(encoding="utf-8"))

    def test_log_file_is_created(self) -> None:
        """The log file must exist after the node runs."""
        with tempfile.TemporaryDirectory() as tmp:
            _run_reviewer_node(tmp)
            log_path = Path(tmp) / "logs" / "run_test.json"
            self.assertTrue(log_path.exists(), "Log file was not created")

    def test_log_file_is_valid_json_array(self) -> None:
        """The log file must be parseable as a JSON array."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "logs" / "run_test.json"
            _run_reviewer_node(tmp)
            try:
                entries = json.loads(log_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.fail(f"Log file is not valid JSON: {exc}")
            self.assertIsInstance(entries, list)

    def test_log_entry_has_required_keys(self) -> None:
        """Every log entry must contain timestamp, agent, tool_calls, and output."""
        with tempfile.TemporaryDirectory() as tmp:
            entries = self._run_and_get_log(tmp)
        self.assertGreater(len(entries), 0)
        entry = entries[-1]
        for key in ("timestamp", "agent", "tool_calls", "output"):
            self.assertIn(key, entry, f"Log entry is missing key '{key}'")

    def test_log_entry_records_parse_review_sections_call(self) -> None:
        """The tool_calls list must include a parse_review_sections() entry."""
        with tempfile.TemporaryDirectory() as tmp:
            entries = self._run_and_get_log(tmp)
        tool_calls: list[str] = entries[-1].get("tool_calls", [])
        self.assertTrue(
            any("parse_review_sections" in tc for tc in tool_calls),
            "parse_review_sections() call should appear in tool_calls log",
        )

# 7. Property-based constraints

class TestPropertyBasedConstraints(unittest.TestCase):
    """
    Property-based tests asserting invariants that must hold for ANY run of
    code_reviewer_node, regardless of inputs.
    """

    def test_errors_is_always_a_list(self) -> None:
        """state['errors'] must always be a list, never None."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(
                tmp,
                write_code_file=False,
                state_overrides={"generated_code": None},
            )
        self.assertIsInstance(result.get("errors"), list)

    def test_review_report_is_str_or_none(self) -> None:
        """state['review_report'] must be a str or None, never another type."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_reviewer_node(tmp)
        report = result.get("review_report")
        self.assertIn(
            type(report),
            (str, type(None)),
            f"review_report must be str or None, got {type(report)}",
        )

    def test_output_deterministic_for_same_mocked_input(self) -> None:
        """Running the node twice with the same mocked LLM response yields identical output."""
        with tempfile.TemporaryDirectory() as tmp_a:
            result_a = _run_reviewer_node(tmp_a)
        with tempfile.TemporaryDirectory() as tmp_b:
            result_b = _run_reviewer_node(tmp_b)
        self.assertEqual(
            result_a.get("review_report"),
            result_b.get("review_report"),
            "Agent output must be deterministic for the same mocked LLM response",
        )

    def test_review_sections_dataclass_fields_are_always_strings(self) -> None:
        """All text fields of ReviewSections returned by parse_review_sections are str."""
        result = parse_review_sections(_SAMPLE_REVIEW_REPORT)
        for field_name in (
            "summary", "requirements_coverage", "code_quality",
            "security_edge_cases", "test_coverage_assessment", "actionable_suggestions",
        ):
            value = getattr(result, field_name)
            self.assertIsInstance(
                value, str,
                f"ReviewSections.{field_name} must be str, got {type(value)}",
            )

    def test_missing_sections_is_always_a_list(self) -> None:
        """ReviewSections.missing_sections is always a list, even for a complete report."""
        result = parse_review_sections(_SAMPLE_REVIEW_REPORT)
        self.assertIsInstance(result.missing_sections, list)
        result_empty = parse_review_sections("")
        self.assertIsInstance(result_empty.missing_sections, list)


if __name__ == "__main__":
    unittest.main()

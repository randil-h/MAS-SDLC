"""
Evaluation harness for the Code Generator Agent.

Tests validate that the agent:
  1. Produces syntactically valid Python (ast.parse).
  2. Creates the output file on disk.
  3. Does not include markdown fences in its output.
  4. Produces functions that carry type hints.
  5. Gracefully falls back to state when the requirements file is missing.
  6. Security: output does not contain dangerous patterns (os.system, eval, exec, subprocess).
  7. Code quality: validate_python_code tool accurately detects issues.
  8. Property-based: all functions in any valid output carry return type annotations.

IT22240088
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.code_generator_agent import (
    _build_prompt,
    code_generator_node,
)
from state import SDLCState
from tools.code_tools import strip_markdown_fences, validate_python_code

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SAMPLE_REQUIREMENTS: dict = {
    "feature_name": "Password Validator",
    "description": "Validates a password string against a set of strength rules.",
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
    "input_spec": "A single string representing the candidate password.",
    "output_spec": "Returns True if valid, raises ValueError with a descriptive message otherwise.",
}

_SAMPLE_CODE = '''\
def validate_password(password: str) -> bool:
    """Validate a password against strength rules.

    Parameters
    ----------
    password : str
        Candidate password string.

    Returns
    -------
    bool
        True if the password meets all requirements.

    Raises
    ------
    TypeError
        If password is not a string.
    ValueError
        If the password fails any strength rule.
    """
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


def _make_state(tmp_dir: str, requirements=None) -> SDLCState:
    """Create a minimal SDLCState pointing output and logs at a temp directory."""
    return SDLCState(
        user_prompt="Build a password validator",
        requirements=requirements,
        generated_code=None,
        test_results=None,
        review_report=None,
        log_path=str(Path(tmp_dir) / "logs" / "run_test.json"),
        errors=[],
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestStripMarkdownFences(unittest.TestCase):
    """Unit tests for the strip_markdown_fences tool in code_tools."""

    def test_removes_python_fence(self) -> None:
        """Fenced code block with ```python tag is stripped to raw code."""
        fenced = "```python\nprint('hello')\n```"
        result = strip_markdown_fences(fenced)
        self.assertNotIn("```", result)
        self.assertIn("print('hello')", result)

    def test_removes_plain_fence(self) -> None:
        """Fenced code block with plain ``` is stripped correctly."""
        fenced = "```\ndef foo(): pass\n```"
        result = strip_markdown_fences(fenced)
        self.assertNotIn("```", result)
        self.assertIn("def foo(): pass", result)

    def test_passthrough_clean_code(self) -> None:
        """Code with no fences is returned unchanged (modulo strip)."""
        clean = "def bar() -> int:\n    return 42\n"
        result = strip_markdown_fences(clean)
        self.assertIn("def bar()", result)
        self.assertNotIn("```", result)


class TestBuildPrompt(unittest.TestCase):
    """Unit tests for the prompt construction helper."""

    def test_prompt_contains_requirements_json(self) -> None:
        """The built prompt must embed the requirements JSON."""
        prompt = _build_prompt(_SAMPLE_REQUIREMENTS)
        self.assertIn("Password Validator", prompt)
        self.assertIn("functional_requirements", prompt)

    def test_prompt_contains_system_instructions(self) -> None:
        """The built prompt must include the core system persona instructions."""
        prompt = _build_prompt(_SAMPLE_REQUIREMENTS)
        self.assertIn("type hints", prompt)
        self.assertIn("docstring", prompt)


class TestCodeGeneratorNodeOutputValidity(unittest.TestCase):
    """Integration-style tests for code_generator_node using a mocked LLM."""

    def _run_node(self, tmp_dir: str, requirements=None) -> SDLCState:
        """
        Run code_generator_node with the LLM mocked to return _SAMPLE_CODE.

        Patches:
          - Ollama.invoke  → returns _SAMPLE_CODE
          - save_to_file   → writes to tmp_dir instead of output/
          - append_log     → no-ops (we don't test log format here)
        """
        original_cwd = str(_PROJECT_ROOT)
        os.chdir(tmp_dir)
        Path("output").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        try:
            state = _make_state(tmp_dir, requirements or _SAMPLE_REQUIREMENTS)
            if requirements is None:
                # Write requirements file so read_from_file succeeds
                req_path = Path("output") / "requirements.json"
                req_path.write_text(json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8")

            with patch(
                "agents.code_generator_agent.OllamaLLM"
            ) as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_CODE
                MockOllama.return_value = mock_llm

                result_state = code_generator_node(state)
        finally:
            os.chdir(original_cwd)

        return result_state

    def test_output_is_valid_python_syntax(self) -> None:
        """Generated code must be parseable by ast.parse without errors."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_node(tmp)
        code = result.get("generated_code", "")
        self.assertTrue(code, "generated_code should not be empty")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            self.fail(f"Generated code is not valid Python: {exc}")

    def test_output_file_is_created(self) -> None:
        """output/generated_code.py must exist after the node runs."""
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            req_path = Path("output") / "requirements.json"
            req_path.write_text(json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8")
            state = _make_state(tmp, _SAMPLE_REQUIREMENTS)

            with patch("agents.code_generator_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_CODE
                MockOllama.return_value = mock_llm
                code_generator_node(state)

            code_file = Path("output") / "generated_code.py"
            self.assertTrue(code_file.exists(), "output/generated_code.py was not created")
            os.chdir(original_cwd)

    def test_output_contains_no_markdown_fences(self) -> None:
        """The saved code must not contain any ``` markdown fences."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_node(tmp)
        code = result.get("generated_code", "")
        self.assertNotIn("```", code, "Markdown fences should be stripped from generated code")

    def test_generated_functions_have_type_hints(self) -> None:
        """At least one function in the output must carry type hints."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_node(tmp)
        code = result.get("generated_code", "")
        # ast inspection: look for annotated function arguments or return annotations
        tree = ast.parse(code)
        functions_with_hints = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and (node.returns is not None or any(a.annotation for a in node.args.args))
        ]
        self.assertTrue(
            len(functions_with_hints) > 0,
            "At least one function should have type hints (argument or return annotation)",
        )

    def test_no_errors_on_valid_requirements(self) -> None:
        """The node must not append errors to state when given valid requirements."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_node(tmp)
        self.assertEqual(
            result.get("errors", []),
            [],
            "No errors should be present for a successful run",
        )


class TestCodeGeneratorNodeFallback(unittest.TestCase):
    """Tests that the node handles missing or malformed inputs gracefully."""

    def test_missing_requirements_file_falls_back_to_state(self) -> None:
        """
        When output/requirements.json does not exist, the node must fall back
        to state['requirements'] and still produce generated_code.
        """
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)

            # Deliberately do NOT create output/requirements.json
            state = _make_state(tmp, requirements=_SAMPLE_REQUIREMENTS)

            with patch("agents.code_generator_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_CODE
                MockOllama.return_value = mock_llm
                result = code_generator_node(state)

            os.chdir(original_cwd)

        self.assertIsNotNone(result.get("generated_code"))
        self.assertIn("validate_password", result.get("generated_code", ""))

    def test_empty_requirements_produces_error_in_state(self) -> None:
        """
        When both the file and state['requirements'] are absent, the node must
        record an error in state['errors'] and not crash.
        """
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)

            state = _make_state(tmp, requirements=None)

            with patch("agents.code_generator_agent.OllamaLLM"):
                result = code_generator_node(state)

            os.chdir(original_cwd)

        self.assertTrue(
            len(result.get("errors", [])) > 0,
            "An error should be recorded when no requirements are available",
        )

    def test_llm_returns_fenced_code_is_cleaned(self) -> None:
        """
        When the LLM wraps its output in markdown fences, the node must strip
        them so the saved file contains only valid Python.
        """
        fenced_code = f"```python\n{_SAMPLE_CODE}\n```"

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            req_path = Path("output") / "requirements.json"
            req_path.write_text(json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8")
            state = _make_state(tmp, _SAMPLE_REQUIREMENTS)

            with patch("agents.code_generator_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = fenced_code
                MockOllama.return_value = mock_llm
                result = code_generator_node(state)

            os.chdir(original_cwd)

        code = result.get("generated_code", "")
        self.assertNotIn("```", code)
        # Must still be valid Python after fence removal
        try:
            ast.parse(code)
        except SyntaxError as exc:
            self.fail(f"Code after fence removal is not valid Python: {exc}")


class TestValidatePythonCodeTool(unittest.TestCase):
    """
    Unit tests for the validate_python_code tool (code_tools.py).

    This is the student's individual custom tool — these tests validate its
    correctness, edge-case handling, and security-detection capabilities.
    """

    def test_valid_typed_function_passes(self) -> None:
        """Well-formed, type-hinted function is marked as valid."""
        code = (
            'def add(a: int, b: int) -> int:\n'
            '    """Return the sum of a and b."""\n'
            '    return a + b\n'
        )
        result = validate_python_code(code)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.syntax_error, "")
        self.assertTrue(result.has_type_hints)
        self.assertEqual(result.function_count, 1)

    def test_syntax_error_detected(self) -> None:
        """Code with a syntax error is correctly flagged as invalid."""
        bad_code = "def foo(\n    return 1"
        result = validate_python_code(bad_code)
        self.assertFalse(result.is_valid)
        self.assertNotEqual(result.syntax_error, "")
        self.assertGreater(result.syntax_error_line, 0)

    def test_empty_string_is_invalid(self) -> None:
        """An empty string must not be accepted as valid code."""
        result = validate_python_code("")
        self.assertFalse(result.is_valid)

    def test_bare_except_generates_warning(self) -> None:
        """A bare 'except:' clause must appear in warnings, not as a hard error."""
        code = (
            'def risky() -> None:\n'
            '    """Does something risky."""\n'
            '    try:\n'
            '        pass\n'
            '    except:\n'
            '        pass\n'
        )
        result = validate_python_code(code)
        self.assertTrue(result.is_valid)
        bare_warnings = [
            w for w in result.warnings
            if "bare" in w.lower() and "except:" in w.lower()
        ]
        self.assertTrue(len(bare_warnings) > 0, "bare except should generate a warning")

    def test_missing_docstring_generates_warning(self) -> None:
        """A function without a docstring must trigger a warning."""
        code = "def no_docs(x: int) -> int:\n    return x * 2\n"
        result = validate_python_code(code)
        self.assertTrue(result.is_valid)
        doc_warnings = [w for w in result.warnings if "docstring" in w]
        self.assertTrue(len(doc_warnings) > 0, "missing docstring should generate a warning")

    def test_function_count_is_accurate(self) -> None:
        """validate_python_code correctly counts the number of function definitions."""
        code = (
            'def alpha() -> None:\n    """Alpha."""\n    pass\n\n'
            'def beta() -> None:\n    """Beta."""\n    pass\n\n'
            'def gamma() -> None:\n    """Gamma."""\n    pass\n'
        )
        result = validate_python_code(code)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.function_count, 3)

    def test_no_type_hints_detected_correctly(self) -> None:
        """Code with no type annotations reports has_type_hints=False."""
        code = 'def legacy(x, y):\n    """Old-style function."""\n    return x + y\n'
        result = validate_python_code(code)
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_type_hints)


class TestSecurityValidation(unittest.TestCase):
    """
    Security-focused evaluation of the Code Generator Agent.

    The LLM must not produce code that contains dangerous shell-execution
    patterns.  This test class acts as a lightweight property-based security
    audit on the agent's output.
    """

    # Dangerous patterns that should never appear in generated library code
    _FORBIDDEN_PATTERNS: list[str] = [
        "os.system(",
        "subprocess.call(",
        "subprocess.Popen(",
        "subprocess.run(",
        "__import__(",
        "eval(",
        "exec(",
        "compile(",
        "open(\"/etc/",
        "open('/etc/",
    ]

    def _assert_no_dangerous_patterns(self, code: str) -> None:
        """Assert that none of the forbidden shell/eval patterns appear in code."""
        for pattern in self._FORBIDDEN_PATTERNS:
            self.assertNotIn(
                pattern,
                code,
                f"Generated code must not contain dangerous pattern: '{pattern}'",
            )

    def test_sample_code_contains_no_dangerous_patterns(self) -> None:
        """The sample code used across all tests must not contain dangerous patterns."""
        self._assert_no_dangerous_patterns(_SAMPLE_CODE)

    def test_agent_output_contains_no_dangerous_patterns(self) -> None:
        """
        When the LLM returns the sample code, the final agent output must
        not contain any dangerous execution patterns.
        """
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            req_path = Path("output") / "requirements.json"
            req_path.write_text(json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8")
            state = _make_state(tmp, _SAMPLE_REQUIREMENTS)

            with patch("agents.code_generator_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_CODE
                MockOllama.return_value = mock_llm
                result = code_generator_node(state)

            os.chdir(original_cwd)

        code = result.get("generated_code", "")
        self._assert_no_dangerous_patterns(code)

    def test_injected_dangerous_code_is_flagged_by_validator(self) -> None:
        """
        If the LLM hallucinates a call to os.system(), the security check
        in tests must catch it — demonstrating the harness is effective.
        """
        dangerous_code = (
            'import os\n\n'
            'def run_command(cmd: str) -> None:\n'
            '    """Run a shell command."""\n'
            '    os.system(cmd)\n'
        )
        # The code itself is syntactically valid — our security test layer
        # (not the syntax validator) must be what catches this.
        validation = validate_python_code(dangerous_code)
        self.assertTrue(validation.is_valid, "os.system call is valid syntax")

        # Verify our test harness catches it
        try:
            self._assert_no_dangerous_patterns(dangerous_code)
            self.fail("_assert_no_dangerous_patterns should have raised AssertionError")
        except AssertionError:
            pass  # expected — the harness correctly caught the dangerous pattern


class TestPropertyBasedConstraints(unittest.TestCase):
    """
    Property-based tests that assert invariants which must hold for ANY
    output the Code Generator produces, regardless of the feature request.

    These mirror how an 'LLM-as-a-judge' evaluation would work, but use
    deterministic AST analysis instead of a second LLM call.
    """

    def _generate_with_mock(self, mock_code: str) -> str:
        """Helper: run the agent with a mocked LLM response and return generated_code."""
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            req_path = Path("output") / "requirements.json"
            req_path.write_text(json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8")
            state = _make_state(tmp, _SAMPLE_REQUIREMENTS)

            with patch("agents.code_generator_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = mock_code
                MockOllama.return_value = mock_llm
                result = code_generator_node(state)

            os.chdir(original_cwd)
        return result.get("generated_code", "")

    def test_property_all_public_functions_have_return_annotation(self) -> None:
        """
        PROPERTY: every public function (not starting with _) in generated
        code must carry a return type annotation.
        """
        code = self._generate_with_mock(_SAMPLE_CODE)
        tree = ast.parse(code)
        violations = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
            and node.returns is None
        ]
        self.assertEqual(
            violations,
            [],
            f"Public functions without return annotation: {violations}",
        )

    def test_property_output_is_deterministically_reproducible(self) -> None:
        """
        PROPERTY: running the agent twice with the same mocked LLM response
        must produce byte-for-byte identical output (no randomness injected).
        """
        code_a = self._generate_with_mock(_SAMPLE_CODE)
        code_b = self._generate_with_mock(_SAMPLE_CODE)
        self.assertEqual(code_a, code_b, "Agent output must be deterministic for the same input")

    def test_property_state_errors_list_is_always_a_list(self) -> None:
        """
        PROPERTY: state['errors'] must always be a list, never None,
        regardless of whether the run succeeded or failed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            state = _make_state(tmp, requirements=None)  # deliberately broken input

            with patch("agents.code_generator_agent.OllamaLLM"):
                result = code_generator_node(state)

            os.chdir(original_cwd)

        self.assertIsInstance(
            result.get("errors"),
            list,
            "state['errors'] must always be a list",
        )

    def test_property_log_file_is_valid_json_array(self) -> None:
        """
        PROPERTY: the observability log file must always be a valid JSON array
        after the agent runs, even when the run encounters errors.
        """
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = str(_PROJECT_ROOT)
            os.chdir(tmp)
            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            req_path = Path("output") / "requirements.json"
            req_path.write_text(json.dumps(_SAMPLE_REQUIREMENTS), encoding="utf-8")
            log_file = Path(tmp) / "logs" / "run_test.json"
            state: SDLCState = SDLCState(
                user_prompt="test",
                requirements=_SAMPLE_REQUIREMENTS,
                generated_code=None,
                test_results=None,
                review_report=None,
                log_path=str(log_file),
                errors=[],
            )

            with patch("agents.code_generator_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_CODE
                MockOllama.return_value = mock_llm
                code_generator_node(state)

            os.chdir(original_cwd)

            self.assertTrue(log_file.exists(), "Log file must be created")
            try:
                entries = json.loads(log_file.read_text())
            except json.JSONDecodeError as exc:
                self.fail(f"Log file is not valid JSON: {exc}")

            self.assertIsInstance(entries, list, "Log file must contain a JSON array")
            self.assertGreater(len(entries), 0, "Log file must contain at least one entry")
            entry = entries[-1]
            for required_key in ("timestamp", "agent", "tool_calls", "output"):
                self.assertIn(
                    required_key, entry,
                    f"Log entry must contain key '{required_key}'"
                )


if __name__ == "__main__":
    unittest.main()

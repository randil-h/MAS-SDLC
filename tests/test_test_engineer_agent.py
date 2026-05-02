import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.test_engineer_agent import test_engineer_node
from state import SDLCState

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


_SAMPLE_CODE = """
def add(a: int, b: int) -> int:
    return a + b
"""


_SAMPLE_TESTS = """
def test_add_happy_path():
    assert add(1,2) == 3
"""


def _make_state(tmp_dir: str) -> SDLCState:
    return SDLCState(
        user_prompt="test",
        requirements={
            "functional_requirements": ["add numbers"],
            "edge_cases": ["zero"]
        },
        generated_code=_SAMPLE_CODE,
        test_results=None,
        review_report=None,
        log_path=str(Path(tmp_dir) / "logs" / "test.json"),
        errors=[],
    )


class TestTestEngineerNode(unittest.TestCase):

    def _run_node(self, tmp_dir: str):
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)

        Path("output").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        try:
            state = _make_state(tmp_dir)

            with patch("agents.test_engineer_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = _SAMPLE_TESTS
                MockOllama.return_value = mock_llm

                with patch("agents.test_engineer_agent.run_pytest") as mock_pytest:
                    mock_pytest.return_value = {
                        "passed": 1,
                        "failed": 0,
                        "errors": 0
                    }

                    result = test_engineer_node(state)

        finally:
            os.chdir(original_cwd)

        return result

    # -------------------------------------------------------

    def test_generates_test_file(self):
        """Test suite file should be created."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_node(tmp)

            test_file = Path(tmp) / "output" / "test_suite.py"
            self.assertTrue(test_file.exists())

    # -------------------------------------------------------

    def test_generated_tests_are_valid_python(self):
        """Generated test code must be syntactically valid."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run_node(tmp)

            code = (Path(tmp) / "output" / "test_suite.py").read_text()

            try:
                ast.parse(code)
            except SyntaxError as e:
                self.fail(f"Generated test code invalid: {e}")

    # -------------------------------------------------------

    def test_pytest_results_structure(self):
        """Test results must contain required keys."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_node(tmp)

        test_results = result["test_results"]

        self.assertIn("passed", test_results)
        self.assertIn("failed", test_results)
        self.assertIn("errors", test_results)

    # -------------------------------------------------------

    def test_no_markdown_fences(self):
        """Generated test code should not contain markdown fences."""
        fenced_output = "```python\n" + _SAMPLE_TESTS + "\n```"

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)

            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)

            state = _make_state(tmp)

            with patch("agents.test_engineer_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = fenced_output
                MockOllama.return_value = mock_llm

                with patch("agents.test_engineer_agent.run_pytest") as mock_pytest:
                    mock_pytest.return_value = {"passed": 1, "failed": 0, "errors": 0}

                    test_engineer_node(state)

            os.chdir(original_cwd)

            code = (Path(tmp) / "output" / "test_suite.py").read_text()
            self.assertNotIn("```", code)

    # -------------------------------------------------------

    def test_handles_invalid_python_from_llm(self):
        """Invalid Python from LLM should be caught and logged as error."""
        bad_code = "def test(:\n pass"

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)

            Path("output").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)

            state = _make_state(tmp)

            with patch("agents.test_engineer_agent.OllamaLLM") as MockOllama:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = bad_code
                MockOllama.return_value = mock_llm

                result = test_engineer_node(state)

            os.chdir(original_cwd)

        self.assertTrue(len(result["errors"]) > 0)


if __name__ == "__main__":
    unittest.main()
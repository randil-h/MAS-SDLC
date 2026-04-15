"""
Code validation and sanitisation utilities for the MAS SDLC pipeline.

Used by the Code Generator Agent to verify that LLM-produced Python source
is syntactically correct and free of obviously unsafe patterns before the
file is written to disk.
"""

import ast
import re
import sys
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """
    Structured result returned by :func:`validate_python_code`.

    Attributes
    ----------
    is_valid : bool
        True when the code passed all checks without errors.
    syntax_error : str
        Human-readable description of the syntax error, empty if none.
    syntax_error_line : int
        1-based line number of the syntax error, 0 if no error.
    warnings : list[str]
        Non-fatal observations (e.g. missing type hints, bare excepts).
    function_count : int
        Number of top-level function definitions found in the module.
    class_count : int
        Number of top-level class definitions found in the module.
    has_type_hints : bool
        True when at least one function carries argument or return annotations.
    """

    is_valid: bool = False
    syntax_error: str = ""
    syntax_error_line: int = 0
    warnings: list[str] = field(default_factory=list)
    function_count: int = 0
    class_count: int = 0
    has_type_hints: bool = False


def validate_python_code(code: str) -> ValidationResult:
    """
    Validate a Python source string for syntax correctness and code quality.

    The function performs three layers of analysis:

    1. **Syntax check** — ``ast.parse`` is used to confirm the code compiles.
       If it fails, the result is immediately returned with ``is_valid=False``
       and a precise error description including the offending line number.

    2. **Structure analysis** — The AST is walked to count functions/classes
       and to detect whether type hints are present on any function.

    3. **Quality warnings** — The AST is inspected for common issues:
       - Functions without a docstring.
       - Bare ``except:`` clauses that swallow all exceptions.
       - Top-level executable statements outside ``if __name__ == "__main__"``.

    Parameters
    ----------
    code : str
        Raw Python source code string to validate. May be empty.

    Returns
    -------
    ValidationResult
        A fully-populated result object.  Callers should check ``is_valid``
        before using the generated code.  ``warnings`` may be non-empty even
        when ``is_valid`` is True.

    Examples
    --------
    >>> result = validate_python_code("def add(a: int, b: int) -> int:\\n    return a + b")
    >>> result.is_valid
    True
    >>> result.has_type_hints
    True
    """
    result = ValidationResult()

    if not code or not code.strip():
        result.syntax_error = "Code string is empty."
        return result

    # ------------------------------------------------------------------ #
    # Layer 1 — Syntax check
    # ------------------------------------------------------------------ #
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.syntax_error = (
            f"SyntaxError on line {exc.lineno}: {exc.msg} "
            f"(text: {exc.text!r})"
        )
        result.syntax_error_line = exc.lineno or 0
        return result
    except Exception as exc:
        result.syntax_error = f"Unexpected parse error: {exc}"
        return result

    result.is_valid = True

    # ------------------------------------------------------------------ #
    # Layer 2 — Structure analysis
    # ------------------------------------------------------------------ #
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            result.function_count += 1
            # Type hints: return annotation OR any annotated argument
            if node.returns is not None or any(
                arg.annotation is not None for arg in node.args.args
            ):
                result.has_type_hints = True

        elif isinstance(node, ast.ClassDef):
            result.class_count += 1

    # ------------------------------------------------------------------ #
    # Layer 3 — Quality warnings
    # ------------------------------------------------------------------ #
    _check_docstrings(tree, result)
    _check_bare_excepts(tree, result)
    _check_top_level_statements(tree, result)

    return result


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #


def _check_docstrings(tree: ast.Module, result: ValidationResult) -> None:
    """
    Append a warning for every function or class that lacks a docstring.

    Parameters
    ----------
    tree : ast.Module
        Parsed AST of the source code.
    result : ValidationResult
        Result object to mutate with any discovered warnings.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                kind = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
                result.warnings.append(
                    f"Line {node.lineno}: {kind} '{node.name}' is missing a docstring."
                )


def _check_bare_excepts(tree: ast.Module, result: ValidationResult) -> None:
    """
    Append a warning for every bare ``except:`` clause found in the AST.

    Bare excepts catch ``SystemExit`` and ``KeyboardInterrupt``, which is
    almost always unintentional and can mask serious errors.

    Parameters
    ----------
    tree : ast.Module
        Parsed AST of the source code.
    result : ValidationResult
        Result object to mutate with any discovered warnings.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            result.warnings.append(
                f"Line {node.lineno}: bare 'except:' clause detected. "
                "Use 'except Exception:' or a more specific exception type."
            )


def _check_top_level_statements(tree: ast.Module, result: ValidationResult) -> None:
    """
    Warn about top-level executable statements outside an ``if __name__`` guard.

    Top-level calls and assignments (other than imports, function/class defs,
    and the ``if __name__ == '__main__'`` guard) suggest the module has
    side-effects on import, which violates the code generator's module contract.

    Parameters
    ----------
    tree : ast.Module
        Parsed AST of the source code.
    result : ValidationResult
        Result object to mutate with any discovered warnings.
    """
    safe_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,       # module-level docstrings are Expr nodes
    )

    for node in tree.body:
        if isinstance(node, ast.If):
            # Allow `if __name__ == "__main__":` guards
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                continue
            result.warnings.append(
                f"Line {node.lineno}: top-level 'if' statement outside "
                "__name__ guard may cause unintended side-effects on import."
            )
        elif isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            # Top-level function call (e.g. print(), run())
            result.warnings.append(
                f"Line {node.lineno}: top-level expression statement detected. "
                "Consider wrapping in 'if __name__ == \"__main__\":'."
            )


def strip_markdown_fences(code: str) -> str:
    """
    Extract Python source code from an LLM response that may contain prose and
    markdown code fences.

    Handles both ` ```python ` and plain ` ``` ` variants.

    When at least one fenced block is present the function returns **only** the
    content inside fence markers, discarding any surrounding prose.  This
    prevents LLM preambles such as "Here is a possible implementation:" from
    being included in the extracted code and causing syntax errors.

    When no fence markers are found at all the entire string is returned
    as-is (stripped), on the assumption that the LLM emitted raw code.

    Parameters
    ----------
    code : str
        Raw LLM response that may contain markdown fencing and/or prose.

    Returns
    -------
    str
        Cleaned source code with fence lines and surrounding prose removed,
        and leading/trailing whitespace stripped.

    Examples
    --------
    >>> raw = "Here is the code:\\n```python\\nprint('hello')\\n```"
    >>> strip_markdown_fences(raw)
    "print('hello')"
    """
    lines = code.splitlines()
    inside_fence = False
    found_fence = False
    fenced_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if not inside_fence:
                inside_fence = True
                found_fence = True
            else:
                inside_fence = False
            continue
        if inside_fence:
            fenced_lines.append(line)

    # If fences were found, return only what was inside them.
    # Otherwise fall back to returning the whole response (raw code path).
    if found_fence:
        return "\n".join(fenced_lines).strip()

    return code.strip()

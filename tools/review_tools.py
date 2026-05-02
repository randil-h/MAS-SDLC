"""
Review parsing utilities for the MAS SDLC pipeline.

Used by the Code Reviewer Agent to extract structured data from the Markdown
report produced by the LLM, making review output machine-readable and
programmatically verifiable by the test harness.
"""

import re
from dataclasses import dataclass, field

# The six section headings the Code Reviewer Agent is instructed to produce,
# in the order they must appear.  Each tuple is (field_name, canonical_heading).
_REQUIRED_SECTIONS: list[tuple[str, str]] = [
    ("summary",                "## Summary"),
    ("requirements_coverage",  "## Requirements Coverage"),
    ("code_quality",           "## Code Quality"),
    ("security_edge_cases",    "## Security & Edge Cases"),
    ("test_coverage_assessment", "## Test Coverage Assessment"),
    ("actionable_suggestions", "## Actionable Suggestions"),
]

# Verdict patterns ordered longest-first so "CONDITIONALLY PASSES" is matched
# before the shorter "PASSES" substring.
_VERDICT_PATTERNS: list[tuple[str, str]] = [
    (r"\bCONDITIONALLY\s+PASSES?\b", "CONDITIONALLY PASSES"),
    (r"\bPASSES?\b",                  "PASSES"),
    (r"\bFAILS?\b",                   "FAILS"),
]


@dataclass
class ReviewSections:
    """
    Structured representation of the Code Reviewer Agent's Markdown output.

    Attributes
    ----------
    summary : str
        Content of the ``## Summary`` section.
    requirements_coverage : str
        Content of the ``## Requirements Coverage`` section.
    code_quality : str
        Content of the ``## Code Quality`` section.
    security_edge_cases : str
        Content of the ``## Security & Edge Cases`` section.
    test_coverage_assessment : str
        Content of the ``## Test Coverage Assessment`` section.
    actionable_suggestions : str
        Content of the ``## Actionable Suggestions`` section.
    is_complete : bool
        ``True`` when all six required sections were found in the report.
    missing_sections : list[str]
        Canonical heading strings of any sections absent from the report.
    verdict : str
        Overall verdict extracted from the Summary section.
        One of ``"PASSES"``, ``"FAILS"``, ``"CONDITIONALLY PASSES"``, or
        ``"UNKNOWN"`` when no recognisable verdict phrase is found.
    """

    summary: str = ""
    requirements_coverage: str = ""
    code_quality: str = ""
    security_edge_cases: str = ""
    test_coverage_assessment: str = ""
    actionable_suggestions: str = ""
    is_complete: bool = False
    missing_sections: list[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"


def parse_review_sections(report: str) -> ReviewSections:
    """
    Parse a structured Markdown code review report into its constituent sections.

    Locates each of the six required ``##``-level headings produced by the Code
    Reviewer Agent's system prompt and extracts the text between consecutive
    headings.  Also extracts the overall verdict from the Summary section.

    Only ``##``-level headings (exactly two hashes) are treated as section
    boundaries; ``###`` sub-headings inside a section are included as content.
    Heading matching is case-insensitive and tolerates trailing colons or extra
    whitespace added by the LLM.

    Parameters
    ----------
    report : str
        Raw Markdown string output from the Code Reviewer Agent's LLM call.
        May be empty, malformed, or missing sections if the LLM deviated from
        the system prompt.

    Returns
    -------
    ReviewSections
        A fully-populated dataclass.  ``is_complete`` is ``True`` only when all
        six sections are present.  ``missing_sections`` lists any absent
        canonical headings.  ``verdict`` is ``"UNKNOWN"`` when no recognisable
        verdict phrase is found in the Summary.

    Examples
    --------
    >>> report = "## Summary\\nCode PASSES review.\\n## Requirements Coverage\\nAll met."
    >>> result = parse_review_sections(report)
    >>> result.verdict
    'PASSES'
    >>> result.is_complete
    False
    """
    result = ReviewSections()

    if not report or not report.strip():
        result.missing_sections = [heading for _, heading in _REQUIRED_SECTIONS]
        return result

    # Build a lookup: normalised heading text → field name
    heading_lookup: dict[str, str] = {
        heading.lower().rstrip(":"): field_name
        for field_name, heading in _REQUIRED_SECTIONS
    }

    lines = report.splitlines()

    # Walk lines and record (field_name, content_start_line_index) for every
    # recognised ## heading.  ### or deeper headings are ignored.
    section_positions: list[tuple[str, int]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Must start with exactly "## " — excludes ###, ####, etc.
        if not stripped.startswith("## "):
            continue
        normalised = stripped.lower().rstrip(":")
        field_name = heading_lookup.get(normalised)
        if field_name is not None:
            # Content starts on the line after the heading
            section_positions.append((field_name, i + 1))

    # Extract content between consecutive recognised headings
    found_fields: set[str] = set()
    for idx, (field_name, start) in enumerate(section_positions):
        end = (
            section_positions[idx + 1][1] - 1
            if idx + 1 < len(section_positions)
            else len(lines)
        )
        content = "\n".join(lines[start:end]).strip()
        setattr(result, field_name, content)
        found_fields.add(field_name)

    result.missing_sections = [
        heading
        for field_name, heading in _REQUIRED_SECTIONS
        if field_name not in found_fields
    ]
    result.is_complete = len(result.missing_sections) == 0
    result.verdict = _extract_verdict(result.summary)

    return result


def _extract_verdict(summary: str) -> str:
    """
    Extract the overall verdict from the Summary section text.

    Checks patterns longest-first so ``"CONDITIONALLY PASSES"`` is matched
    before the shorter ``"PASSES"`` substring.

    Parameters
    ----------
    summary : str
        Text content of the ``## Summary`` section.

    Returns
    -------
    str
        One of ``"PASSES"``, ``"FAILS"``, ``"CONDITIONALLY PASSES"``, or
        ``"UNKNOWN"``.
    """
    if not summary:
        return "UNKNOWN"
    for pattern, verdict in _VERDICT_PATTERNS:
        if re.search(pattern, summary, re.IGNORECASE):
            return verdict
    return "UNKNOWN"

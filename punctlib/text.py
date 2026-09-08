"""Shared text normalisation used before punctuation extraction."""

from __future__ import annotations

import re


GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",
    re.IGNORECASE,
)
LEGAL_PREAMBLE = re.compile(
    r"This eBook is for the use of anyone anywhere.*?(?:\n\n|\r\n\r\n)",
    re.IGNORECASE | re.DOTALL,
)


def strip_boilerplate(text: str) -> str:
    """Remove the same Project Gutenberg wrapper omitted by the parse cache."""
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end() :]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return LEGAL_PREAMBLE.sub("", text, count=1)


def normalise_text(text: str, *, strip: bool = True) -> str:
    """Apply the frozen preprocessing policy before tokenisation."""
    if strip:
        text = strip_boilerplate(text)
    return text.replace("...", "^")

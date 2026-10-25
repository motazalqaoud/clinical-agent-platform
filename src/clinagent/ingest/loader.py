"""Load guideline documents and split them into heading-scoped sections.

Sections carry their heading so every citation can point a clinician to the
exact part of the guideline it came from, not just the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass
class Section:
    source: str
    title: str
    heading: str
    text: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def split_sections(text: str, source: str) -> list[Section]:
    """Split a markdown document into sections at level-2+ headings.

    The level-1 heading (if any) becomes the document title. Text before the first
    level-2 heading is kept as an ``overview`` section.
    """
    title_match = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else Path(source).stem

    sections: list[Section] = []
    matches = [m for m in _HEADING.finditer(text) if len(m.group(1)) >= 2]
    starts = [m.start() for m in matches] + [len(text)]

    preamble = text[: starts[0]] if matches else text
    preamble = re.sub(r"^#\s+.*$", "", preamble, count=1, flags=re.MULTILINE).strip()
    if preamble:
        sections.append(Section(source, title, "overview", preamble))

    for m, end in zip(matches, starts[1:], strict=True):
        body = text[m.end() : end].strip()
        if body:
            sections.append(Section(source, title, _slug(m.group(2)), body))
    return sections


def load_directory(path: str | Path) -> list[Section]:
    """Load every supported file under ``path`` (recursively) as sections."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    sections: list[Section] = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
            rel = file.relative_to(root).as_posix()
            sections.extend(split_sections(file.read_text(encoding="utf-8"), rel))
    return sections

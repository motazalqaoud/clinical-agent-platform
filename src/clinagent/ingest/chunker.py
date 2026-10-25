"""Dosage-safe chunking.

Chunk boundaries are placed at sentence ends, and never inside a protected clinical
phrase: a dosage ("500 mg", "15 mg/kg/day"), a lab threshold ("eGFR < 30"), a blood
pressure ("140/90 mmHg") or a frequency abbreviation ("BID"). Size limits are soft;
phrase integrity is hard. A chunk may run long before "500" and "mg" are ever split
into different chunks, because a retriever that returns half a dose is worse than one
that returns nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DOSAGE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:mg|mcg|g|mL|ml|IU|units?|mmol)\b(?:/kg)?(?:/(?:day|dose|h|hr|min))?",
    re.IGNORECASE,
)
LAB_VALUE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*\s*(?:[<>=≤≥]|<=|>=)\s*\d+(?:\.\d+)?%?")
BLOOD_PRESSURE_PATTERN = re.compile(r"\d{2,3}/\d{2,3}\s*mmHg", re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(r"\b(?:QD|BID|TID|QID|PRN|QHS|q\d{1,2}h)\b")

PROTECTED_PATTERNS = (DOSAGE_PATTERN, LAB_VALUE_PATTERN, BLOOD_PRESSURE_PATTERN, FREQUENCY_PATTERN)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\-*])|\n{2,}|\n(?=[-*]\s)")


@dataclass(frozen=True)
class ChunkerConfig:
    target_chunk_size: int = 600
    overlap: int = 80
    min_chunk_size: int = 80


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_protected_spans(text: str) -> list[tuple[int, int]]:
    """Merged character spans of every protected clinical phrase in ``text``."""
    return _merge([(m.start(), m.end()) for p in PROTECTED_PATTERNS for m in p.finditer(text)])


def _inside(pos: int, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    for start, end in spans:
        if start < pos < end:
            return start, end
    return None


def _sentence_spans(text: str, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cuts = {0, len(text)}
    for m in _SENTENCE_BOUNDARY.finditer(text):
        if _inside(m.start(), protected) is None:
            cuts.add(m.start())
    ordered = sorted(cuts)
    return [(a, b) for a, b in zip(ordered, ordered[1:], strict=False) if a < b]


def _hard_split(start: int, end: int, protected: list[tuple[int, int]], size: int):
    pos = start
    while pos < end:
        cut = min(pos + size, end)
        hit = _inside(cut, protected)
        if hit:
            cut = hit[1]  # extend forward rather than cut a protected phrase
        yield pos, max(cut, pos + 1)
        pos = max(cut, pos + 1)


def chunk_text(text: str, config: ChunkerConfig | None = None) -> list[str]:
    """Split ``text`` into overlapping chunks that never cut a protected phrase."""
    cfg = config or ChunkerConfig()
    if not text.strip():
        return []
    protected = find_protected_spans(text)
    sentences = _sentence_spans(text, protected)
    chunks: list[tuple[int, int]] = []
    i, n = 0, len(sentences)
    while i < n:
        start, end = sentences[i]
        j = i + 1
        while j < n:
            cand = sentences[j][1]
            if cand - start > cfg.target_chunk_size and end - start >= cfg.min_chunk_size:
                break
            end = cand
            j += 1
        if j == i + 1 and end - start > cfg.target_chunk_size:
            chunks.extend(_hard_split(start, end, protected, cfg.target_chunk_size))
        else:
            chunks.append((start, end))
        if j >= n:
            break
        if cfg.overlap > 0 and j - 1 > i:
            m = j - 1
            while end - sentences[m][0] < cfg.overlap and m > i + 1:
                m -= 1
            i = m
        else:
            i = j
    return [text[a:b].strip() for a, b in chunks if text[a:b].strip()]


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text``, never split inside a protected clinical phrase."""
    protected = find_protected_spans(text)
    return [text[a:b].strip() for a, b in _sentence_spans(text, protected) if text[a:b].strip()]

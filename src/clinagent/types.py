"""Shared data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceKind = Literal["guideline", "fhir"]


@dataclass
class Source:
    """One piece of evidence an answer may cite.

    Attributes:
        ref: Stable reference, e.g. ``"cap.md#dosing/2"`` or ``"AllergyIntolerance/al-001"``.
        kind: Where it came from: a guideline chunk or a FHIR resource.
        title: Human-readable label shown next to the citation.
        text: The evidence text itself.
        score: Retrieval similarity for guideline chunks; 1.0 for FHIR facts.
    """

    ref: str
    kind: SourceKind
    title: str
    text: str
    score: float = 1.0


@dataclass
class GuardrailEvent:
    """A guardrail that fired during a request, kept for audit and metrics."""

    guard: str
    action: Literal["redacted", "blocked", "quarantined", "flagged"]
    detail: str


@dataclass
class Citation:
    index: int
    ref: str
    title: str


@dataclass
class Answer:
    """The service's response to one question."""

    text: str
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = field(default_factory=list)
    refusal_reason: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None

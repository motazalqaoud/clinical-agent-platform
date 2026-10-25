"""Prompt-injection screening for retrieved content and tool output.

Anything that reaches the model from a document store or an external system is
untrusted data. A guideline uploaded by mistake -- or on purpose -- can contain text
addressed to the model ("ignore previous instructions..."). Matching evidence is
quarantined: removed from the context, recorded as a guardrail event, and counted in
metrics so a poisoned document is noticed rather than silently absorbed.

Pattern matching is a cheap first layer, not a complete defence. The structural
defences matter more: evidence is passed as quoted, numbered data; the model's answer
is verified against that evidence (see generation/verify.py); and the model is given
no tools that can act on the outside world.
"""

from __future__ import annotations

import re

from clinagent.types import GuardrailEvent, Source

_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"\b(?:ignore|disregard|forget|override)\b.{0,30}\b(?:previous|prior|above|earlier|all)\b"
        r".{0,20}\b(?:instructions?|rules|prompts?|messages?)\b",
        r"\byou are now\b",
        r"\b(?:developer|jailbreak|god) mode\b",
        r"\b(?:reveal|print|show|repeat)\b.{0,30}\b(?:system prompt|instructions|hidden)\b",
        r"^\s*(?:SYSTEM|ASSISTANT|DEVELOPER)\s*:",
        r"\bdo not (?:cite|mention|disclose|tell)\b",
        r"<\s*/?\s*(?:system|instructions?)\s*>",
    )
)


def injection_score(text: str) -> int:
    """Number of distinct injection patterns present in ``text``."""
    return sum(1 for p in _PATTERNS if p.search(text))


def screen(sources: list[Source]) -> tuple[list[Source], list[GuardrailEvent]]:
    """Split sources into clean evidence and quarantine events."""
    clean: list[Source] = []
    events: list[GuardrailEvent] = []
    for src in sources:
        score = injection_score(src.text)
        if score:
            events.append(
                GuardrailEvent("prompt_injection", "quarantined", f"{src.ref} ({score} pattern(s))")
            )
        else:
            clean.append(src)
    return clean, events

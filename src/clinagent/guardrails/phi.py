"""Direct-identifier detection and redaction.

Scope, stated plainly: this catches *structured* direct identifiers -- SSNs, phone
numbers, email addresses, medical record numbers, street addresses and labelled dates
of birth. It does not reliably catch free-text personal names; that needs an NER model
(e.g. Presidio or a clinical de-identification model) and is listed as an open risk in
docs/risk-register.md. The service's main defence is architectural: identifiers are
never fetched into evidence in the first place (see fhir/context.py). This module is
the second line, applied to user input before it is logged or sent to a model, and to
model output before it is returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clinagent.types import GuardrailEvent


@dataclass(frozen=True)
class _Detector:
    label: str
    pattern: re.Pattern[str]


DETECTORS: tuple[_Detector, ...] = (
    _Detector("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    _Detector("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    _Detector(
        "MRN",
        re.compile(
            r"\b(?:MRN|medical record (?:number|no\.?))\s*[:#]?\s*[A-Z0-9-]{4,}\b|\bSYN-\d{4}\b",
            re.IGNORECASE,
        ),
    ),
    _Detector(
        "DOB",
        re.compile(
            r"\b(?:DOB|date of birth|born(?: on)?)\s*[:#]?\s*"
            r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})",
            re.IGNORECASE,
        ),
    ),
    _Detector(
        "PHONE",
        re.compile(r"(?<![\w/.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w/])"),
    ),
    _Detector(
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b"
        ),
    ),
)


def redact(text: str, where: str) -> tuple[str, list[GuardrailEvent]]:
    """Replace every detected identifier with ``[LABEL]``.

    Returns the redacted text and one event per identifier type found. Events record the
    type and count, never the identifier itself, so audit logs stay clean.
    """
    events: list[GuardrailEvent] = []
    for det in DETECTORS:
        text, count = det.pattern.subn(f"[{det.label}]", text)
        if count:
            events.append(GuardrailEvent("phi", "redacted", f"{count} {det.label} in {where}"))
    return text, events


_IDENTIFIER_REQUEST = re.compile(
    r"\b(?:what|give|show|tell|list|provide|reveal)\b.{0,40}\b(?:ssn|social security|phone|"
    r"telephone|e-?mail|address|mrn|medical record number|date of birth|dob|full name)\b",
    re.IGNORECASE,
)


def is_identifier_request(question: str) -> bool:
    """True when the question asks the service to disclose a patient identifier."""
    return bool(_IDENTIFIER_REQUEST.search(question))

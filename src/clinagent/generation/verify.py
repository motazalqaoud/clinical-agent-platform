"""Check a generated answer against its evidence before it is returned.

Two checks, both deterministic:

1. **Citation validity** -- every ``[n]`` must refer to a source that was provided.
   A citation to a source that does not exist is a fabricated reference.
2. **Quantity fidelity** -- every dose, threshold or measurement with a unit in the
   answer ("1 g", "eGFR < 30", "120 kg", "5 days") must appear verbatim (modulo
   whitespace and case) in the sources the sentence cites. A model that writes "2 g"
   where the guideline says "1 g" is blocked, not returned.

Prose that the evidence does not support but that contains no quantity is outside what
string matching can catch; that is measured offline with an LLM judge or clinician
review (see docs/risk-register.md), not claimed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clinagent.ingest.chunker import split_sentences
from clinagent.types import GuardrailEvent, Source

CITATION = re.compile(r"\[(\d+)\]")
QUANTITY = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:mg/kg(?:/day|/h)?|mL/kg/h|mL/min(?:/1\.73m2)?|mmol/L|umol/L|mmHg|mg|mcg|g|mL|kg|IU|"
    r"units?|%|hours?|days?|minutes?|breaths per minute)(?![a-zA-Z])",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower()).replace("–", "-")


@dataclass
class Verification:
    ok: bool
    cited: list[int] = field(default_factory=list)
    events: list[GuardrailEvent] = field(default_factory=list)


def verify(answer: str, sources: list[Source]) -> Verification:
    events: list[GuardrailEvent] = []
    cited_all = sorted({int(n) for n in CITATION.findall(answer)})
    invalid = [n for n in cited_all if not 1 <= n <= len(sources)]
    if invalid:
        events.append(
            GuardrailEvent("citation", "blocked", f"citations to missing sources {invalid}")
        )

    for sentence in split_sentences(answer):
        quantities = QUANTITY.findall(sentence)
        if not quantities:
            continue
        cited = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(sources)]
        if not cited:
            events.append(
                GuardrailEvent("quantity_fidelity", "blocked", f"uncited quantities {quantities}")
            )
            continue
        evidence = _norm(" ".join(sources[n - 1].text for n in cited))
        missing = [q for q in quantities if _norm(q) not in evidence]
        if missing:
            events.append(
                GuardrailEvent(
                    "quantity_fidelity", "blocked", f"quantities not in cited sources: {missing}"
                )
            )

    valid_cited = [n for n in cited_all if 1 <= n <= len(sources)]
    if not valid_cited:
        events.append(GuardrailEvent("citation", "blocked", "answer cites no evidence"))
    return Verification(ok=not events, cited=valid_cited, events=events)

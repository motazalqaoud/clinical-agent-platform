"""Answer generators.

``LLMGenerator`` is the production path. ``ExtractiveGenerator`` is a deterministic
baseline that composes an answer from verbatim evidence sentences. It cannot reason
(it will put "cefazolin 3 g for >= 120 kg" next to "weight 128 kg" without concluding
anything), but it never invents a number, it runs in milliseconds, and it gives the CI
evaluation gate a stable signal for everything *around* the model: retrieval, refusal,
guardrails, citation plumbing and latency.
"""

from __future__ import annotations

import re
from typing import Protocol

from clinagent.generation.prompt import INSUFFICIENT, SYSTEM_PROMPT, build_user_message
from clinagent.ingest.chunker import split_sentences
from clinagent.llm.clients import LLMClient
from clinagent.retrieval.embedders import tokenize
from clinagent.types import Source


class Generator(Protocol):
    name: str

    def generate(self, question: str, sources: list[Source]) -> str: ...


_UNIT_TERMS = frozenset({"kg", "mg", "ml"})


def _terms(text: str) -> set[str]:
    """Content words (and dose units) used to match evidence to a question."""
    return {
        t
        for t in tokenize(text)
        if "_" not in t and not t.isdigit() and (len(t) > 2 or t in _UNIT_TERMS)
    }


_NOISE = re.compile(r"^(?:#+\s|>\s)")


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_guideline_sentences: int = 2, max_facts: int = 2):
        self.max_guideline_sentences = max_guideline_sentences
        self.max_facts = max_facts

    def generate(self, question: str, sources: list[Source]) -> str:
        q = _terms(question)
        # Terms from the patient's own record (e.g. "anaphylaxis") count for half: they
        # steer toward the guideline sentence that applies to *this* patient.
        patient_terms = set().union(*(_terms(s.text) for s in sources if s.kind == "fhir")) - q
        scored: list[tuple[float, int, str]] = []
        for idx, src in enumerate(sources, start=1):
            if src.kind != "guideline":
                continue
            title_terms = _terms(src.title)
            for sent in split_sentences(src.text):
                if _NOISE.match(sent):
                    continue
                sent_terms = _terms(sent)
                overlap = len(q & sent_terms)
                if not overlap:
                    continue
                score = (
                    overlap
                    + 0.5 * len(q & title_terms)
                    + 0.5 * len(patient_terms & sent_terms)
                    + src.score  # prefer higher-ranked chunks on ties
                )
                scored.append((score, idx, sent))
        scored.sort(key=lambda t: -t[0])
        chosen = scored[: self.max_guideline_sentences]
        if not chosen:
            return INSUFFICIENT

        # Cross-reference: patient facts that share terms with the question or with the
        # chosen guideline sentences (a guideline sentence about "kg" pulls in weight).
        context_terms = q.union(*(_terms(s) for _, _, s in chosen))
        facts = []
        for idx, src in enumerate(sources, start=1):
            if src.kind == "fhir" and not src.ref.startswith("Patient/"):
                overlap = len(context_terms & _terms(src.text))
                if overlap:
                    facts.append((overlap, idx, src.text))
        facts.sort(key=lambda t: -t[0])

        lines = [f"{sent} [{idx}]" for _, idx, sent in chosen]
        lines += [f"{text} [{idx}]" for _, idx, text in facts[: self.max_facts]]
        return " ".join(lines)


class LLMGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.name = llm.name

    def generate(self, question: str, sources: list[Source]) -> str:
        return self.llm.complete(
            SYSTEM_PROMPT, [{"role": "user", "content": build_user_message(question, sources)}]
        ).strip()


def make_generator(spec: str) -> Generator:
    if spec in ("", "extractive"):
        return ExtractiveGenerator()
    from clinagent.llm.clients import make_llm

    return LLMGenerator(make_llm(spec))

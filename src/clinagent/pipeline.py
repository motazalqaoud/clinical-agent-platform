"""The request pipeline: one fixed, audited path from question to answer.

    question
      -> input guard        redact identifiers; refuse identifier-disclosure requests
      -> patient context    minimum-necessary FHIR facts (if a patient id is given)
      -> retrieve           top-k guideline chunks, query expanded with active conditions
      -> injection screen   quarantine evidence that addresses the model
      -> refuse?            no guideline evidence above the threshold -> refuse
      -> generate           extractive baseline or an LLM
      -> verify             citations exist; every quantity is in its cited source
      -> output guard       redact identifiers from the answer
    answer + citations + guardrail events + per-stage trace

Patient-context questions use this fixed workflow rather than a free-running agent:
every request touches the same data in the same order, which is what makes it
auditable and testable. The ReAct agent (clinagent.agent) is available for exploratory
use and is evaluated separately. See docs/adr/0001-fixed-workflow-over-agent.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinagent.fhir.context import PatientContext
from clinagent.generation.generators import Generator
from clinagent.generation.prompt import INSUFFICIENT
from clinagent.generation.verify import CITATION, verify
from clinagent.guardrails import injection, phi
from clinagent.observability.metrics import Metrics
from clinagent.observability.tracing import Trace
from clinagent.retrieval.index import GuidelineIndex
from clinagent.types import Answer, Citation, GuardrailEvent, Source

DISCLAIMER = (
    "Reference information for licensed clinicians, drawn only from the cited sources. "
    "It is not a treatment decision and does not replace clinical judgement."
)
REFUSAL_NO_EVIDENCE = "I can't answer this from the approved guidelines available to me."
REFUSAL_IDENTIFIERS = "I can't disclose patient identifiers."
REFUSAL_UNVERIFIED = (
    "I couldn't produce an answer that is fully supported by the sources, so I'm not "
    "returning one. Please consult the cited guideline sections directly."
)


@dataclass
class PipelineConfig:
    top_k: int = 4
    min_score: float = 0.12


class ClinicalQA:
    def __init__(
        self,
        index: GuidelineIndex,
        generator: Generator,
        patients: PatientContext | None = None,
        config: PipelineConfig | None = None,
        metrics: Metrics | None = None,
    ):
        self.index = index
        self.generator = generator
        self.patients = patients
        self.config = config or PipelineConfig()
        self.metrics = metrics or Metrics()

    def ask(self, question: str, patient_id: str | None = None) -> tuple[Answer, Trace]:
        trace = Trace()
        events: list[GuardrailEvent] = []
        answer = self._run(question, patient_id, trace, events)
        answer.guardrail_events = events
        answer.trace_id = trace.trace_id
        answer.latency_ms = trace.elapsed_ms
        outcome = "answered" if answer.grounded else (answer.refusal_reason or "refused")
        self.metrics.observe(outcome, answer.latency_ms, [(e.guard, e.action) for e in events])
        return answer, trace

    @staticmethod
    def _refuse(reason: str, text: str) -> Answer:
        return Answer(text=text, grounded=False, refusal_reason=reason)

    def _run(
        self, question: str, patient_id: str | None, trace: Trace, events: list[GuardrailEvent]
    ) -> Answer:
        with trace.span("input_guard") as s:
            question, redactions = phi.redact(question, "question")
            events.extend(redactions)
            s["redactions"] = len(redactions)
            if phi.is_identifier_request(question):
                events.append(GuardrailEvent("identifier_request", "blocked", "question"))
                return self._refuse("identifier_request", REFUSAL_IDENTIFIERS)

        facts: list[Source] = []
        if patient_id:
            if self.patients is None:
                return self._refuse("no_patient_source", "Patient data is not configured.")
            with trace.span("patient_context") as s:
                try:
                    facts = self.patients.gather(patient_id)
                except LookupError:
                    return self._refuse("unknown_patient", f"No record found for {patient_id}.")
                s["facts"] = len(facts)

        # Patient-aware retrieval: the patient's active conditions are appended to the
        # query, so "which antibiotic given this patient's allergy?" for a patient
        # with pneumonia retrieves the pneumonia guideline, not every allergy section.
        conditions = " ".join(
            f.text.removeprefix("Active condition: ").split(" (SNOMED")[0]
            for f in facts
            if f.ref.startswith("Condition/")
        )
        query = f"{question} {conditions}".strip()
        with trace.span("retrieve", k=self.config.top_k, patient_aware=bool(conditions)) as s:
            guideline = self.index.search(query, k=self.config.top_k)
            s["hits"] = len(guideline)
            s["top_score"] = round(guideline[0].score, 4) if guideline else 0.0

        with trace.span("injection_screen") as s:
            guideline, quarantined = injection.screen(guideline)
            facts, quarantined_facts = injection.screen(facts)
            events.extend(quarantined + quarantined_facts)
            s["quarantined"] = len(quarantined) + len(quarantined_facts)

        relevant = [g for g in guideline if g.score >= self.config.min_score]
        if not relevant:
            return self._refuse("no_evidence", REFUSAL_NO_EVIDENCE)

        sources = relevant + facts
        with trace.span("generate", generator=self.generator.name) as s:
            raw = self.generator.generate(question, sources)
            s["chars"] = len(raw)
        if raw.strip().upper().startswith(INSUFFICIENT):
            return self._refuse("no_evidence", REFUSAL_NO_EVIDENCE)

        with trace.span("verify") as s:
            check = verify(raw, sources)
            events.extend(check.events)
            s["ok"] = check.ok
        if not check.ok:
            return self._refuse("unverified", REFUSAL_UNVERIFIED)

        with trace.span("output_guard") as s:
            text, redactions = phi.redact(raw, "answer")
            events.extend(redactions)
            s["redactions"] = len(redactions)

        # Renumber citations densely (1..n) in order of first use, so the client never
        # sees gaps such as [1][4] when only two sources were cited.
        order: list[int] = []
        for n in CITATION.findall(text):
            if int(n) not in order:
                order.append(int(n))
        remap = {old: new for new, old in enumerate(order, start=1)}
        text = CITATION.sub(lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", text)
        cited = [sources[old - 1] for old in order]
        return Answer(
            text=text,
            grounded=True,
            citations=[Citation(i, s.ref, s.title) for i, s in enumerate(cited, start=1)],
            sources=cited,
        )

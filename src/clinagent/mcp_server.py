"""MCP server: the same capabilities, exposed to any MCP-capable assistant or IDE.

Tools:
    ask_clinical_question   the full guarded pipeline (retrieval, FHIR, verification)
    search_guidelines       raw guideline retrieval with references and scores
    get_patient_context     minimum-necessary facts for one patient, by section

Run over stdio (for desktop assistants) with ``clinagent mcp``, or over streamable HTTP
with ``clinagent mcp --http``. Requires ``pip install clinagent[mcp]``.
"""

from __future__ import annotations

from typing import Any, Literal

from clinagent.pipeline import DISCLAIMER, ClinicalQA
from clinagent.service import build_service

Section = Literal["all", "allergies", "conditions", "medications", "observations"]


def create_server(qa: ClinicalQA | None = None) -> Any:
    from mcp.server.mcpserver import MCPServer

    service = qa or build_service()
    server = MCPServer(
        name="clinical-agent-platform",
        instructions=(
            "Clinical reference tools over approved guidelines and FHIR patient data. "
            "Answers are evidence summaries for licensed clinicians, not treatment decisions."
        ),
    )

    @server.tool()
    def ask_clinical_question(question: str, patient_id: str | None = None) -> dict[str, Any]:
        """Answer a clinical question from approved guidelines, optionally in the context of
        one patient's FHIR record. Returns a cited answer or an explicit refusal."""
        answer, _ = service.ask(question, patient_id)
        return {
            "answer": answer.text,
            "grounded": answer.grounded,
            "refusal_reason": answer.refusal_reason,
            "citations": [
                {"index": c.index, "ref": c.ref, "title": c.title} for c in answer.citations
            ],
            "guardrail_events": [vars(e) for e in answer.guardrail_events],
            "disclaimer": DISCLAIMER,
        }

    @server.tool()
    def search_guidelines(query: str, k: int = 4) -> list[dict[str, Any]]:
        """Search the approved guideline corpus. Returns chunks with references and scores."""
        from clinagent.guardrails import injection

        hits, _ = injection.screen(service.index.search(query, k=max(1, min(k, 10))))
        return [
            {"ref": h.ref, "title": h.title, "text": h.text, "score": round(h.score, 4)}
            for h in hits
        ]

    @server.tool()
    def get_patient_context(patient_id: str, section: Section = "all") -> list[dict[str, str]]:
        """Minimum-necessary clinical facts for one patient (no identifiers)."""
        if service.patients is None:
            raise ValueError("Patient data is not configured")
        p = service.patients
        fetch = {
            "all": p.gather,
            "allergies": p.allergies,
            "conditions": p.conditions,
            "medications": p.medications,
            "observations": p.observations,
        }[section]
        return [{"ref": f.ref, "text": f.text} for f in fetch(patient_id)]

    return server

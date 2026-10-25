"""HTTP API.

    POST /v1/ask     question (+ optional patient_id) -> answer, citations, guardrail events
    GET  /healthz    liveness
    GET  /readyz     readiness: index loaded, generator configured
    GET  /metrics    Prometheus exposition

If ``CLINAGENT_API_KEY`` is set, /v1/* requires ``Authorization: Bearer <key>``. In a
real deployment this sits behind the hospital's identity provider (OIDC/SMART on FHIR);
the static key exists so the service is never accidentally exposed unauthenticated.
"""

from __future__ import annotations

import hmac
import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from clinagent import __version__
from clinagent.pipeline import DISCLAIMER, ClinicalQA
from clinagent.service import build_service
from clinagent.settings import Settings

log = logging.getLogger("clinagent.audit")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    patient_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9\-.]{1,64}$")


class CitationOut(BaseModel):
    index: int
    ref: str
    title: str
    text: str


class EventOut(BaseModel):
    guard: str
    action: str
    detail: str


class SpanOut(BaseModel):
    name: str
    duration_ms: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    guardrail_events: list[EventOut]
    disclaimer: str
    trace_id: str
    latency_ms: float
    spans: list[SpanOut]


def create_app(settings: Settings | None = None, qa: ClinicalQA | None = None) -> FastAPI:
    settings = settings or Settings()
    service = qa or build_service(settings)
    app = FastAPI(title="Clinical Agent Platform", version=__version__)

    def require_key(authorization: Annotated[str | None, Header()] = None) -> None:
        if not settings.api_key:
            return
        expected = f"Bearer {settings.api_key}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, object]:
        return {
            "status": "ready",
            "chunks": len(service.index),
            "embedder": service.index.embedder.name,
            "generator": service.generator.name,
            "patient_data": service.patients is not None,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return service.metrics.render()

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    def ask(req: AskRequest) -> AskResponse:
        answer, trace = service.ask(req.question, req.patient_id)
        # Audit log: outcome and references only -- no question text, no answer text.
        log.info(
            json.dumps(
                {
                    "trace_id": answer.trace_id,
                    "grounded": answer.grounded,
                    "refusal": answer.refusal_reason,
                    "patient_context": req.patient_id is not None,
                    "cited": [c.ref for c in answer.citations],
                    "guardrails": [f"{e.guard}:{e.action}" for e in answer.guardrail_events],
                    "latency_ms": answer.latency_ms,
                }
            )
        )
        return AskResponse(
            answer=answer.text,
            grounded=answer.grounded,
            refusal_reason=answer.refusal_reason,
            citations=[
                CitationOut(index=c.index, ref=c.ref, title=c.title, text=s.text)
                for c, s in zip(answer.citations, answer.sources, strict=True)
            ],
            guardrail_events=[EventOut(**vars(e)) for e in answer.guardrail_events],
            disclaimer=DISCLAIMER,
            trace_id=answer.trace_id or trace.trace_id,
            latency_ms=answer.latency_ms or trace.elapsed_ms,
            spans=[SpanOut(name=s.name, duration_ms=s.duration_ms) for s in trace.spans],
        )

    return app

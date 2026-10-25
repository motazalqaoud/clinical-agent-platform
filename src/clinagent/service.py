"""Build a ready-to-serve ClinicalQA from Settings. Shared by the API, CLI and MCP server."""

from __future__ import annotations

from clinagent.fhir.backend import make_backend
from clinagent.fhir.context import PatientContext
from clinagent.generation.generators import make_generator
from clinagent.pipeline import ClinicalQA, PipelineConfig
from clinagent.retrieval.index import GuidelineIndex
from clinagent.settings import Settings


def build_index(settings: Settings) -> GuidelineIndex:
    if settings.index_dir:
        return GuidelineIndex.load(settings.index_dir)
    return GuidelineIndex.from_directory(settings.corpus_dir, settings.embedder)


def build_service(settings: Settings | None = None) -> ClinicalQA:
    s = settings or Settings()
    patients = PatientContext(make_backend(s.fhir)) if s.fhir else None
    return ClinicalQA(
        index=build_index(s),
        generator=make_generator(s.llm),
        patients=patients,
        config=PipelineConfig(top_k=s.top_k, min_score=s.min_score),
    )

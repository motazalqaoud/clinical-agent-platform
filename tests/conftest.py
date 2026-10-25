from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from clinagent.fhir.backend import InMemoryFhirBackend
from clinagent.fhir.context import PatientContext
from clinagent.generation.generators import ExtractiveGenerator
from clinagent.ingest.loader import load_directory
from clinagent.pipeline import ClinicalQA
from clinagent.retrieval.embedders import TfidfEmbedder
from clinagent.retrieval.index import GuidelineIndex

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def index() -> GuidelineIndex:
    return GuidelineIndex.build(load_directory(ROOT / "corpus" / "guidelines"), TfidfEmbedder())


@pytest.fixture(scope="session")
def poisoned_index() -> GuidelineIndex:
    sections = load_directory(ROOT / "corpus" / "guidelines")
    sections += load_directory(ROOT / "evals" / "fixtures" / "poisoned")
    return GuidelineIndex.build(sections, TfidfEmbedder())


@pytest.fixture(scope="session")
def backend() -> InMemoryFhirBackend:
    return InMemoryFhirBackend.from_bundle(ROOT / "data" / "fhir" / "synthetic-bundle.json")


@pytest.fixture
def patients(backend: InMemoryFhirBackend) -> PatientContext:
    return PatientContext(backend, as_of=date(2026, 10, 25))


@pytest.fixture
def qa(index: GuidelineIndex, patients: PatientContext) -> ClinicalQA:
    return ClinicalQA(index, ExtractiveGenerator(), patients)

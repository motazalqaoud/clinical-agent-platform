"""Configuration from environment variables (12-factor), with safe offline defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    corpus_dir: str = field(
        default_factory=lambda: _env("CLINAGENT_CORPUS", str(_ROOT / "corpus" / "guidelines"))
    )
    index_dir: str = field(default_factory=lambda: _env("CLINAGENT_INDEX", ""))
    embedder: str = field(default_factory=lambda: _env("CLINAGENT_EMBEDDER", "tfidf"))
    llm: str = field(default_factory=lambda: _env("CLINAGENT_LLM", "extractive"))
    fhir: str = field(
        default_factory=lambda: _env(
            "CLINAGENT_FHIR", str(_ROOT / "data" / "fhir" / "synthetic-bundle.json")
        )
    )
    api_key: str = field(default_factory=lambda: _env("CLINAGENT_API_KEY", ""))
    top_k: int = field(default_factory=lambda: int(_env("CLINAGENT_TOP_K", "4")))
    min_score: float = field(default_factory=lambda: float(_env("CLINAGENT_MIN_SCORE", "0.12")))

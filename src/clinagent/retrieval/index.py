"""A small, persisted vector index over guideline chunks.

Brute-force inner product in NumPy. For the corpus sizes a single clinical service
indexes (thousands of guideline chunks, not millions) this is exact, fast enough
(well under a millisecond per query at 10k chunks) and removes a native dependency.
Swap in FAISS or a vector database behind the same ``search`` signature when the
corpus outgrows it -- see docs/adr/0002-numpy-index.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from clinagent.ingest.chunker import ChunkerConfig, chunk_text
from clinagent.ingest.loader import Section, load_directory
from clinagent.retrieval.embedders import Embedder, embedder_from_state, make_embedder
from clinagent.types import Source

_VECTORS = "vectors.npy"
_META = "index.json"


@dataclass
class ChunkRecord:
    ref: str
    title: str
    text: str


class GuidelineIndex:
    def __init__(self, embedder: Embedder, records: list[ChunkRecord], vectors: np.ndarray):
        self.embedder = embedder
        self.records = records
        self.vectors = vectors

    @classmethod
    def build(
        cls,
        sections: list[Section],
        embedder: Embedder,
        config: ChunkerConfig | None = None,
    ) -> GuidelineIndex:
        records: list[ChunkRecord] = []
        for sec in sections:
            for i, chunk in enumerate(chunk_text(sec.text, config)):
                records.append(
                    ChunkRecord(
                        ref=f"{sec.source}#{sec.heading}/{i}",
                        title=f"{sec.title} › {sec.heading.replace('-', ' ')}",
                        text=chunk,
                    )
                )
        if not records:
            raise ValueError("No chunks produced; is the corpus empty?")
        # Embed the heading with the chunk: a dosing table under "## Penicillin
        # allergy" should match a question about penicillin allergy even if the
        # chunk body never repeats the phrase.
        texts = [f"{r.title}. {r.text}" for r in records]
        embedder.fit(texts)
        return cls(embedder, records, embedder.encode(texts))

    @classmethod
    def from_directory(cls, path: str | Path, embedder_kind: str = "tfidf") -> GuidelineIndex:
        return cls.build(load_directory(path), make_embedder(embedder_kind))

    def search(self, query: str, k: int = 4) -> list[Source]:
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [
            Source(
                ref=self.records[i].ref,
                kind="guideline",
                title=self.records[i].title,
                text=self.records[i].text,
                score=float(scores[i]),
            )
            for i in top
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self.records)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / _VECTORS, self.vectors)
        meta = {"embedder": self.embedder.state(), "records": [asdict(r) for r in self.records]}
        (out / _META).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GuidelineIndex:
        src = Path(path)
        meta = json.loads((src / _META).read_text(encoding="utf-8"))
        return cls(
            embedder_from_state(meta["embedder"]),
            [ChunkRecord(**r) for r in meta["records"]],
            np.load(src / _VECTORS),
        )

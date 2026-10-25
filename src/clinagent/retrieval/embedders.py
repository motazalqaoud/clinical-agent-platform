"""Embedders.

Two implementations behind one interface:

* ``TfidfEmbedder`` -- deterministic, dependency-free, fitted on the corpus at index
  time. It is the default because it lets the whole service, the test suite and the CI
  evaluation gate run offline in seconds with reproducible scores.
* ``SentenceTransformerEmbedder`` -- dense semantic embeddings (``pip install
  clinagent[dense]``). Better recall on paraphrased questions; use it in deployment.

Both return L2-normalised float32 rows, so inner product equals cosine similarity and
the refusal threshold means the same thing for either backend (though its value must
be re-tuned per backend -- see ``evals/``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    """
    a an and are as at be by can do does for from has have how if in into is it its
    of on or should than that the their then there these this to was what when where
    which who why will with would
    """.split()  # noqa: SIM905
)


def _stem(token: str) -> str:
    # Deliberately light: enough to match "antibiotics"/"antibiotic" and
    # "dosing"/"dose" without a stemming dependency.
    if token.endswith("ss") or token.isdigit():
        return token
    for suffix in ("ies", "ing", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


# A deliberately small clinical synonym map, applied identically at index and query
# time. It fixes the most common lexical misses ("medications" vs "drugs", "renal" vs
# "kidney") without pretending to be a terminology service; dense embeddings or a
# UMLS-backed expansion are the real fix at scale.
_CANONICAL = {
    "medication": "drug",
    "medicine": "drug",
    "drug": "drug",
    "renal": "kidney",
    "nephro": "kidney",
    "dosing": "dose",
    "dosage": "dose",
    "dos": "dose",
    "antimicrobial": "antibiotic",
    "spo2": "saturation",
    "sat": "saturation",
    "weigh": "weight",
    "allergic": "allergy",
}


def tokenize(text: str) -> list[str]:
    words = [
        _CANONICAL.get(w, w)
        for w in (_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS)
        if len(w) > 1
    ]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    return words + bigrams


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def state(self) -> dict[str, Any]: ...


class TfidfEmbedder:
    """Sublinear TF-IDF over stemmed unigrams and bigrams, fitted on the corpus."""

    name = "tfidf"

    def __init__(self, vocab: dict[str, int] | None = None, idf: list[float] | None = None):
        self.vocab: dict[str, int] = vocab or {}
        self.idf = np.asarray(idf or [], dtype=np.float32)

    def fit(self, texts: list[str]) -> None:
        doc_freq: Counter[str] = Counter()
        for text in texts:
            doc_freq.update(set(tokenize(text)))
        terms = sorted(doc_freq)
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = len(texts)
        self.idf = np.asarray(
            [math.log((1 + n) / (1 + doc_freq[t])) + 1.0 for t in terms], dtype=np.float32
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError("TfidfEmbedder must be fitted before encoding")
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                col = self.vocab.get(term)
                if col is not None:
                    out[row, col] = (1.0 + math.log(count)) * self.idf[col]
        return _normalise(out)

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "vocab": self.vocab, "idf": self.idf.tolist()}


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> None:  # pretrained; nothing to fit
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.asarray(self._ensure().encode(texts), dtype=np.float32))

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "model_name": self.model_name}


def embedder_from_state(state: dict[str, Any]) -> Embedder:
    if state["name"] == TfidfEmbedder.name:
        return TfidfEmbedder(vocab=state["vocab"], idf=state["idf"])
    if state["name"] == SentenceTransformerEmbedder.name:
        return SentenceTransformerEmbedder(state["model_name"])
    raise ValueError(f"Unknown embedder: {state['name']}")


def make_embedder(kind: str) -> Embedder:
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind in ("dense", "sentence-transformers"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder kind '{kind}' (expected 'tfidf' or 'dense')")

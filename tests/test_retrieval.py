import numpy as np

from clinagent.ingest.loader import split_sections
from clinagent.retrieval.embedders import TfidfEmbedder, tokenize
from clinagent.retrieval.index import GuidelineIndex


def test_tokenize_stems_and_canonicalises_synonyms():
    toks = tokenize("Which medications need renal dosing?")
    assert "drug" in toks and "kidney" in toks and "dose" in toks


def test_vectors_are_normalised():
    emb = TfidfEmbedder()
    emb.fit(["alpha beta", "beta gamma"])
    norms = np.linalg.norm(emb.encode(["alpha beta gamma"]), axis=1)
    assert np.allclose(norms, 1.0)


def test_sections_carry_headings():
    secs = split_sections("# Title\n\nIntro text.\n\n## Dosing\n\nGive 1 g.", "doc.md")
    assert [s.heading for s in secs] == ["overview", "dosing"]
    assert secs[1].title == "Title"


def test_search_finds_expected_section(index):
    hits = index.search("cefazolin dose for adults weighing 120 kg or more", k=3)
    assert hits[0].ref.startswith("surgical-antibiotic-prophylaxis.md#standard-agent-and-dose")
    assert hits[0].kind == "guideline"


def test_unrelated_query_scores_zero(index):
    assert index.search("capital of France") == []


def test_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = GuidelineIndex.load(tmp_path)
    q = "oxygen saturation target asthma"
    assert [h.ref for h in loaded.search(q)] == [h.ref for h in index.search(q)]

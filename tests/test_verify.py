from clinagent.generation.verify import verify
from clinagent.types import Source

SOURCES = [
    Source(
        "a",
        "guideline",
        "CAP",
        "First-line therapy is amoxicillin 1 g orally TID. Treat for 5 days.",
    ),
    Source("b", "fhir", "Weight", "Latest Body weight: 128 kg on 2026-10-19."),
]


def test_supported_answer_passes():
    result = verify("Give amoxicillin 1 g orally TID [1]. The patient weighs 128 kg [2].", SOURCES)
    assert result.ok and result.cited == [1, 2]


def test_changed_dose_is_blocked():
    result = verify("Give amoxicillin 2 g orally TID [1].", SOURCES)
    assert not result.ok
    assert any("2 g" in e.detail for e in result.events)


def test_dose_cited_to_the_wrong_source_is_blocked():
    assert not verify("Give amoxicillin 1 g [2].", SOURCES).ok


def test_citation_to_missing_source_is_blocked():
    assert not verify("Treat for 5 days [3].", SOURCES).ok


def test_uncited_quantity_is_blocked():
    assert not verify("Treat for 5 days.", SOURCES).ok


def test_answer_with_no_citations_is_blocked():
    assert not verify("Amoxicillin is first line.", SOURCES).ok

from clinagent.ingest.chunker import (
    ChunkerConfig,
    chunk_text,
    find_protected_spans,
    split_sentences,
)

DOSE_TEXT = (
    "Give cefazolin 2 g IV for adults under 120 kg. "
    "Use vancomycin 15 mg/kg IV for anaphylaxis. "
    "Avoid nitrofurantoin when eGFR < 30 mL/min. "
    "Target 140/90 mmHg. Give amoxicillin 1 g orally TID. "
) * 6


def test_protected_spans_cover_doses_labs_bp_and_frequency():
    spans = [DOSE_TEXT[a:b] for a, b in find_protected_spans(DOSE_TEXT[:220])]
    joined = " | ".join(spans)
    for phrase in ("2 g", "15 mg/kg", "eGFR < 30", "140/90 mmHg", "TID"):
        assert phrase in joined


def test_no_chunk_boundary_splits_a_protected_phrase():
    chunks = chunk_text(
        DOSE_TEXT, ChunkerConfig(target_chunk_size=60, overlap=0, min_chunk_size=10)
    )
    assert len(chunks) > 3
    for chunk in chunks:
        # every dose that starts in a chunk must end in the same chunk
        assert not chunk.rstrip().endswith(("15", "2", "eGFR <", "140/90"))
        assert not chunk.lstrip().startswith(("mg", "g IV", "mmHg"))


def test_hard_split_extends_rather_than_cuts():
    text = "x" * 50 + " amoxicillin 1000 mg " + "y" * 50
    chunks = chunk_text(text, ChunkerConfig(target_chunk_size=60, overlap=0, min_chunk_size=1))
    assert any("1000 mg" in c for c in chunks)


def test_decimal_points_are_not_sentence_ends():
    sentences = split_sentences("Stage 1 is 1.5 to 1.9 times baseline. Next sentence.")
    assert sentences[0].startswith("Stage 1 is 1.5")
    assert len(sentences) == 2


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []

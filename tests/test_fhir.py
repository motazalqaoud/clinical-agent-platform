import pytest

from clinagent.fhir.backend import InMemoryFhirBackend


def test_search_filters_by_patient_and_status(backend):
    meds = backend.search("MedicationRequest", {"patient": "pt-003", "status": "active"})
    assert {m["id"] for m in meds} == {"med-003", "med-004"}


def test_search_sorts_and_limits(backend):
    latest = backend.search(
        "Observation", {"patient": "pt-003", "code": "2160-0", "_sort": "-date", "_count": "1"}
    )
    assert latest[0]["id"] == "obs-021"


def test_unknown_search_parameter_raises(backend):
    with pytest.raises(ValueError):
        backend.search("Observation", {"patient": "pt-001", "bogus": "1"})


def test_context_has_no_identifiers(patients):
    text = " ".join(s.text for s in patients.gather("pt-001"))
    for identifier in ("Synthetic", "Alpha", "SYN-0001", "555-01", "1968-03-14"):
        assert identifier not in text
    assert "58-year-old female" in text


def test_context_reports_trend(patients):
    creat = [s for s in patients.observations("pt-003") if "Creatinine" in s.text][0]
    assert "190" in creat.text and "previous 110" in creat.text


def test_allergy_fact_is_citable(patients):
    allergy = patients.allergies("pt-001")[0]
    assert allergy.ref == "AllergyIntolerance/al-001"
    assert "Anaphylaxis" in allergy.text and "high" in allergy.text


def test_unknown_patient_raises(patients):
    with pytest.raises(LookupError):
        patients.gather("pt-999")


def test_in_memory_backend_from_resources():
    b = InMemoryFhirBackend([{"resourceType": "Patient", "id": "p1"}])
    assert b.read("Patient", "p1") == {"resourceType": "Patient", "id": "p1"}
    assert b.read("Patient", "nope") is None

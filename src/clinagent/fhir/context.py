"""Turn FHIR resources into citable, minimum-necessary evidence.

Each fact becomes a ``Source`` whose ``ref`` is the FHIR resource reference, so an
answer that says "penicillin anaphylaxis [2]" points back to the exact
AllergyIntolerance resource. Identifiers are never copied into evidence: no name, MRN,
phone number or birth date ever reaches a prompt -- only age, sex and the clinical facts
the question needs (the HIPAA minimum-necessary principle, enforced in code).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from clinagent.fhir.backend import FhirBackend, Resource
from clinagent.types import Source


def _text(concept: dict[str, Any]) -> str:
    if concept.get("text"):
        return str(concept["text"])
    codings = concept.get("coding", [])
    return (
        str(codings[0].get("display", codings[0].get("code", "unknown"))) if codings else "unknown"
    )


def _code(concept: dict[str, Any]) -> str:
    codings = concept.get("coding", [])
    return str(codings[0].get("code", "")) if codings else ""


def _age(birth_date: str, as_of: date) -> int:
    born = date.fromisoformat(birth_date[:10])
    return as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))


def _fact(ref: str, title: str, text: str) -> Source:
    return Source(ref=ref, kind="fhir", title=title, text=text, score=1.0)


class PatientContext:
    def __init__(self, backend: FhirBackend, as_of: date | None = None):
        self.backend = backend
        self.as_of = as_of or date.today()

    def demographics(self, patient_id: str) -> list[Source]:
        patient = self.backend.read("Patient", patient_id)
        if patient is None:
            raise LookupError(f"Patient/{patient_id} not found")
        parts = []
        if patient.get("birthDate"):
            parts.append(f"{_age(patient['birthDate'], self.as_of)}-year-old")
        parts.append(patient.get("gender", "unknown sex"))
        return [
            _fact(f"Patient/{patient_id}", "Patient demographics", f"Patient: {' '.join(parts)}.")
        ]

    def conditions(self, patient_id: str) -> list[Source]:
        found = self.backend.search(
            "Condition", {"patient": patient_id, "clinical-status": "active"}
        )
        return [
            _fact(
                f"Condition/{c['id']}",
                "Active condition",
                f"Active condition: {_text(c['code'])} (SNOMED {_code(c['code'])}), "
                f"onset {c.get('onsetDateTime', 'unknown')[:10]}.",
            )
            for c in found
        ]

    def medications(self, patient_id: str) -> list[Source]:
        found = self.backend.search(
            "MedicationRequest", {"patient": patient_id, "status": "active", "_sort": "-date"}
        )
        out = []
        for m in found:
            dose = (
                "; ".join(d.get("text", "") for d in m.get("dosageInstruction", [])) or "no dosage"
            )
            out.append(
                _fact(
                    f"MedicationRequest/{m['id']}",
                    "Active medication",
                    f"Active medication: {_text(m.get('medicationCodeableConcept', {}))}, "
                    f"{dose} (ordered {m.get('authoredOn', 'unknown')[:10]}).",
                )
            )
        return out

    def allergies(self, patient_id: str) -> list[Source]:
        found = self.backend.search(
            "AllergyIntolerance", {"patient": patient_id, "clinical-status": "active"}
        )
        out = []
        for a in found:
            reactions = [
                _text(man) for r in a.get("reaction", []) for man in r.get("manifestation", [])
            ]
            out.append(
                _fact(
                    f"AllergyIntolerance/{a['id']}",
                    "Allergy",
                    f"Allergy: {_text(a.get('code', {}))}; reaction: "
                    f"{', '.join(reactions) or 'not recorded'}; "
                    f"criticality: {a.get('criticality', 'unknown')}.",
                )
            )
        return out

    def observations(self, patient_id: str) -> list[Source]:
        """Latest value per observation code, with the previous value for trends."""
        found: list[Resource] = self.backend.search(
            "Observation", {"patient": patient_id, "_sort": "-date"}
        )
        by_code: dict[str, list[Resource]] = {}
        for o in found:
            by_code.setdefault(_code(o["code"]), []).append(o)
        out = []
        for series in by_code.values():
            latest = series[0]
            q = latest.get("valueQuantity", {})
            text = (
                f"Latest {_text(latest['code'])}: {q.get('value')} {q.get('unit', '')} "
                f"on {latest.get('effectiveDateTime', '')[:10]}"
            )
            if len(series) > 1:
                prev = series[1].get("valueQuantity", {})
                text += (
                    f" (previous {prev.get('value')} {prev.get('unit', '')} on "
                    f"{series[1].get('effectiveDateTime', '')[:10]})"
                )
            out.append(_fact(f"Observation/{latest['id']}", "Observation", text + "."))
        return out

    def gather(self, patient_id: str) -> list[Source]:
        """Everything above, demographics first."""
        return (
            self.demographics(patient_id)
            + self.allergies(patient_id)
            + self.conditions(patient_id)
            + self.medications(patient_id)
            + self.observations(patient_id)
        )

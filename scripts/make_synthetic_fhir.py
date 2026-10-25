"""Generate the small, fully synthetic FHIR R4 transaction bundle used by tests and demos.

Every patient here is invented. Names use the family name "Synthetic", phone numbers
sit in the fictional 555-01xx range, and identifiers are prefixed SYN-. The bundle is
small on purpose: each patient exists to exercise one behaviour of the service
(allergy-aware prescribing, renal dose review, weight-based dosing, a distractor).

For realistic volume, generate patients with Synthea and load them into the HAPI FHIR
server with scripts/load_fhir_bundles.py instead.

Usage:
    python scripts/make_synthetic_fhir.py            # writes data/fhir/synthetic-bundle.json
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "fhir" / "synthetic-bundle.json"

SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
UCUM = "http://unitsofmeasure.org"


def patient(pid: str, given: str, gender: str, birth: str, n: int) -> dict:
    return {
        "resourceType": "Patient",
        "id": pid,
        "identifier": [{"system": "urn:synthetic:mrn", "value": f"SYN-{n:04d}"}],
        "name": [{"family": "Synthetic", "given": [given]}],
        "gender": gender,
        "birthDate": birth,
        "telecom": [{"system": "phone", "value": f"612-555-01{n:02d}"}],
    }


def condition(cid: str, pid: str, code: str, display: str, onset: str, status="active") -> dict:
    return {
        "resourceType": "Condition",
        "id": cid,
        "subject": {"reference": f"Patient/{pid}"},
        "clinicalStatus": {"coding": [{"code": status}]},
        "code": {"coding": [{"system": SNOMED, "code": code, "display": display}], "text": display},
        "onsetDateTime": onset,
    }


def med(mid: str, pid: str, rx: str, name: str, dosage: str, authored: str) -> dict:
    return {
        "resourceType": "MedicationRequest",
        "id": mid,
        "status": "active",
        "intent": "order",
        "subject": {"reference": f"Patient/{pid}"},
        "medicationCodeableConcept": {
            "coding": [{"system": RXNORM, "code": rx, "display": name}],
            "text": name,
        },
        "dosageInstruction": [{"text": dosage}],
        "authoredOn": authored,
    }


def allergy(aid: str, pid: str, criticality: str, reaction_code: str, reaction: str) -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "id": aid,
        "patient": {"reference": f"Patient/{pid}"},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "criticality": criticality,
        "code": {
            "coding": [{"system": SNOMED, "code": "764146007", "display": "Penicillin"}],
            "text": "Penicillin",
        },
        "reaction": [
            {
                "manifestation": [
                    {
                        "coding": [{"system": SNOMED, "code": reaction_code, "display": reaction}],
                        "text": reaction,
                    }
                ]
            }
        ],
    }


def obs(oid: str, pid: str, loinc: str, display: str, value: float, unit: str, when: str) -> dict:
    return {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "subject": {"reference": f"Patient/{pid}"},
        "code": {"coding": [{"system": LOINC, "code": loinc, "display": display}], "text": display},
        "valueQuantity": {"value": value, "unit": unit, "system": UCUM, "code": unit},
        "effectiveDateTime": when,
    }


def build() -> list[dict]:
    r: list[dict] = []
    # pt-001: pneumonia with penicillin anaphylaxis -> the beta-lactam must be avoided.
    r += [
        patient("pt-001", "Alpha", "female", "1968-03-14", 1),
        condition("cond-001", "pt-001", "233604007", "Pneumonia", "2026-10-21"),
        allergy("al-001", "pt-001", "high", "39579001", "Anaphylaxis"),
        obs("obs-001", "pt-001", "8310-5", "Body temperature", 38.4, "Cel", "2026-10-22T08:10:00Z"),
        obs(
            "obs-002",
            "pt-001",
            "59408-5",
            "Oxygen saturation by pulse oximetry",
            93,
            "%",
            "2026-10-22T08:10:00Z",
        ),
        obs("obs-003", "pt-001", "9279-1", "Respiratory rate", 24, "/min", "2026-10-22T08:10:00Z"),
        obs("obs-004", "pt-001", "29463-7", "Body weight", 72, "kg", "2026-10-22T08:10:00Z"),
    ]
    # pt-002: asthma exacerbation, hypoxic.
    r += [
        patient("pt-002", "Bravo", "male", "1992-07-02", 2),
        condition("cond-002", "pt-002", "195967001", "Asthma", "2009-05-01"),
        med(
            "med-001",
            "pt-002",
            "435",
            "Albuterol",
            "2 puffs inhaled every 4 hours PRN",
            "2026-01-10",
        ),
        med("med-002", "pt-002", "41126", "Fluticasone", "250 mcg inhaled BID", "2026-01-10"),
        obs(
            "obs-010",
            "pt-002",
            "59408-5",
            "Oxygen saturation by pulse oximetry",
            91,
            "%",
            "2026-10-23T14:02:00Z",
        ),
        obs("obs-011", "pt-002", "9279-1", "Respiratory rate", 28, "/min", "2026-10-23T14:02:00Z"),
    ]
    # pt-003: CKD with an AKI on top, still taking two drugs the guideline says to review.
    r += [
        patient("pt-003", "Charlie", "male", "1955-11-30", 3),
        condition(
            "cond-003", "pt-003", "433144002", "Chronic kidney disease stage 3", "2021-02-01"
        ),
        condition("cond-004", "pt-003", "14669001", "Acute renal failure syndrome", "2026-10-20"),
        med("med-003", "pt-003", "7454", "Nitrofurantoin", "100 mg orally BID", "2026-10-15"),
        med("med-004", "pt-003", "5640", "Ibuprofen", "400 mg orally TID PRN", "2026-09-01"),
        obs(
            "obs-020",
            "pt-003",
            "2160-0",
            "Creatinine [Mass/volume] in Serum or Plasma",
            110,
            "umol/L",
            "2026-09-28T09:00:00Z",
        ),
        obs(
            "obs-021",
            "pt-003",
            "2160-0",
            "Creatinine [Mass/volume] in Serum or Plasma",
            190,
            "umol/L",
            "2026-10-22T09:00:00Z",
        ),
        obs(
            "obs-022",
            "pt-003",
            "33914-3",
            "Glomerular filtration rate (eGFR)",
            28,
            "mL/min/1.73m2",
            "2026-10-22T09:00:00Z",
        ),
        obs(
            "obs-023",
            "pt-003",
            "2823-3",
            "Potassium [Moles/volume] in Serum or Plasma",
            5.6,
            "mmol/L",
            "2026-10-22T09:00:00Z",
        ),
    ]
    # pt-004: pre-operative, weight >= 120 kg, non-severe penicillin allergy.
    r += [
        patient("pt-004", "Delta", "female", "1981-04-19", 4),
        condition("cond-005", "pt-004", "239873007", "Osteoarthritis of knee", "2019-06-12"),
        allergy("al-002", "pt-004", "low", "271807003", "Eruption of skin"),
        obs("obs-030", "pt-004", "29463-7", "Body weight", 128, "kg", "2026-10-19T11:30:00Z"),
    ]
    # pt-005: distractor; nothing in the corpus applies.
    r += [
        patient("pt-005", "Echo", "female", "1997-09-08", 5),
        condition("cond-006", "pt-005", "37796009", "Migraine", "2015-03-03"),
        med("med-005", "pt-005", "37418", "Sumatriptan", "50 mg orally PRN", "2025-12-01"),
    ]
    return r


def main() -> None:
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": res,
                "request": {"method": "PUT", "url": f"{res['resourceType']}/{res['id']}"},
            }
            for res in build()
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(bundle['entry'])} resources to {OUT}")


if __name__ == "__main__":
    main()

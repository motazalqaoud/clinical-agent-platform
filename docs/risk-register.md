# Risk register

Hazards a clinical question-answering service can cause, the controls in this repository, and what remains. Severity and likelihood are qualitative (H/M/L) and assume deployment as a clinician-facing reference tool, not a decision-making device.

| # | Hazard | Cause | Control in this repo | Verified by | Residual |
| --- | --- | --- | --- | --- | --- |
| R1 | Wrong dose presented as guideline | Model alters a number; chunk splits a dose | Dosage-safe chunker; quantity-fidelity check blocks any quantity not in its cited source | `test_verify.py`, `test_llm_that_changes_a_dose_is_blocked`, chunker tests | **L** — a dose copied correctly from the *wrong* sentence of a cited source still passes |
| R2 | Fabricated reference | Model cites a source that was not provided | Citation-validity check; refusal if nothing is cited | `test_verify.py` | L |
| R3 | Confident answer to an unanswerable question | Retrieval returns loosely related text | Similarity threshold; generator `INSUFFICIENT` path; eval tracks refusal rate | eval `correct_refusal_rate` (0.83) | **M** — in-domain-sounding questions pass the threshold (`oos-acl`) |
| R4 | Patient identifiers reach the model, logs or a vendor | Identifiers in record or question | Identifiers never fetched into evidence; regex redaction of question and answer; audit log excludes text; trace attributes exclude text | `test_context_has_no_identifiers`, eval `phi_leaks = 0` | **M** — free-text names in a question are not detected (needs NER) |
| R5 | Disclosure of identifiers on request | User asks for phone/MRN/DOB | Identifier-request refusal | eval `identifier` cases | L |
| R6 | Behaviour hijacked by a poisoned document | Injection text in corpus or tool output | Pattern screen quarantines evidence; evidence passed as delimited data; no write-capable tools | eval `injection_quarantine_rate = 1.0` | **M** — paraphrased or encoded injections can evade patterns; structural controls limit impact |
| R7 | Answer ignores patient-specific contraindication | Allergy or renal function not considered | Fixed workflow always gathers allergies, conditions, medications, labs; retrieval expanded with conditions | eval `patient` cases | **M** — baseline generator juxtaposes rather than reasons; LLM reasoning must be measured |
| R8 | Outdated guidance | Corpus not maintained | Corpus is configuration; refs carry file and section | — | **H** — governance process outside the software |
| R9 | Silent quality regression | Code or corpus change | Eval suite is a required CI check with absolute safety gates | CI `eval-gate` job | L |
| R10 | Over-reliance by clinicians | Answers read as decisions | Disclaimer on every response; prompt forbids final treatment decisions | — | **M** — a human-factors question, not a software one |
| R11 | Unauthorised access | Exposed endpoint | Bearer-token check; strict input validation on `patient_id` | `test_api_key_enforced`, `test_invalid_patient_id_rejected` | **M** — production needs OIDC/SMART on FHIR and per-patient authorisation |

## Open items

1. Prose faithfulness beyond quantities (LLM judge plus clinician review sample).
2. NER-based de-identification for free text (e.g. Presidio, a clinical de-id model).
3. Per-user authorisation against the patient's care team (SMART on FHIR scopes).
4. Post-deployment monitoring of refusal rate and guardrail rates by corpus version.

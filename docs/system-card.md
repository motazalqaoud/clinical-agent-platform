# System card: clinical-agent-platform

Structured after the Good Machine Learning Practice guiding principles (FDA, Health Canada, MHRA, 2021). This is a research and portfolio system, **not a medical device**; the card shows what such documentation contains.

## Intended use

Reference look-up for licensed clinicians: what an approved guideline says about a question, optionally alongside the relevant facts from one patient's record, with every statement cited. **Not** for diagnosis, treatment decisions, patient-facing use, or use without the cited sources being checked.

## Components and data

| Component | Default | Notes |
| --- | --- | --- |
| Guideline corpus | 4 synthetic guidelines, 26 chunks | Illustrative values, written for testing. Replace with institution-approved documents. |
| Patient data | 5 synthetic FHIR R4 patients | Each exercises one behaviour (allergy, renal dosing, weight-based dosing, hypoxia, distractor). Synthea supported for volume. |
| Retrieval | TF-IDF (offline) or sentence-transformers | Patient-aware query expansion. |
| Generator | Extractive baseline, or any LLM | LLM is swappable by configuration; it never sees identifiers. |

No real patient data was used to build or evaluate this system.

## GMLP principles

| Principle | How it is addressed |
| --- | --- |
| 1. Multi-disciplinary expertise across the life cycle | Clinical safety requirements are encoded as tests and eval cases. Clinician review of the case set is a stated prerequisite for any real use. |
| 2. Good software engineering and security | Typed, linted, 56 unit tests, CI on 3 Python versions, non-root container, bearer auth, input validation, audit log without free text. |
| 3. Representative data | The synthetic corpus and patients are **not** representative of any population; stated as a limitation. Subgroup evaluation needs real, governed data. |
| 4. Training and test sets independent | No model is trained here. Eval cases are separate from the retrieval code. The refusal threshold was set by inspecting this case set, so the out-of-scope cases are not a held-out test of it. |
| 5. Reference datasets based on best available methods | Eval cases state the expected guideline section and record items explicitly; they need clinician sign-off before carrying weight. |
| 6. Design tailored to data and intended use | Dosage-safe chunking, quantity verification and minimum-necessary context follow from the intended use. |
| 7. Human-AI team performance | Answers present evidence and citations for a clinician to judge; the prompt forbids final decisions. Human-factors testing is not done. |
| 8. Testing in clinically relevant conditions | Not done. Eval is offline and synthetic. |
| 9. Clear, essential information for users | Disclaimer on every response; refusals state their reason; citations carry the evidence text. |
| 10. Monitoring of deployed models | Per-stage traces, Prometheus metrics for refusals and guardrail events, eval suite runnable on any corpus version. |

## Performance

See the Results table in the [README](../README.md). The CI numbers measure the system around the model with a deterministic generator; LLM results are reported separately on the same cases.

## Known limitations

- Similarity thresholds cannot reject in-domain-sounding out-of-scope questions (`oos-acl`).
- The extractive baseline juxtaposes evidence; it does not reason (e.g. it shows "3 g if ≥ 120 kg" next to "128 kg" without concluding).
- Free-text names are not redacted.
- Guideline currency depends on corpus governance outside the software.

See the [risk register](risk-register.md).

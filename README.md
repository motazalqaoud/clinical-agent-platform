# clinical-agent-platform

**A deployable clinical question-answering service.** It answers clinicians' questions from approved guidelines, in the context of a patient's FHIR record, and refuses rather than guesses. Every dose in an answer is checked against the source it cites before the answer is returned, and an evaluation suite gates every pull request.

[![ci](https://github.com/motazalqaoud/clinical-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/motazalqaoud/clinical-agent-platform/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> **Not a medical device and not for clinical use.** The guideline corpus and every patient in this repository are synthetic. See [corpus/README.md](corpus/README.md).

## Results

The evaluation suite ([evals/cases.yaml](evals/cases.yaml), 39 cases) runs in CI on every pull request with the deterministic extractive generator. Every change must meet the thresholds in [evals/thresholds.yaml](evals/thresholds.yaml).

| Metric | Result | Gate |
| --- | --- | --- |
| PHI leaks in answers | **0** | = 0 |
| Prompt-injection payloads repeated | **0** | = 0 |
| Prompt-injection cases quarantined | **2 / 2** | all |
| Identifiers in questions redacted | **2 / 2** | all |
| Retrieval recall@4 | 1.00 | ≥ 0.95 |
| Correct section cited | 0.94 | ≥ 0.90 |
| Patient record cited (patient questions) | 1.00 | ≥ 0.80 |
| Correct refusals (out of scope, identifier requests) | 0.83 | ≥ 0.80 |
| Latency p95 (in-process, extractive) | ~1 ms | ≤ 250 ms |

## How a request flows

```mermaid
flowchart LR
    Q[Question<br/>+ patient id] --> IG[Input guard<br/>redact identifiers<br/>refuse identifier requests]
    IG --> PC[Patient context<br/>FHIR R4, minimum necessary]
    PC --> R[Retrieve<br/>query + active conditions]
    R --> IS[Injection screen<br/>quarantine poisoned evidence]
    IS --> RF{Evidence above<br/>threshold?}
    RF -- no --> REF[Refuse]
    RF -- yes --> G[Generate<br/>extractive or LLM]
    G --> V{Verify<br/>citations exist<br/>doses match sources}
    V -- fail --> REF
    V -- pass --> OG[Output guard] --> A[Answer + citations<br/>+ guardrail events + trace]
```

Patient questions go through this **fixed, audited workflow**, not a free-running agent. Every request touches the same data in the same order, which is what makes it testable and reviewable ([ADR 0001](docs/adr/0001-fixed-workflow-over-agent.md)). A ReAct agent with the same tools and the same verification is included for exploratory use.

## Key features

- **Dosage-safe chunking.** Guideline text is never split inside a dose, lab threshold, blood pressure or dosing frequency ([chunker.py](src/clinagent/ingest/chunker.py)).
- **Dose and citation verification.** Every quantity in an answer must appear in the source its sentence cites, and every citation must point to a real source; otherwise the answer is withheld ([verify.py](src/clinagent/generation/verify.py)).
- **Prompt-injection screening.** Retrieved text that tries to instruct the model is quarantined, logged and counted in `/metrics` ([injection.py](src/clinagent/guardrails/injection.py)).
- **Minimum-necessary patient context.** Only age, sex and the relevant clinical facts reach the model; identifiers are never fetched, and structured identifiers in questions are redacted ([context.py](src/clinagent/fhir/context.py), [phi.py](src/clinagent/guardrails/phi.py)).
- **Patient-aware retrieval.** Guideline search is expanded with the patient's active conditions.
- **Built-in observability.** Per-stage timings with every response, Prometheus metrics, and an evaluation suite that runs as a required CI check.

## Quickstart

```bash
pip install -e ".[dev]"

clinagent ask "What cefazolin prophylaxis dose applies for this patient's weight?" --patient pt-004
clinagent eval                 # run the 39-case suite and enforce the gate
clinagent serve                # HTTP API on :8000
clinagent mcp                  # MCP server over stdio
```

```
Give cefazolin 2 g IV for adults under 120 kg and 3 g IV for adults weighing 120 kg or more. [1]
Patients with a non-severe penicillin allergy can usually receive cefazolin. [2]
Allergy: Penicillin; reaction: Eruption of skin; criticality: low. [3]
Latest Body weight: 128 kg on 2026-10-19. [4]
  [1] Surgical Antibiotic Prophylaxis › standard agent and dose
  [3] AllergyIntolerance/al-002
  [4] Observation/obs-030
```

### With a real model and a real FHIR server

```bash
docker compose -f deploy/docker-compose.yml --profile llm up -d
python scripts/load_fhir_bundles.py --server http://localhost:8080/fhir data/fhir   # or Synthea output
docker compose -f deploy/docker-compose.yml exec ollama ollama pull qwen2.5:3b-instruct

export CLINAGENT_LLM="openai:qwen2.5:3b-instruct@http://localhost:11434/v1"
export CLINAGENT_FHIR="http://localhost:8080/fhir"
clinagent eval --llm "$CLINAGENT_LLM" --no-gate     # measure the model on the same suite
```

Any OpenAI-compatible server works (vLLM, Ollama, LM Studio, TGI), as do `anthropic:<model>` and `transformers:<model>`.

## API

`POST /v1/ask`

```json
{ "question": "Should nitrofurantoin be avoided for this patient given their eGFR?", "patient_id": "pt-003" }
```

The response carries `answer`, `grounded`, `refusal_reason`, `citations` (with the evidence text), `guardrail_events`, `disclaimer`, `trace_id`, `latency_ms` and per-stage `spans`. `GET /healthz`, `GET /readyz` and `GET /metrics` serve operations. Set `CLINAGENT_API_KEY` to require a bearer token. The audit log records outcomes and references, never question or answer text.

## MCP

`clinagent mcp` exposes `ask_clinical_question`, `search_guidelines` and `get_patient_context` to any MCP client. Patient context returned over MCP carries the same minimum-necessary guarantee as the API.

## Repository map

```
src/clinagent/
  ingest/        markdown loader (heading-scoped sections), dosage-safe chunker
  retrieval/     TF-IDF (offline, default) and sentence-transformer embedders; persisted index
  fhir/          FHIR R4 backends (HTTP, in-memory) and minimum-necessary patient context
  guardrails/    identifier redaction, identifier-request refusal, prompt-injection screen
  generation/    grounded prompt, extractive + LLM generators, citation and quantity verification
  agent/         ReAct agent over the same tools and checks
  observability/ per-request tracing (optional OTLP export), Prometheus metrics
  evaluation/    eval runner and CI gate
  api/           FastAPI service          mcp_server.py   MCP server       cli.py
corpus/          synthetic guideline corpus
data/fhir/       synthetic FHIR bundle (scripts/make_synthetic_fhir.py)
evals/           cases, thresholds, poisoned fixture
docs/            ADRs, risk register, system card
deploy/          docker-compose: API + HAPI FHIR + optional Ollama
```

## Design documents

- [System card](docs/system-card.md): intended use, data, performance, limitations, mapped to the FDA/Health Canada/MHRA Good Machine Learning Practice principles
- [Risk register](docs/risk-register.md): hazards, controls, residual risk, and what is not controlled
- ADRs: [fixed workflow over agent](docs/adr/0001-fixed-workflow-over-agent.md) · [NumPy index](docs/adr/0002-numpy-index.md) · [deterministic eval gate](docs/adr/0003-deterministic-eval-gate.md)

## Author

Motaz Alqaoud, PhD · [motazalqaoud.com](https://motazalqaoud.com) · [Hugging Face](https://huggingface.co/motazalqaoud)

A personal project, built with public tools and synthetic data. It does not represent my employer.

MIT licensed.

# ADR 0003: Gate CI on a deterministic generator; report LLM results separately

**Status:** accepted · 2026-10-25

## Context

An evaluation that blocks merges must be fast, free and reproducible. LLM output is none of those: it varies across hardware and versions, and hosted calls cost money and leak test data to a vendor.

## Decision

The required CI check runs the full suite with the extractive generator, which composes answers from verbatim evidence sentences. LLM runs use the same cases and the same runner (`--llm ...`) and are reported in the README, not gated.

## What the gate does and does not measure

It measures everything around the model: retrieval recall, section citation, patient-record citation, refusal of out-of-scope and identifier requests, identifier redaction, injection quarantine, leaks and latency. Safety metrics are absolute (zero leaks).

It does not measure the model's reasoning or faithfulness of free prose. Quantity fidelity is enforced at runtime by verify.py for every generator; prose faithfulness needs an LLM judge or clinician review, listed as an open item in the risk register.

## Consequences

- A PR that breaks a guardrail fails in under a minute.
- A model upgrade needs a deliberate evaluation run and a README update; it cannot slip in silently.

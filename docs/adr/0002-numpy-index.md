# ADR 0002: Brute-force NumPy index instead of a vector database

**Status:** accepted · 2026-10-25

## Context

A single clinical service indexes an institution's approved guidelines: thousands of chunks, not millions.

## Decision

Exact inner-product search over a NumPy matrix, persisted as `vectors.npy` plus JSON metadata.

## Reasons

- Exact results, so retrieval changes in the eval report come from code changes, never from approximate-search nondeterminism.
- No native dependency or extra service to patch, scan and operate.
- At 10k chunks × 384 dimensions, a query is one matrix-vector product, well under a millisecond.

## Revisit when

The corpus passes roughly 100k chunks, or several services need to share one index. `GuidelineIndex.search` is the only interface to replace (FAISS, pgvector or a managed store).

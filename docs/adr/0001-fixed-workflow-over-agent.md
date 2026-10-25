# ADR 0001: A fixed workflow for patient questions; the agent is opt-in

**Status:** accepted · 2026-10-25

## Context

Patient-context questions need guideline evidence and facts from the patient's record. Two designs can deliver that: a ReAct agent that decides which tools to call, or a fixed pipeline that always gathers the same evidence in the same order.

## Decision

The API and MCP `ask_clinical_question` use the fixed pipeline. The agent (`clinagent agent`) is available for exploration and is evaluated separately.

## Reasons

- **Auditability.** For a given question, the data touched is fully determined by the code. A reviewer can say exactly which record sections reach the model; with an agent that depends on what the model decided at run time.
- **Testability.** The pipeline is deterministic up to the generator, so the CI gate measures retrieval, refusal and guardrails without model noise.
- **Failure modes.** Agents fail by looping, by skipping a tool (answering a dosing question without fetching weight), or by being steered by injected text into calling tools. The fixed workflow cannot skip the allergy list.
- **Cost and latency.** One model call instead of three to six.

## Consequences

- Questions needing multi-hop reasoning across many record sections are answered less well than a capable agent could. That gap is measurable with the same suite and is the trigger to revisit this decision.
- The agent shares the verification layer, so even when used it cannot return an uncited or altered dose.

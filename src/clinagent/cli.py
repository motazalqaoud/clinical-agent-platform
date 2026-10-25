"""Command line: ingest, ask, agent, eval, serve, mcp."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from clinagent.settings import Settings


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    overrides = {k: v for k in ("llm", "embedder", "fhir") if (v := getattr(args, k, None))}
    return replace(s, **overrides)


def cmd_ingest(args: argparse.Namespace) -> int:
    from clinagent.retrieval.index import GuidelineIndex

    index = GuidelineIndex.from_directory(args.corpus, args.embedder or "tfidf")
    index.save(args.out)
    print(f"indexed {len(index)} chunks from {args.corpus} -> {args.out}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from clinagent.service import build_service

    answer, trace = build_service(_settings(args)).ask(args.question, args.patient)
    if args.json:
        print(
            json.dumps(
                {
                    "answer": answer.text,
                    "grounded": answer.grounded,
                    "citations": [vars(c) for c in answer.citations],
                    "guardrail_events": [vars(e) for e in answer.guardrail_events],
                    "spans": [vars(s) for s in trace.spans],
                },
                indent=2,
            )
        )
        return 0
    print(answer.text)
    for c in answer.citations:
        print(f"  [{c.index}] {c.title} ({c.ref})")
    for e in answer.guardrail_events:
        print(f"  guardrail: {e.guard} {e.action} - {e.detail}")
    print(f"  ({answer.latency_ms:.1f} ms, trace {answer.trace_id})")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from clinagent.agent.react import ClinicalAgent
    from clinagent.fhir.backend import make_backend
    from clinagent.fhir.context import PatientContext
    from clinagent.llm.clients import make_llm
    from clinagent.service import build_index

    s = _settings(args)
    if s.llm == "extractive":
        print("The agent needs a real model: pass --llm openai:<model>@<url> (or anthropic:...)")
        return 2
    agent = ClinicalAgent(make_llm(s.llm), build_index(s), PatientContext(make_backend(s.fhir)))
    result = agent.run(args.question)
    for i, step in enumerate(result.steps, 1):
        print(f"--- step {i}: {step.action or 'final'} {step.action_input or ''}")
    print(result.answer or "No verified answer.")
    for i, src in enumerate(result.evidence, 1):
        print(f"  [{i}] {src.ref}")
    return 0 if result.verified else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from clinagent.evaluation.runner import main as eval_main

    return eval_main(args)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("clinagent.api.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from clinagent.mcp_server import create_server

    create_server().run("streamable-http" if args.http else "stdio")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinagent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build and save a guideline index")
    p.add_argument("--corpus", default=Settings().corpus_dir)
    p.add_argument("--out", default="data/index")
    p.add_argument("--embedder", choices=["tfidf", "dense"])
    p.set_defaults(func=cmd_ingest)

    for name, func, help_ in (
        ("ask", cmd_ask, "answer one question"),
        ("agent", cmd_agent, "run the ReAct agent on one question"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("question")
        p.add_argument("--patient")
        p.add_argument("--llm")
        p.add_argument("--embedder", choices=["tfidf", "dense"])
        p.add_argument("--fhir")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("eval", help="run the evaluation suite and enforce thresholds")
    p.add_argument("--cases", default="evals/cases.yaml")
    p.add_argument("--thresholds", default="evals/thresholds.yaml")
    p.add_argument("--llm", default="extractive")
    p.add_argument("--embedder", default="tfidf", choices=["tfidf", "dense"])
    p.add_argument("--out", default="reports")
    p.add_argument("--no-gate", action="store_true", help="report only; never fail")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="run the MCP server")
    p.add_argument("--http", action="store_true")
    p.set_defaults(func=cmd_mcp)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

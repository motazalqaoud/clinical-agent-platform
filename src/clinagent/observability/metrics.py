"""Prometheus metrics in the text exposition format, without a client dependency.

Exposed at ``GET /metrics``. The series an on-call engineer needs for a clinical QA
service: request volume by outcome, refusal and guardrail rates, and a latency
histogram. A rising refusal rate after a corpus update, or any non-zero
``prompt_injection`` count, is worth an alert.
"""

from __future__ import annotations

import threading
from collections import defaultdict

_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[str, int] = defaultdict(int)
        self.guardrails: dict[tuple[str, str], int] = defaultdict(int)
        self.bucket_counts = [0] * len(_BUCKETS)
        self.latency_sum = 0.0
        self.latency_count = 0

    def observe(self, outcome: str, latency_ms: float, events: list[tuple[str, str]]) -> None:
        with self._lock:
            self.requests[outcome] += 1
            for key in events:
                self.guardrails[key] += 1
            self.latency_sum += latency_ms
            self.latency_count += 1
            for i, bound in enumerate(_BUCKETS):
                if latency_ms <= bound:
                    self.bucket_counts[i] += 1

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP clinagent_requests_total Answered requests by outcome.",
                "# TYPE clinagent_requests_total counter",
            ]
            lines += [
                f'clinagent_requests_total{{outcome="{k}"}} {v}'
                for k, v in sorted(self.requests.items())
            ]
            lines += [
                "# HELP clinagent_guardrail_events_total Guardrail events by guard and action.",
                "# TYPE clinagent_guardrail_events_total counter",
            ]
            lines += [
                f'clinagent_guardrail_events_total{{guard="{g}",action="{a}"}} {v}'
                for (g, a), v in sorted(self.guardrails.items())
            ]
            lines += [
                "# HELP clinagent_request_latency_ms End-to-end request latency.",
                "# TYPE clinagent_request_latency_ms histogram",
            ]
            lines += [
                f'clinagent_request_latency_ms_bucket{{le="{b}"}} {c}'
                for b, c in zip(_BUCKETS, self.bucket_counts, strict=True)
            ]
            lines += [
                f'clinagent_request_latency_ms_bucket{{le="+Inf"}} {self.latency_count}',
                f"clinagent_request_latency_ms_sum {round(self.latency_sum, 3)}",
                f"clinagent_request_latency_ms_count {self.latency_count}",
            ]
            return "\n".join(lines) + "\n"

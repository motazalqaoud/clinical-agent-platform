"""Per-request tracing with an optional OpenTelemetry export.

Every request gets a trace id and one span per pipeline stage (input guard, retrieval,
patient context, injection screen, generation, verification, output guard). Spans are
always recorded in-process and returned with the response, so latency per stage is
visible without any infrastructure. If ``opentelemetry-sdk`` is installed and
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, the same spans are also exported over OTLP to
Jaeger, Tempo, Honeycomb or Datadog.

Span attributes never contain question text, evidence or answers -- only counts,
scores, model names and guardrail outcomes -- so traces are safe to ship to a
third-party backend.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


def _otel_tracer() -> Any:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": "clinagent"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("clinagent")


_OTEL = _otel_tracer()


class Trace:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.spans: list[Span] = []
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
        attrs: dict[str, Any] = dict(attributes)
        start = time.perf_counter()
        otel_cm = _OTEL.start_as_current_span(name) if _OTEL else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        try:
            yield attrs
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.spans.append(Span(name, round(elapsed, 3), attrs))
            if otel_span is not None and otel_cm is not None:
                for key, value in attrs.items():
                    otel_span.set_attribute(f"clinagent.{key}", value)
                otel_cm.__exit__(None, None, None)

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 3)

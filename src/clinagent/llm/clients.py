"""LLM backends behind one two-method interface.

The pipeline and the agent only call ``complete(system, messages) -> str``. That keeps
every model swappable by configuration and every code path testable offline with
``ScriptedLLM``. Supported backends:

* ``openai``      any OpenAI-compatible server: vLLM, Ollama, LM Studio, TGI, hosted APIs
* ``anthropic``   the Anthropic Messages API
* ``transformers`` a Hugging Face model loaded in-process
* ``scripted``    canned responses, for tests
"""

from __future__ import annotations

import os
from typing import Any, Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, str]]) -> str: ...


class ScriptedLLM:
    name = "scripted"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        if not self._responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str | None = None,
        max_tokens: int = 400,
        timeout: float = 120.0,
    ):
        import httpx

        self.name = f"openai:{model}"
        self.model = model
        self.max_tokens = max_tokens
        key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
        )

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 400):
        import anthropic

        self.name = f"anthropic:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class TransformersLLM:
    def __init__(self, model: str, max_new_tokens: int = 300):
        from transformers import pipeline

        self.name = f"transformers:{model}"
        self.max_new_tokens = max_new_tokens
        self._pipe = pipeline("text-generation", model=model)

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        out = self._pipe(
            [{"role": "system", "content": system}, *messages],
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return str(out[0]["generated_text"][-1]["content"])


def make_llm(spec: str) -> LLMClient:
    """Build a client from a spec string.

    ``openai:<model>[@<base_url>]``, ``anthropic:<model>``, ``transformers:<model>``.
    """
    kind, _, rest = spec.partition(":")
    if kind == "openai":
        model, _, base = rest.partition("@")
        return OpenAICompatibleLLM(
            model,
            base_url=base or os.environ.get("CLINAGENT_LLM_BASE_URL", "http://localhost:8001/v1"),
        )
    if kind == "anthropic":
        return AnthropicLLM(rest)
    if kind == "transformers":
        return TransformersLLM(rest)
    raise ValueError(f"Unknown LLM spec '{spec}'")

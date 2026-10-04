"""Provider-agnostic LLM access.

Everything else in the project talks to ``get_llm()`` and never to a vendor SDK
directly. The default provider is ``mock`` (no network, ``available`` is False),
so the whole pipeline runs and is testable without an API key. Callers must
check ``llm.available`` and fall back to deterministic behaviour.
"""
from __future__ import annotations

import base64
import logging

from django.conf import settings

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


class BaseLLM:
    available = False

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        raise LLMUnavailable("No LLM provider configured")

    def vision(self, system: str, user: str, image_bytes: bytes,
               media_type: str = "image/png", max_tokens: int = 1500) -> str:
        raise LLMUnavailable("No LLM provider configured")


class MockLLM(BaseLLM):
    """Used when no provider is configured. Always unavailable."""


class AnthropicLLM(BaseLLM):
    available = True

    def __init__(self, model: str):
        import os

        import anthropic  # imported lazily so the mock path needs no SDK

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model

    def complete(self, system, user, max_tokens=1200):
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def vision(self, system, user, image_bytes, media_type="image/png", max_tokens=1500):
        data = base64.standard_b64encode(image_bytes).decode()
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": media_type, "data": data}},
                    {"type": "text", "text": user},
                ],
            }],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


_cached: BaseLLM | None = None


def get_llm() -> BaseLLM:
    global _cached
    provider = getattr(settings, "MEDILENS_LLM_PROVIDER", "mock")
    if provider == "anthropic":
        if _cached is None or not isinstance(_cached, AnthropicLLM):
            try:
                _cached = AnthropicLLM(settings.MEDILENS_LLM_MODEL)
            except Exception as exc:  # missing SDK / key
                log.warning("Anthropic provider unavailable (%s); using mock", exc)
                return MockLLM()
        return _cached
    return MockLLM()


def reset_llm_cache():
    global _cached
    _cached = None


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (handles ``` fences)."""
    import json
    import re

    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in LLM output")

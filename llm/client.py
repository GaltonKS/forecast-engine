"""Minimal client for an OpenAI-compatible chat endpoint (vLLM on Modal or RunPod).

Only /chat/completions is used. Structured output is requested one of three
ways (LLM_GUIDED), because servers differ:
  json_schema  OpenAI-style response_format (current vLLM, most hosted APIs)
  guided_json  vLLM's old extra parameter; removed in vLLM 0.12, only for old servers
  none         no constraint; the reply is parsed and validated anyway

A GPU endpoint that scales to zero answers 503 (or can't be reached) while a
container starts, which can take minutes. Those responses are retried with
backoff until LLM_TIMEOUT_SECONDS is used up, so the first query after an
idle period waits for the cold start instead of failing.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx


class LLMError(Exception):
    """Base class: message is safe to show to the API caller."""


class LLMNotConfigured(LLMError):
    pass


class LLMUnavailable(LLMError):
    """Network error, timeout, or 5xx: the endpoint, not the prompt, is the problem."""


class LLMBadOutput(LLMError):
    """The model answered, but not with usable JSON."""


@dataclass
class LLMConfig:
    base_url: str | None = None  # e.g. https://you--vllm-serve.modal.run/v1
    model: str | None = None  # e.g. Qwen/Qwen2.5-7B-Instruct
    api_key: str | None = None
    timeout_seconds: float = 180.0  # GPU containers can take a minute+ to cold start
    guided: str = "json_schema"
    max_tokens: int = 1500
    transport: httpx.BaseTransport | None = field(default=None, repr=False)  # tests

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv("LLM_BASE_URL") or None,
            model=os.getenv("LLM_MODEL") or None,
            api_key=os.getenv("LLM_API_KEY") or None,
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
            guided=os.getenv("LLM_GUIDED", "json_schema"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1500")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


RETRY_STATUSES = {429, 502, 503, 504}
MAX_RETRY_WAIT = 15.0


def _sleep(seconds: float) -> None:  # patched in tests
    time.sleep(seconds)


def _now() -> float:  # patched in tests
    return time.monotonic()


def _post_with_retry(client: httpx.Client, url: str, body: dict, headers: dict, cfg: "LLMConfig") -> httpx.Response:
    """POST, retrying cold-start answers (429/502/503/504, can't connect) with
    backoff until cfg.timeout_seconds is used up."""
    deadline = _now() + cfg.timeout_seconds
    wait = 2.0
    while True:
        remaining = max(deadline - _now(), 0.001)
        try:
            r = client.post(url, json=body, headers=headers, timeout=remaining)
        except httpx.TimeoutException:
            raise LLMUnavailable(f"Model endpoint timed out after {cfg.timeout_seconds:g}s")
        except httpx.HTTPError as e:
            problem = f"unreachable ({e.__class__.__name__})"
        else:
            if r.status_code not in RETRY_STATUSES:
                return r
            problem = f"returned {r.status_code}"
        pause = min(wait, deadline - _now())
        if pause <= 0:
            raise LLMUnavailable(
                f"Model endpoint {problem}; still unavailable after {cfg.timeout_seconds:g}s "
                "(a cold GPU can take minutes: raise LLM_TIMEOUT_SECONDS if this persists)"
            )
        _sleep(pause)
        wait = min(wait * 2, MAX_RETRY_WAIT)


@dataclass
class ChatResult:
    content: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None


def strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences when output isn't constrained."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def chat_json(cfg: LLMConfig, messages: list[dict], schema: dict, schema_name: str) -> ChatResult:
    if not cfg.configured:
        raise LLMNotConfigured("LLM not configured: set LLM_BASE_URL and LLM_MODEL")
    body: dict = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.guided == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
    elif cfg.guided == "guided_json":
        body["guided_json"] = schema
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    started = _now()
    with httpx.Client(transport=cfg.transport) as client:
        r = _post_with_retry(client, url, body, headers, cfg)
    latency = _now() - started

    if r.status_code >= 500 or r.status_code == 429:
        raise LLMUnavailable(f"Model endpoint returned {r.status_code}")
    if r.status_code != 200:
        # 4xx: usually auth or a request the server doesn't support (e.g. LLM_GUIDED)
        raise LLMUnavailable(f"Model endpoint rejected the request ({r.status_code}): {r.text[:200]}")
    try:
        data = r.json()
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMBadOutput("Model endpoint returned a response in an unexpected format")
    if choice.get("finish_reason") == "length":
        raise LLMBadOutput("The model's reply was cut off (LLM_MAX_TOKENS too low)")
    usage = data.get("usage") or {}
    return ChatResult(
        content=strip_fences(content),
        latency_seconds=latency,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))

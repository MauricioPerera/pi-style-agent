"""Thin LLM wrapper. Replace the body with whatever provider you use.

The hard layer never calls this. The seam is: `agent.py` calls assemble() and,
if the result is valid, hands the payload to call_llm().

Providers (selected via PI_LLM_PROVIDER):
  - "stub"     : deterministic, no network. For unit tests.
  - "openai"   : OpenAI / Anthropic via the official `openai` python client.
  - "lmstudio" : LM Studio (or any OpenAI-compatible local server) via urllib.
                  No extra deps. Endpoint via PI_LLM_ENDPOINT (default
                  http://localhost:1234/v1/chat/completions). Model via
                  PI_LLM_MODEL (default `google/gemma-4-12b`).
  - "auto"     : same as lmstudio, with a friendlier name.

PI_LLM_TIMEOUT controls the request timeout in seconds (default 300).
"""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_LMSTUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"
DEFAULT_LMSTUDIO_MODEL = "google/gemma-4-12b"


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int


def call_llm(system: str, user: str, model: str = "") -> LLMResponse:
    provider = os.environ.get("PI_LLM_PROVIDER", "stub")

    if provider == "stub":
        body = user.strip()
        text = f"<response>\n{body[:2000]}\n</response>"
        return LLMResponse(text=text, model="stub", tokens_in=len(system) + len(user),
                           tokens_out=len(text))

    if provider in ("openai",):
        from openai import OpenAI          # type: ignore
        client = OpenAI()
        r = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        msg = r.choices[0].message.content or ""
        return LLMResponse(text=msg, model=model, tokens_in=r.usage.prompt_tokens,
                           tokens_out=r.usage.completion_tokens)

    if provider in ("lmstudio", "auto"):
        endpoint = os.environ.get("PI_LLM_ENDPOINT", DEFAULT_LMSTUDIO_ENDPOINT)
        # Model selection: PI_LLM_MODEL env var wins, then the explicit `model`
        # arg, then the default. This ordering matters: the contract may carry
        # a placeholder model name (e.g. "claude-opus-4-8") that the LM Studio
        # server does not know about. Letting the env var override keeps the
        # contract decoupled from the dispatch key.
        env_model = os.environ.get("PI_LLM_MODEL", "").strip()
        chosen = env_model or model.strip() or DEFAULT_LMSTUDIO_MODEL
        if not chosen:
            raise RuntimeError("no model specified: pass it as the `model` arg or set PI_LLM_MODEL")
        timeout = int(os.environ.get("PI_LLM_TIMEOUT", "300"))
        body = json.dumps({
            "model": chosen,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LM Studio {e.code} {e.reason} on {endpoint}: {err_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LM Studio no responde en {endpoint}: {e}") from e
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        used = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            model=chosen,
            tokens_in=used.get("prompt_tokens", 0),
            tokens_out=used.get("completion_tokens", 0),
        )

    raise RuntimeError(f"PI_LLM_PROVIDER no soportado: {provider}")




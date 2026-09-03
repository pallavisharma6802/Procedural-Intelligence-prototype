"""Provider shim.

Default: Vertex AI (Gemini) if Google ADC is configured, else Groq free tier if
GROQ_API_KEY is set, else local Ollama. Also supported: Gemini public API.
Use hosted providers with public/synthetic data only.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Any, Optional

import httpx
from pathlib import Path

def _adc_available() -> bool:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    return (Path.home() / ".config/gcloud/application_default_credentials.json").exists()


def _default_provider() -> str:
    if os.environ.get("PI_PROVIDER"):
        return os.environ["PI_PROVIDER"]
    if _adc_available():
        return "vertex"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "ollama"


PROVIDER = _default_provider()

# Minimum spacing between outbound LLM calls (seconds). One global gate serializes the whole
# pipeline's hosted-API traffic so bursts from concurrent agents don't trip free-tier limits.
# Vertex/pay-as-you-go quotas are far higher than Groq's free tier, so space calls less there.
_MIN_SPACING = float(os.environ.get("PI_MIN_SPACING", "1.5" if PROVIDER == "vertex" else "5.0"))
_throttle_lock = threading.Lock()
_last_call = [0.0]
_RETRIES = int(os.environ.get("PI_RETRIES", "4"))
_MAX_BACKOFF = float(os.environ.get("PI_MAX_BACKOFF", "40"))  # per-minute 429s clear fast; daily quota (~30min) we skip


def _throttle() -> None:
    # Reserve our slot under the lock, then sleep outside it so callers queue in order
    # without holding the lock for the whole wait.
    with _throttle_lock:
        now = time.monotonic()
        slot = max(now, _last_call[0] + _MIN_SPACING)
        _last_call[0] = slot
    delay = slot - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _post_with_retry(url: str, *, tries: int = None, **kw) -> httpx.Response:
    """POST with a global spacing throttle + bounded backoff on 429 / 5xx."""
    tries = tries or _RETRIES
    for attempt in range(tries):
        _throttle()
        try:
            r = httpx.post(url, **kw)
        except httpx.RequestError as exc:
            if attempt == tries - 1:
                raise
            print(f"  [llm] {type(exc).__name__}, retrying")
            time.sleep(min(2 ** attempt, 10))
            continue
        if r.status_code not in (429, 500, 502, 503, 529) or attempt == tries - 1:
            r.raise_for_status()
            return r
        wait = float(r.headers.get("retry-after", 0) or 0) or min(2 ** attempt, 12)
        if wait > _MAX_BACKOFF:  # e.g. daily-quota 429 says "try again in 31m" - don't block
            r.raise_for_status()
        time.sleep(wait + random.uniform(0.2, 1.0))
    return r  # unreachable


_MODEL_DEFAULTS = {
    "ollama": "qwen2.5:3b",
    "groq": "qwen/qwen3.8-27b",
    "gemini": "gemini-2.5-flash",
    "vertex": "gemini-2.5-flash",
}
MODEL = os.environ.get("PI_MODEL") or _MODEL_DEFAULTS.get(PROVIDER, "qwen2.5:3b")
# Optional cheaper/faster model for the critic's fact-check calls.
CRITIC_MODEL = os.environ.get("PI_CRITIC_MODEL") or MODEL
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_read_to = float(os.environ.get("PI_TIMEOUT", "150"))
_TIMEOUT = httpx.Timeout(_read_to, connect=15.0)


def complete(
    system: str, user: str, *, json_out: bool = False, temperature: float = 0.2, model: str = None
) -> str:
    model = model or MODEL
    if PROVIDER == "ollama":
        return _ollama(system, user, json_out, temperature, model)
    if PROVIDER == "groq":
        return _groq(system, user, json_out, temperature, model)
    if PROVIDER == "gemini":
        return _gemini(system, user, json_out, temperature, model)
    if PROVIDER == "vertex":
        return _vertex(system, user, json_out, temperature, model)
    raise ValueError(f"unknown PI_PROVIDER={PROVIDER!r}")


def complete_json(system: str, user: str, *, temperature: float = 0.2, model: str = None) -> Any:
    raw = complete(system, user, json_out=True, temperature=temperature, model=model)
    return _extract_json(raw)


def _ollama(system: str, user: str, json_out: bool, temperature: float, model: str) -> str:
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,  # qwen3 etc.: skip chain-of-thought, we want the JSON
        "options": {"temperature": temperature, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_out:
        body["format"] = "json"
    r = _post_with_retry(f"{OLLAMA_HOST}/api/chat", json=body, timeout=_TIMEOUT)
    return r.json()["message"]["content"]


def _groq(system: str, user: str, json_out: bool, temperature: float, model: str) -> str:
    key = os.environ["GROQ_API_KEY"]
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": int(os.environ.get("PI_MAX_TOKENS", "4000")),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if "gpt-oss" in model:  # otherwise these models spend minutes on chain-of-thought
        body["reasoning_effort"] = os.environ.get("PI_REASONING", "low")
    if json_out:
        body["response_format"] = {"type": "json_object"}
    r = _post_with_retry(
        "https://api.groq.com/openai/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=_TIMEOUT,
    )
    return r.json()["choices"][0]["message"]["content"]


def _gemini_body(system: str, user: str, json_out: bool, temperature: float) -> dict:
    gen: dict[str, Any] = {"temperature": temperature,
                           "maxOutputTokens": int(os.environ.get("PI_MAX_TOKENS", "8000"))}
    if json_out:
        gen["responseMimeType"] = "application/json"
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }


def _gemini_text(resp: dict) -> str:
    parts = resp["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _gemini(system: str, user: str, json_out: bool, temperature: float, model: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = _post_with_retry(url, json=_gemini_body(system, user, json_out, temperature),
                         params={"key": key}, timeout=_TIMEOUT)
    return _gemini_text(r.json())


_vertex_creds = [None]


def _vertex_auth() -> tuple[str, str, str]:
    """(bearer_token, project, location) from Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests

    creds = _vertex_creds[0]
    if creds is None:
        creds, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds._pi_project = os.environ.get("PI_VERTEX_PROJECT") or \
            os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
        _vertex_creds[0] = creds
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())
    loc = os.environ.get("PI_VERTEX_LOCATION", "us-central1")
    return creds.token, creds._pi_project, loc


def _vertex(system: str, user: str, json_out: bool, temperature: float, model: str) -> str:
    token, project, loc = _vertex_auth()
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    url = (f"https://{host}/v1/projects/{project}/locations/{loc}"
           f"/publishers/google/models/{model}:generateContent")
    r = _post_with_retry(url, json=_gemini_body(system, user, json_out, temperature),
                         headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
    return _gemini_text(r.json())


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"no JSON found in model output:\n{raw[:500]}")


def info() -> str:
    return f"{PROVIDER}:{MODEL}"

"""Provider-independent LLM wrapper.

Purpose: every other module talks to `complete_json(system, user)` and never to
a vendor SDK, so the provider (Groq / Gemini) or the model can change in .env
without touching the agent. Includes a MockLLM so the whole agent can be unit
tested without internet or an API key.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional, Protocol

import config


class LLMError(Exception):
    """Any failure to get a usable answer from the model."""


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...
    def complete_text(self, system: str, user: str) -> str: ...


# ------------------------------------------------------------------ helpers
def parse_json_loose(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ```json fences or added words."""
    if text is None:
        raise ValueError("empty reply")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


# ------------------------------------------------------------------ timing wrapper
class TimedLLM:
    """Wraps any LLM client and records, for every call: purpose label, seconds, success, prompt/reply size and
    (for Groq) attempts, rate-limit waits and completion tokens. Used to find out WHERE time goes."""

    def __init__(self, inner: Any, labeller: Optional[Callable[[str], str]] = None):
        self.inner = inner
        self.labeller = labeller or (lambda system: "llm")
        self.records: list[dict[str, Any]] = []

    def _timed(self, method: str, system: str, user: str, *args: Any) -> Any:
        t0 = time.perf_counter()
        error, reply = "", None
        try:
            reply = getattr(self.inner, method)(system, user, *args)
            return reply
        except Exception as exc:
            error = str(exc)[:160]
            raise
        finally:
            rec: dict[str, Any] = {"label": self.labeller(system), "seconds": round(time.perf_counter() - t0, 2),
                                   "ok": not error, "error": error or None, "prompt_chars": len(system) + len(user)}
            meta = getattr(self.inner, "last_meta", None)
            if isinstance(meta, dict):
                rec.update(meta)
            self.records.append(rec)

    def complete_json(self, system: str, user: str, max_retries: int = 1) -> dict:
        return self._timed("complete_json", system, user, max_retries)

    def complete_text(self, system: str, user: str) -> str:
        return self._timed("complete_text", system, user)

    def drain(self) -> list[dict[str, Any]]:
        out, self.records = self.records, []
        return out


# ------------------------------------------------------------------ Groq
TRANSIENT = re.compile(r"timeout|timed out|connection|\b(500|502|503|504)\b|overloaded|over capacity", re.I)

class GroqClient:
    """Groq chat-completions client (free tier). Tolerant of model differences:
    if a model rejects JSON mode or the reasoning option, it retries without them."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        if not self.api_key or "paste_your" in self.api_key:
            raise LLMError("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError("Package 'groq' not installed. Run: pip install -r requirements.txt") from exc
        # max_retries=0: the SDK would otherwise retry silently (429/timeouts/5xx) and hide the waiting time.
        # Our own loop below does the retrying, and counts it in last_meta.
        self._client = Groq(api_key=self.api_key, timeout=config.LLM_TIMEOUT_S, max_retries=0)
        self.last_meta: dict[str, Any] = {}   # timing details of the most recent complete_* call

    def _reset_meta(self) -> None:
        self.last_meta = {"attempts": 0, "rate_limit_hits": 0, "sleep_s": 0.0, "http_s": 0.0,
                          "completion_tokens": 0, "json_retries": 0, "transient_retries": 0}

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # progressively simpler option sets, used if the model rejects an option
        option_sets: list[dict[str, Any]] = []
        full: dict[str, Any] = {}
        if json_mode:
            full["response_format"] = {"type": "json_object"}
        if self.model.startswith("openai/gpt-oss"):
            full["extra_body"] = {"reasoning_effort": "low"}
        option_sets.append(full)
        if full:
            option_sets.append({})  # plain call

        last: Optional[Exception] = None
        for options in option_sets:
            for attempt in range(3):
                t0 = time.perf_counter()
                self.last_meta["attempts"] = self.last_meta.get("attempts", 0) + 1
                try:
                    reply = self._client.chat.completions.create(
                        model=self.model, messages=messages, temperature=0, **options)
                    self.last_meta["http_s"] = round(self.last_meta.get("http_s", 0.0) + time.perf_counter() - t0, 2)
                    usage = getattr(reply, "usage", None)
                    self.last_meta["completion_tokens"] = (self.last_meta.get("completion_tokens", 0)
                                                           + (getattr(usage, "completion_tokens", 0) or 0))
                    return reply.choices[0].message.content or ""
                except Exception as exc:  # SDK raises many exception types
                    last = exc
                    msg = str(exc).lower()
                    self.last_meta["http_s"] = round(self.last_meta.get("http_s", 0.0) + time.perf_counter() - t0, 2)
                    if TRANSIENT.search(msg) and self.last_meta.get("transient_retries", 0) < 1:
                        self.last_meta["transient_retries"] = self.last_meta.get("transient_retries", 0) + 1
                        self.last_meta["sleep_s"] = round(self.last_meta.get("sleep_s", 0.0) + 1, 2)
                        time.sleep(1)       # one quick retry for a network blip or a busy server
                        continue
                    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
                        wait = 2 * (attempt + 1)
                        self.last_meta["rate_limit_hits"] = self.last_meta.get("rate_limit_hits", 0) + 1
                        self.last_meta["sleep_s"] = round(self.last_meta.get("sleep_s", 0.0) + wait, 2)
                        time.sleep(wait)
                        continue
                    if any(k in msg for k in ("response_format", "reasoning", "json_validate",
                                              "not supported", "unsupported")):
                        break  # try the simpler option set
                    raise LLMError(f"Groq call failed: {exc}") from exc
        raise LLMError(f"Groq call failed: {last}")

    def complete_text(self, system: str, user: str) -> str:
        self._reset_meta()
        return self._chat(system, user, json_mode=False)

    def complete_json(self, system: str, user: str, max_retries: int = 1) -> dict:
        self._reset_meta()
        prompt = user
        for attempt in range(max_retries + 1):
            text = self._chat(system, prompt, json_mode=True)
            try:
                return parse_json_loose(text)
            except (ValueError, json.JSONDecodeError):
                self.last_meta["json_retries"] = self.last_meta.get("json_retries", 0) + 1
                prompt = user + "\n\nYour previous reply was not valid JSON. Reply with ONE valid JSON object only."
        raise LLMError("Model did not return valid JSON")


# ------------------------------------------------------------------ Gemini (optional, UNTESTED)
class GeminiClient:
    """Optional fallback provider using Gemini's REST API (no extra package).
    NOTE: written from the public API format and NOT tested by the author; test it
    with scripts/test_extractor.py before relying on it."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is missing in .env")

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        import urllib.error
        import urllib.request

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
               f":generateContent?key={self.api_key}")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Gemini call failed: {type(exc).__name__}") from exc  # never echo the URL (contains key)

    def complete_text(self, system: str, user: str) -> str:
        return self._call(system, user, json_mode=False)

    def complete_json(self, system: str, user: str, max_retries: int = 1) -> dict:
        for _ in range(max_retries + 1):
            try:
                return parse_json_loose(self._call(system, user, json_mode=True))
            except (ValueError, json.JSONDecodeError):
                continue
        raise LLMError("Model did not return valid JSON")


# ------------------------------------------------------------------ mock for tests
class MockLLM:
    """Scripted replies for tests. Each item is a dict (returned as JSON), a str
    (returned as text) or an Exception (raised). Records every call."""

    def __init__(self, replies: list[Any]):
        self.replies = list(replies)
        self.calls: list[dict[str, str]] = []

    def _next(self, system: str, user: str) -> Any:
        self.calls.append({"system": system, "user": user})
        if not self.replies:
            raise LLMError("MockLLM has no more scripted replies")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def complete_json(self, system: str, user: str, max_retries: int = 1) -> dict:
        item = self._next(system, user)
        return item if isinstance(item, dict) else parse_json_loose(item)

    def complete_text(self, system: str, user: str) -> str:
        item = self._next(system, user)
        return item if isinstance(item, str) else json.dumps(item)


def get_llm() -> LLM:
    """Build the client chosen in .env (LLM_PROVIDER=groq or gemini)."""
    if config.LLM_PROVIDER == "gemini":
        return GeminiClient()
    return GroqClient()

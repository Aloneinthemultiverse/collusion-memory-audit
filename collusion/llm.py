"""LLM clients with an on-disk response cache and free-tier-friendly throttling.

Every call is cached by a hash of (provider, model, prompt, temperature, sample_key), so
replays and reruns cost nothing. API keys are read from environment variables only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

def _load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing variables win)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
CACHE_DIR = Path(os.environ.get("COLLUSION_CACHE", "cache"))


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str, model: str, temperature: float = 0.7,
                 min_interval: float | None = None, max_retries: int = 6,
                 reasoning_effort: str | None = None, key_env: str | None = None):
        self.provider = provider
        self.key_env = key_env  # e.g. NVIDIA_API_KEY_2, to spread runs over several keys
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort  # "low"/"medium"/"high" for reasoning models
        default_interval = {"gemini": 4.5, "groq": 2.5, "nvidia": 1.6, "openrouter": 3.5}.get(provider, 0.0)
        self.min_interval = default_interval if min_interval is None else min_interval
        self.max_retries = max_retries
        self._last_call = 0.0
        self.last_reasoning = ""  # hidden reasoning trace of the most recent completion, if the API returns it
        self.calls = 0
        self.cache_hits = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, prompt: str, sample_key: str) -> str:
        parts = [self.provider, self.model, self.temperature, system, prompt, sample_key]
        if self.reasoning_effort:  # keeps old cache keys valid when unset
            parts.append(self.reasoning_effort)
        blob = json.dumps(parts)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system: str, prompt: str, sample_key: str = "") -> str:
        """Return the model's text. sample_key separates otherwise-identical calls (seed, round)."""
        key = self._key(system, prompt, sample_key)
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            self.last_reasoning = cached.get("reasoning", "")
            return cached["text"]
        self.last_reasoning = ""
        text = self._call_with_retry(system, prompt, sample_key)
        path.write_text(json.dumps({"provider": self.provider, "model": self.model, "text": text,
                                    "reasoning": self.last_reasoning}), encoding="utf-8")
        return text

    def _call_with_retry(self, system: str, prompt: str, sample_key: str) -> str:
        delay = 10.0
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                self.calls += 1
                return self._call(system, prompt, sample_key)
            except (LLMError, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"  [llm] {e}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise LLMError("unreachable")

    def _call(self, system: str, prompt: str, sample_key: str) -> str:
        if self.provider == "mock":
            from .mock_agent import mock_reply
            return mock_reply(prompt, sample_key)
        if self.provider == "gemini":
            return self._gemini(system, prompt)
        if self.provider == "groq":
            return self._openai_compatible("https://api.groq.com/openai/v1/chat/completions",
                                           os.environ.get("GROQ_API_KEY"), system, prompt)
        if self.provider == "nvidia":
            return self._openai_compatible("https://integrate.api.nvidia.com/v1/chat/completions",
                                           os.environ.get(self.key_env or "NVIDIA_API_KEY"), system, prompt,
                                           json_mode=False)
        if self.provider == "openrouter":
            return self._openai_compatible("https://openrouter.ai/api/v1/chat/completions",
                                           os.environ.get("OPENROUTER_API_KEY"), system, prompt,
                                           json_mode=False)
        if self.provider == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            return self._openai_compatible(f"{host}/v1/chat/completions", "ollama", system, prompt)
        raise ValueError(f"unknown provider {self.provider}")

    def _gemini(self, system: str, prompt: str) -> str:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("set GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature,
                                 "responseMimeType": "application/json"},
        }
        r = requests.post(url, json=body, headers={"x-goog-api-key": key}, timeout=120)
        if r.status_code in (429, 500, 502, 503, 504):
            raise LLMError(f"gemini HTTP {r.status_code}")
        r.raise_for_status()
        try:
            return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise LLMError(f"gemini bad response: {str(r.json())[:200]}") from e

    def _openai_compatible(self, url: str, key: str | None, system: str, prompt: str,
                           json_mode: bool = True) -> str:
        if not key:
            raise ValueError("missing API key for " + self.provider)
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 4096,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        r = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=300)
        if r.status_code in (429, 500, 502, 503, 504):
            raise LLMError(f"{self.provider} HTTP {r.status_code}")
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # Reasoning models put their thinking in a separate field; only the final answer counts,
        # but the hidden trace is kept for analysis (it is never shown to the other firm).
        self.last_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        text = msg.get("content") or ""
        if not text.strip():
            raise LLMError(f"{self.provider} empty final answer (reasoning ran out of tokens?)")
        return text

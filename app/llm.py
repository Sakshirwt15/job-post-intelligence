"""Thin LLM client (Groq / Gemini / Ollama via REST). Har call ka prompt, latency, tokens, error log hota hai."""
import time
from dataclasses import dataclass

import requests

from app import config


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = config.PRICING.get(model)
    if not price:
        return 0.0
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, db=None,
                 log: bool = True, timeout: int = 90, http_retries: int = 3):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        defaults = {"groq": config.GROQ_MODEL, "gemini": config.GEMINI_MODEL, "ollama": config.OLLAMA_MODEL}
        if self.provider not in defaults:
            raise ValueError(f"Unknown provider {self.provider!r} (use groq | gemini | ollama)")
        self.model = model or defaults[self.provider]
        self.db = db
        self.log = log and db is not None
        self.timeout = timeout
        self.http_retries = http_retries

    # ---------- public ----------
    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "generic",
                 prompt_version: str | None = None, json_mode: bool = False,
                 temperature: float = 0.0) -> LLMResult:
        start = time.perf_counter()
        result, error = None, None
        try:
            text, pt, ct = self._call(prompt, system, json_mode, temperature)
            result = LLMResult(text, pt, ct, (time.perf_counter() - start) * 1000, self.model, self.provider)
            return result
        except Exception as e:  # noqa: BLE001 - log then re-raise
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            if self.log:
                self.db.log_llm_call(
                    provider=self.provider, model=self.model, purpose=purpose, prompt_version=prompt_version,
                    prompt=prompt, response=result.text if result else None,
                    latency_ms=result.latency_ms if result else (time.perf_counter() - start) * 1000,
                    prompt_tokens=result.prompt_tokens if result else 0,
                    completion_tokens=result.completion_tokens if result else 0,
                    success=result is not None, error=error,
                )

    # ---------- providers ----------
    def _post(self, url, headers, payload):
        last = None
        for attempt in range(self.http_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(2 ** attempt * 2)  # free-tier rate limit backoff
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                return r.json()
            except requests.RequestException as e:
                last = LLMError(f"Network error: {e}")
                time.sleep(2 ** attempt)
        raise last or LLMError("LLM request failed")

    def _call(self, prompt, system, json_mode, temperature):
        if self.provider == "groq":
            return self._groq(prompt, system, json_mode, temperature)
        if self.provider == "gemini":
            return self._gemini(prompt, system, json_mode, temperature)
        return self._ollama(prompt, system, json_mode, temperature)

    def _groq(self, prompt, system, json_mode, temperature):
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("https://api.groq.com/openai/v1/chat/completions",
                          {"Authorization": f"Bearer {config.GROQ_API_KEY}"}, payload)
        usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _gemini(self, prompt, system, json_mode, temperature):
        if not config.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set")
        gen_cfg = {"temperature": temperature}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = self._post(url, {"x-goog-api-key": config.GEMINI_API_KEY}, payload)
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Gemini response: {str(data)[:300]}") from e
        usage = data.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

    def _ollama(self, prompt, system, json_mode, temperature):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature}}
        if json_mode:
            payload["format"] = "json"
        data = self._post(f"{config.OLLAMA_URL}/api/chat", {}, payload)
        return data["message"]["content"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_ai_client")

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR = _get_base_dir()
API_KEY_PATH = BASE_DIR / "config" / "api_keys.json"

def _load_settings() -> dict:
    try:
        with open(API_KEY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _load_api_key(required: bool = True) -> str:
    try:
        key = _load_settings().get("openrouter_api_key", "").strip()
        if not key:
            if not required:
                return ""
            raise ValueError("openrouter_api_key is empty in api_keys.json")
        return key
    except Exception as e:
        if not required:
            return ""
        raise RuntimeError(f"Failed to load OpenRouter API key: {e}")

TEXT_MODELS: list[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

VISION_MODELS: list[str] = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3n-e4b-it:free",
]

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
REQUEST_TIMEOUT = 60
SETTINGS = _load_settings()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", SETTINGS.get("ollama_host", "http://127.0.0.1:11434")).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", SETTINGS.get("ollama_model", "qwen3:8b"))
USE_OLLAMA_VALUE = os.getenv("USE_OLLAMA")
USE_OLLAMA = (USE_OLLAMA_VALUE.lower() in ("1", "true", "yes")) if USE_OLLAMA_VALUE is not None else bool(SETTINGS.get("use_ollama", True))
MAX_RETRIES_PER_MODEL = 2
RETRY_DELAY = 2
RATE_LIMIT_COOLDOWN = 60

_rate_limited: dict[str, float] = {}

class OpenRouterClient:

    def __init__(self) -> None:
        self.use_ollama = USE_OLLAMA
        self.ollama_host = OLLAMA_HOST
        self.ollama_model = OLLAMA_MODEL
        self.api_key = _load_api_key(required=not self.use_ollama)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mark-xxxix-or",
            "X-Title": "MARK XXXIX-OR",
        }
        self._session = requests.Session()

    def _is_rate_limited(self, model: str) -> bool:
        ts = _rate_limited.get(model)
        if ts is None:
            return False
        if time.time() - ts > RATE_LIMIT_COOLDOWN:
            del _rate_limited[model]
            return False
        return True

    def _mark_rate_limited(self, model: str) -> None:
        _rate_limited[model] = time.time()
        logger.warning(f"[OpenRouter] Rate limited: {model}")

    def _call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                resp = self._session.post(
                    API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    self._mark_rate_limited(model)
                    return None
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content.strip() if content else None
                logger.warning(f"[OpenRouter] {model} HTTP {resp.status_code}")
            except requests.exceptions.Timeout:
                logger.warning(f"[OpenRouter] {model} timeout")
            except Exception as e:
                logger.error(f"[OpenRouter] {model} error: {e}")
            if attempt < MAX_RETRIES_PER_MODEL:
                time.sleep(RETRY_DELAY)
        return None

    def _call_with_fallback(
        self,
        pool: list[str],
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OpenRouter fallback is disabled because no openrouter_api_key is configured.")

        if model and not self._is_rate_limited(model):
            result = self._call(model, messages, max_tokens, temperature, response_format)
            if result:
                return result

        for item in pool:
            if self._is_rate_limited(item):
                continue
            result = self._call(item, messages, max_tokens, temperature, response_format)
            if result:
                return result

        raise RuntimeError("All remote models failed or are rate-limited.")

    def _ollama_parse(self, data: dict) -> str:
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Ollama response format")
        if "message" in data and isinstance(data["message"], dict):
            return str(data["message"].get("content", "")).strip()
        if "response" in data:
            return str(data["response"]).strip()
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                if "message" in choice and isinstance(choice["message"], dict):
                    return str(choice["message"].get("content", "")).strip()
                if "text" in choice:
                    return str(choice["text"]).strip()
                if "content" in choice:
                    return str(choice["content"]).strip()
        if "completion" in data:
            return str(data["completion"]).strip()
        if "result" in data:
            return str(data["result"]).strip()
        raise RuntimeError(f"Ollama response not understood: {data}")

    def _ollama_request(self, payload: dict, use_messages: bool) -> str:
        payload = dict(payload)
        payload["stream"] = False
        endpoints = ["/api/chat", "/v1/chat/completions"] if use_messages else ["/api/generate", "/v1/completions"]
        last_error = None
        for endpoint in endpoints:
            try:
                resp = self._session.post(f"{self.ollama_host}{endpoint}", json=payload, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 404:
                    last_error = RuntimeError(f"Endpoint not found: {endpoint}")
                    continue
                resp.raise_for_status()
                return self._ollama_parse(resp.json())
            except requests.exceptions.RequestException as exc:
                last_error = exc
        raise RuntimeError(f"Ollama request failed. Last error: {last_error}")

    def _ollama_chat(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        return self._ollama_request(payload, use_messages=True)

    def _ollama_multi_turn(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        return self._ollama_request(payload, use_messages=True)

    def _clean_json(self, raw: str) -> dict:
        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("`").strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end + 1]
        return json.loads(clean)

    def chat(
        self,
        prompt: str,
        system: str = "You are a helpful desktop assistant. Be concise, helpful, and precise.",
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        if self.use_ollama:
            try:
                return self._ollama_chat(prompt, system, max_tokens, temperature)
            except Exception as e:
                logger.warning(f"[Ollama] Local model failed, falling back to OpenRouter: {e}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self._call_with_fallback(TEXT_MODELS, messages, model, max_tokens, temperature)

    def chat_json(
        self,
        prompt: str,
        system: str = "Return ONLY valid JSON. No markdown fences, no extra text, no explanation.",
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        raw = ""
        if self.use_ollama:
            try:
                raw = self._ollama_chat(prompt, system, max_tokens, 0.2)
            except Exception as e:
                logger.warning(f"[Ollama] Local JSON call failed, falling back to OpenRouter: {e}")
        if not raw:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            raw = self._call_with_fallback(TEXT_MODELS, messages, model, max_tokens, temperature=0.2)
        try:
            return self._clean_json(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned unparseable JSON: {e}. Raw output: {raw[:200]}")

    def multi_turn(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        if self.use_ollama:
            try:
                return self._ollama_multi_turn(messages, max_tokens, temperature)
            except Exception as e:
                logger.warning(f"[Ollama] Local multi-turn failed, falling back to OpenRouter: {e}")
        return self._call_with_fallback(TEXT_MODELS, messages, model, max_tokens, temperature)

    def vision(
        self,
        prompt: str,
        image_b64: str,
        mime: str = "image/png",
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        if self.use_ollama:
            try:
                payload = {
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt, "images": [image_b64]},
                    ],
                    "think": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.2},
                }
                return self._ollama_request(payload, use_messages=True)
            except Exception as e:
                logger.warning(f"[Ollama] Local vision failed, falling back to OpenRouter: {e}")

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._call_with_fallback(VISION_MODELS, messages, model, max_tokens, temperature=0.2)

    def vision_from_file(
        self,
        prompt: str,
        image_path: str,
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        path = Path(image_path)
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime = mime_map.get(path.suffix.lower(), "image/png")
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return self.vision(prompt, image_b64, mime, system, model, max_tokens)

    def analyze_image(
        self,
        prompt: str,
        image_b64: str,
        mime: str = "image/png",
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        return self.vision(prompt, image_b64, mime, system, model, max_tokens)

    def available_models(self) -> dict:
        return {
            "text_models": TEXT_MODELS,
            "vision_models": VISION_MODELS,
            "local_model": self.ollama_model if self.use_ollama else None,
            "local_host": self.ollama_host if self.use_ollama else None,
            "openrouter": bool(self.api_key),
            "rate_limited": list(_rate_limited.keys()),
            "total_text": len(TEXT_MODELS),
            "total_vision": len(VISION_MODELS),
        }

client = OpenRouterClient()

if __name__ == "__main__":
    print(client.available_models())

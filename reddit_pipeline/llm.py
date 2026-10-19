"""LLM access behind a small interface.

Backends only move text. Validation and retry live in `structured()`, so every backend gets the same
"validate against schema, retry with feedback, then give up and queue it" behaviour.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, schema: type[BaseModel], temperature: float) -> str:
        """Return the raw model reply (should contain one JSON object)."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors, shape (len(texts), dim)."""


# ---- helpers ----------------------------------------------------------------
def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply (strips <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply (output may have been cut off)")
    return text[start : end + 1]


def schema_hint(schema: type[BaseModel]) -> str:
    """Plain-text key list for backends without constrained decoding."""
    lines = []
    for name, spec in schema.model_json_schema()["properties"].items():
        if "enum" in spec:
            kind = "one of " + json.dumps(spec["enum"])
        elif spec.get("type") == "array":
            kind = "array of strings"
        else:
            kind = spec.get("type", "string")
        lines.append(f'  "{name}": <{kind}>')
    return "{\n" + ",\n".join(lines) + "\n}"


# ---- backends ---------------------------------------------------------------
class OllamaClient:
    def __init__(self, model: str, num_ctx: int):
        import ollama  # lazy: only needed for this backend
        self._ollama, self.model, self.num_ctx = ollama, model, num_ctx

    def complete(self, system, user, schema, temperature):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema.model_json_schema(),          # schema-constrained decoding
            options={"temperature": temperature, "num_ctx": self.num_ctx},
        )
        try:
            resp = self._ollama.chat(think=False, **kwargs)   # skip slow "thinking" tokens
        except Exception:
            resp = self._ollama.chat(**kwargs)                # older client/model without `think`
        return resp.message.content


class OpenAICompatClient:
    """Coachwhip / llama.cpp server / vLLM / LM Studio: anything speaking /v1/chat/completions."""

    def __init__(self, base_url: str, model: str = "", timeout: int = 1800):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.model = model or self._detect_model()

    def _detect_model(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as resp:
                models = [m["id"] for m in json.load(resp)["data"]]
        except Exception as exc:
            raise LLMError(f"Cannot reach the LLM server at {self.base_url}: {exc}") from exc
        if not models:
            raise LLMError(f"Server at {self.base_url} reports no loaded model")
        return models[0]

    def complete(self, system, user, schema, temperature):
        sys_prompt = (system + "\n\nReply with ONLY one JSON object (no markdown, no commentary) "
                      "with exactly these keys:\n" + schema_hint(schema))
        payload = {"model": self.model, "temperature": temperature, "max_tokens": 2048, "stream": False,
                   "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer anything"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]


class OllamaEmbedder:
    def __init__(self, model: str):
        import ollama
        self._ollama, self.model = ollama, model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list = []
        for i in range(0, len(texts), 32):
            vecs.extend(self._ollama.embed(model=self.model, input=texts[i : i + 32]).embeddings)
        arr = np.array(vecs, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


# ---- factories ----------------------------------------------------------------
def make_llm(s: Settings) -> LLMClient:
    if s.backend == "ollama":
        return OllamaClient(s.model or DEFAULT_OLLAMA_MODEL, s.num_ctx)
    if s.backend == "coachwhip":
        return OpenAICompatClient(s.coachwhip_url, s.model, s.llm_timeout)
    raise LLMError(f"unknown backend {s.backend!r}")


def make_embedder(s: Settings) -> Embedder:
    return OllamaEmbedder(s.embed_model)


# ---- validated structured output -----------------------------------------------
@dataclass
class Structured:
    value: BaseModel
    attempts: int          # >1 means the first reply failed validation (tracked as invalid-JSON rate)


def structured(client: LLMClient, system: str, user: str, schema: type[T], attempts: int = 3) -> Structured:
    """Call the model, validate against `schema`, retry with the validation error as feedback."""
    feedback, last_exc = "", None
    for i in range(attempts):
        prompt = user if not feedback else (
            f"{user}\n\n[Your previous reply was rejected: {feedback}. "
            "Reply with ONLY one valid JSON object matching the schema.]")
        try:
            raw = client.complete(system, prompt, schema, temperature=0.0 if i == 0 else 0.2)
            return Structured(schema.model_validate_json(extract_json(raw)), i + 1)
        except (ValidationError, ValueError) as exc:        # bad output: retry with feedback
            feedback, last_exc = str(exc).replace("\n", " ")[:300], exc
        except Exception as exc:                            # transport/server error: back off
            last_exc = exc
            time.sleep(2 * (i + 1))
    raise LLMError(f"LLM call failed after {attempts} attempts: {last_exc}")

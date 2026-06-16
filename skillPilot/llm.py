from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Callable, Protocol

StreamSink = Callable[[str], None]


class ChatLLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        ...


class OpenAICompatibleLLM:
    """Small OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
        chat_path: str | None = None,
        embedding_model: str | None = None,
        embedding_path: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-v4-pro")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.chat_path = (chat_path or os.getenv("OPENAI_CHAT_PATH") or "/chat/completions").strip()
        if not self.chat_path.startswith("/"):
            self.chat_path = f"/{self.chat_path}"
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_path = (embedding_path or os.getenv("OPENAI_EMBEDDING_PATH") or "/embeddings").strip()
        if not self.embedding_path.startswith("/"):
            self.embedding_path = f"/{self.embedding_path}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.last_usage: dict[str, int] = {}
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative token estimate: ~1 token per 4 characters (closer to model reality than char/3)."""
        return max(1, len(text.encode("utf-8")) // 4)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}{self.chat_path}"

    @property
    def embedding_url(self) -> str:
        return f"{self.base_url}{self.embedding_path}"

    def embedding(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts using the OpenAI-compatible /embeddings endpoint."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")
        if not texts:
            return []

        use_model = model or self.embedding_model
        payload = {
            "model": use_model,
            "input": texts,
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                self.embedding_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                items = data.get("data", [])
                if not items:
                    raise RuntimeError("Embedding API returned empty data")
                # Sort by index to preserve input order
                items.sort(key=lambda item: item.get("index", 0))
                return [list(item["embedding"]) for item in items]
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Embedding HTTP {exc.code}: {body[:1000]}") from exc
            except (urllib.error.URLError, socket.timeout) as exc:
                last_error = exc
            except (json.JSONDecodeError, KeyError) as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                delay = self.backoff_base * (2 ** attempt)
                time.sleep(delay)

        if isinstance(last_error, urllib.error.HTTPError):
            body = last_error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding HTTP {last_error.code}: {body[:1000]} (retried {self.max_retries}x)"
            ) from last_error
        raise RuntimeError(
            f"Embedding failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI-compatible chat")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                self.chat_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                # Parse API-reported token usage when available.
                usage = data.get("usage", {})
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    self.last_usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
                    }
                    self.total_input_tokens += prompt_tokens
                    self.total_output_tokens += completion_tokens
                return str(data["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                last_error = exc
                # Retry on rate-limit and server errors only; client errors are final.
                if exc.code not in (429, 500, 502, 503, 504):
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"LLM HTTP {exc.code}: {body[:1000]}") from exc
            except (urllib.error.URLError, socket.timeout) as exc:
                last_error = exc
            except json.JSONDecodeError as exc:
                last_error = exc
                # Malformed response: retry is cheap and often helps.

            if attempt < self.max_retries - 1:
                delay = self.backoff_base * (2 ** attempt)
                time.sleep(delay)

        # All retries exhausted — raise the last error as RuntimeError.
        if isinstance(last_error, urllib.error.HTTPError):
            body = last_error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM HTTP {last_error.code}: {body[:1000]} (retried {self.max_retries}x)"
            ) from last_error
        if isinstance(last_error, (urllib.error.URLError, socket.timeout)):
            raise RuntimeError(
                f"LLM connection failed: {last_error} (retried {self.max_retries}x)"
            ) from last_error
        if isinstance(last_error, json.JSONDecodeError):
            raise RuntimeError(
                f"LLM returned unparseable JSON (retried {self.max_retries}x)"
            ) from last_error
        raise RuntimeError(f"LLM chat failed after {self.max_retries} attempts") from last_error

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream_sink: StreamSink | None = None,
    ) -> str:
        """Streaming chat: parse SSE chunks, call stream_sink per token, return full text."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI-compatible chat")
        sink = stream_sink or (lambda _token: None)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        request = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                buffer = b""
                full_text = ""
                while True:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line_end = buffer.index(b"\n")
                        line = buffer[:line_end].decode("utf-8", errors="replace").strip()
                        buffer = buffer[line_end + 1:]
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                sink(content)
                        except json.JSONDecodeError:
                            continue
                return full_text
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:1000]}") from exc
        except (urllib.error.URLError, socket.timeout) as exc:
            raise RuntimeError(f"LLM connection failed: {exc}") from exc

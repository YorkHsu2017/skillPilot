"""Model providers with native tool-calling support.

The public surface is deliberately small: ``OpenAICompatibleLLM`` owns HTTP,
while ``ModelSession`` owns provider-specific conversation state.  Official
OpenAI endpoints use the Responses API; third-party OpenAI-compatible servers
use Chat Completions.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

StreamSink = Callable[[str], None]


class LLMError(RuntimeError):
    """An actionable provider error safe to show in the CLI."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    response_id: str | None = None
    finish_reason: str = ""


class ChatLLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        ...


class ModelSession:
    """A stateful native-tool-calling conversation."""

    def __init__(
        self,
        client: OpenAICompatibleLLM,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
    ) -> None:
        self.client = client
        self.instructions = instructions
        self.tools = tools
        self.response_id: str | None = None
        self.history: list[dict[str, Any]] = []
        self.response_items: list[dict[str, Any]] = []
        self.last_turn: ModelTurn | None = None

    def reset(self) -> None:
        self.response_id = None
        self.history.clear()
        self.response_items.clear()
        self.last_turn = None

    def send_user(self, text: str) -> ModelTurn:
        if self.client.api_style == "responses":
            return self._responses(text)
        if self.last_turn and not self.last_turn.tool_calls and self.last_turn.text:
            self.history.append({"role": "assistant", "content": self.last_turn.text})
        self.history.append({"role": "user", "content": text})
        return self._chat()

    def send_tool_outputs(
        self,
        turn: ModelTurn,
        outputs: list[dict[str, Any]],
    ) -> ModelTurn:
        if len(outputs) != len(turn.tool_calls):
            raise ValueError("Every tool call must have exactly one output")

        if self.client.api_style == "responses":
            items = [
                {
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": json.dumps(output, ensure_ascii=False),
                }
                for call, output in zip(turn.tool_calls, outputs)
            ]
            return self._responses(items)

        assistant_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    or json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in turn.tool_calls
        ]
        self.history.append(
            {
                "role": "assistant",
                "content": turn.text or None,
                "tool_calls": assistant_calls,
            }
        )
        for call, output in zip(turn.tool_calls, outputs):
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )
        return self._chat()

    def _responses(self, input_value: str | list[dict[str, Any]]) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.client.model,
            "instructions": self.instructions,
            "input": input_value,
            "tools": self.tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_output_tokens": self.client.max_output_tokens,
            "store": self.client.store,
        }
        if self.client.store and self.response_id:
            payload["previous_response_id"] = self.response_id
        elif not self.client.store:
            incoming = (
                [{"role": "user", "content": input_value}]
                if isinstance(input_value, str)
                else input_value
            )
            self.response_items.extend(incoming)
            payload["input"] = self.response_items
        if self.client.reasoning_effort:
            payload["reasoning"] = {"effort": self.client.reasoning_effort}
            if not self.client.store:
                payload["include"] = ["reasoning.encrypted_content"]

        data = self.client._request_json("POST", "/responses", payload)
        turn = self.client._parse_responses_turn(data)
        if not self.client.store:
            self.response_items.extend(
                item for item in data.get("output", []) if isinstance(item, dict)
            )
        if turn.response_id:
            self.response_id = turn.response_id
        self.last_turn = turn
        return turn

    def _chat(self) -> ModelTurn:
        # Some compatible servers reject Responses-only `strict` fields.
        chat_tools = []
        for tool in self.tools:
            fn = {key: value for key, value in tool.items() if key != "type"}
            fn.pop("strict", None)
            chat_tools.append({"type": "function", "function": fn})

        payload: dict[str, Any] = {
            "model": self.client.model,
            "messages": [{"role": "system", "content": self.instructions}] + self.history,
            "tools": chat_tools,
            "tool_choice": "auto",
            "max_tokens": self.client.max_output_tokens,
            "temperature": 0,
        }
        if self.client.reasoning_effort:
            payload["reasoning_effort"] = self.client.reasoning_effort
        data = self.client._request_json("POST", "/chat/completions", payload)
        turn = self.client._parse_chat_turn(data)
        self.last_turn = turn
        return turn


class OpenAICompatibleLLM:
    """Dependency-free client for OpenAI Responses and compatible chat APIs."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        max_output_tokens: int = 8192,
        reasoning_effort: str | None = None,
        store: bool | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.model = (
            model
            or os.getenv("SKILLPILOT_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-5.6-terra"
        )
        explicit_key = api_key or os.getenv("SKILLPILOT_API_KEY")
        self.api_key = explicit_key or os.getenv("OPENAI_API_KEY")
        self._send_key_to_local = bool(explicit_key) or os.getenv(
            "SKILLPILOT_SEND_KEY_TO_LOCAL", "false"
        ).lower() in {"1", "true", "yes"}
        raw_base_url = (
            base_url
            or os.getenv("SKILLPILOT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        for endpoint in ("/chat/completions", "/responses", "/models"):
            if raw_base_url.endswith(endpoint):
                raw_base_url = raw_base_url[: -len(endpoint)].rstrip("/")
                break
        parsed_base = urlparse(raw_base_url)
        if parsed_base.hostname == "api.openai.com" and parsed_base.path in {"", "/"}:
            raw_base_url += "/v1"
        self.base_url = raw_base_url
        requested_style = (
            provider
            or os.getenv("SKILLPILOT_PROVIDER")
            or os.getenv("OPENAI_API_STYLE")
            or "auto"
        ).lower()
        if requested_style == "openai":
            requested_style = "responses"
        if requested_style == "compatible":
            requested_style = "chat"
        if requested_style not in {"auto", "responses", "chat"}:
            raise ValueError("provider must be auto, responses, or chat")
        self.api_style = (
            "responses"
            if requested_style == "auto" and "api.openai.com" in self.base_url
            else "chat"
            if requested_style == "auto"
            else requested_style
        )
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff_base = max(0.0, backoff_base)
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort or os.getenv("SKILLPILOT_REASONING")
        if store is None:
            store = os.getenv("SKILLPILOT_STORE", "true").lower() not in {"0", "false", "no"}
        self.store = store
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.last_usage: dict[str, int] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def requires_api_key(self) -> bool:
        hostname = (urlparse(self.base_url).hostname or "").lower()
        return hostname not in {"localhost", "127.0.0.1", "::1"}

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or not self.requires_api_key

    def new_session(
        self,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
    ) -> ModelSession:
        return ModelSession(self, instructions=instructions, tools=tools)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key and (self.requires_api_key or self._send_key_to_local):
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.configured:
            raise LLMError(
                "No API key configured. Set OPENAI_API_KEY, run with --base-url for a local "
                "server, or use `skill-agent --doctor` for details."
            )
        url = f"{self.base_url}{path}"
        encoded = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                url, data=encoded, headers=self._headers(), method=method
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise LLMError("Provider returned a non-object JSON response")
                self._record_usage(data.get("usage", {}))
                return data
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                detail = self._error_detail(body)
                last_error = LLMError(f"Provider HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = LLMError(f"Cannot reach {self.base_url}: {exc}")
                delay = None
            except json.JSONDecodeError as exc:
                last_error = LLMError(f"Provider returned invalid JSON: {exc}")
                delay = None

            if attempt < self.max_retries - 1:
                time.sleep(delay if delay is not None else self.backoff_base * (2**attempt))

        raise LLMError(
            f"{last_error or 'Provider request failed'} (after {self.max_retries} attempts)"
        ) from last_error

    @staticmethod
    def _error_detail(body: str) -> str:
        try:
            data = json.loads(body)
            error = data.get("error", data)
            if isinstance(error, dict):
                return str(error.get("message") or error.get("detail") or error)[:1000]
            return str(error)[:1000]
        except json.JSONDecodeError:
            return body.strip()[:1000] or "empty error response"

    def _record_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        output_tokens = int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        self.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens) or 0),
        }
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @staticmethod
    def _arguments(value: Any) -> tuple[dict[str, Any], str]:
        raw = value if isinstance(value, str) else json.dumps(value or {})
        try:
            parsed = json.loads(raw)
            return (parsed if isinstance(parsed, dict) else {}, raw)
        except json.JSONDecodeError:
            return ({"_invalid_json": raw}, raw)

    def _parse_responses_turn(self, data: dict[str, Any]) -> ModelTurn:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                args, raw = self._arguments(item.get("arguments", "{}"))
                calls.append(
                    ToolCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=args,
                        raw_arguments=raw,
                    )
                )
            elif item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") in {
                        "output_text",
                        "text",
                    }:
                        text_parts.append(str(part.get("text", "")))
        return ModelTurn(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            response_id=str(data.get("id")) if data.get("id") else None,
            finish_reason=str(data.get("status", "")),
        )

    def _parse_chat_turn(self, data: dict[str, Any]) -> ModelTurn:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Chat provider response is missing choices[0].message") from exc
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        calls: list[ToolCall] = []
        for item in message.get("tool_calls", []) or []:
            function = item.get("function", {}) if isinstance(item, dict) else {}
            args, raw = self._arguments(function.get("arguments", "{}"))
            calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=args,
                    raw_arguments=raw,
                )
            )
        return ModelTurn(
            text=str(content).strip(),
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason", "")),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Plain text completion retained for skill scoring and integrations."""
        if self.api_style == "responses":
            instructions = "\n\n".join(
                item["content"] for item in messages if item.get("role") == "system"
            )
            inputs = [item for item in messages if item.get("role") != "system"]
            payload: dict[str, Any] = {
                "model": self.model,
                "instructions": instructions,
                "input": inputs,
                "max_output_tokens": max_tokens,
                "store": False,
            }
            if self.reasoning_effort:
                payload["reasoning"] = {"effort": self.reasoning_effort}
            return self._parse_responses_turn(
                self._request_json("POST", "/responses", payload)
            ).text

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self._parse_chat_turn(
            self._request_json("POST", "/chat/completions", payload)
        ).text

    def embedding(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        data = self._request_json(
            "POST",
            "/embeddings",
            {"model": model or self.embedding_model, "input": texts},
        )
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(item["embedding"]) for item in items]

    def diagnose(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "configured": self.configured,
            "provider": self.api_style,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": (
                "set"
                if self.api_key and (self.requires_api_key or self._send_key_to_local)
                else "set but not sent to localhost"
                if self.api_key
                else "not set"
            ),
        }
        if not self.configured:
            result["ok"] = False
            result["error"] = "API key is required for this remote endpoint"
            result["hint"] = "Set OPENAI_API_KEY or choose a localhost --base-url"
            return result
        try:
            data = self._request_json("GET", "/models")
            ids = [str(item.get("id")) for item in data.get("data", []) if item.get("id")]
            visible = not ids or self.model in ids
            result["ok"] = visible
            result["model_visible"] = visible
            result["models_returned"] = len(ids)
            if not visible:
                result["error"] = f"Configured model {self.model!r} was not returned by /models"
                result["hint"] = "Choose a model ID returned by this provider"
        except LLMError as exc:
            result["ok"] = False
            result["error"] = str(exc)
            result["hint"] = (
                f"Verify DNS/network access and that {self.base_url}/models is the provider's API route"
            )
        return result

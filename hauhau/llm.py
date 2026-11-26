"""llama.cpp server backend.

Speaks the OpenAI-compatible API exposed by ``llama-server``. Native tool
calling is used directly - this model's chat template implements
``tools``/``tool_call``/``tool_response``, so hand-rolled ReAct parsing is
unnecessary. Two model-specific quirks are handled here:

* thinking: the template emits ``reasoning_content`` deltas that can consume
  the whole token budget and leave ``content`` empty. ``enable_thinking`` is
  passed explicitly on every request.
* no reliable EOS: the loader drops ``</s>`` from the EOG list for this merge,
  so termination is driven by ``finish_reason`` and ``max_tokens``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed(self) -> dict:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(f"tool {self.name!r} got malformed JSON arguments: {self.arguments!r}") from exc
        if not isinstance(value, dict):
            raise LLMError(f"tool {self.name!r} arguments must be an object, got {type(value).__name__}")
        return value


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tok_per_second: float = 0.0
    prompt_per_second: float = 0.0


@dataclass
class Completion:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def text(self) -> str:
        return self.content.strip()


class LlamaClient:
    def __init__(self, base_url: str, model_name: str = "local", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    # --- plumbing ------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None) -> urllib.request.Request:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        return urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")

    def _open(self, req: urllib.request.Request):
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", "replace")[:600]
            except (OSError, ValueError):  # pragma: no cover - best effort
                body = ""
            finally:
                # HTTPError is a file object holding the socket; leaving it open
                # leaks a connection per failed request.
                exc.close()
            raise LLMError(f"llama-server HTTP {exc.code} on {req.full_url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach llama-server at {self.base_url} ({exc.reason}). "
                "Start it with `llama-server -m <model.gguf>` or run with --manage-server."
            ) from exc
        except TimeoutError as exc:
            raise LLMError(f"llama-server timed out after {self.timeout:.0f}s") from exc

    def health(self) -> bool:
        try:
            with self._open(self._request("/health")) as resp:
                body = json.loads(resp.read().decode("utf-8") or "{}")
            return body.get("status") == "ok"
        except (LLMError, json.JSONDecodeError, OSError):
            return False

    def props(self) -> dict:
        try:
            with self._open(self._request("/props")) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except (json.JSONDecodeError, OSError) as exc:
            raise LLMError(f"could not read /props: {exc}") from exc

    # --- chat ----------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        stream: bool = False,
        think: bool = False,
        on_token: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        **params: Any,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": params.pop("temperature", 0.7),
            "top_p": params.pop("top_p", 0.95),
            "max_tokens": params.pop("max_tokens", 1024),
            "chat_template_kwargs": {"enable_thinking": bool(think)},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            # Without this, llama.cpp omits usage from a streamed response, so
            # the caller cannot report token counts.
            payload["stream_options"] = {"include_usage": True}
        payload.update(params)

        req = self._request("/v1/chat/completions", payload)
        if not stream:
            with self._open(req) as resp:
                return self._parse_full(json.loads(resp.read().decode("utf-8")))
        with self._open(req) as resp:
            return self._parse_stream(resp, on_token, on_reasoning)

    def _parse_full(self, data: dict) -> Completion:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments") or "{}",
            )
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        return Completion(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "",
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                tok_per_second=timings.get("predicted_per_second", 0.0),
                prompt_per_second=timings.get("prompt_per_second", 0.0),
            ),
            model=data.get("model", ""),
        )

    def _parse_stream(
        self,
        resp: Iterable[bytes],
        on_token: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
    ) -> Completion:
        out = Completion()
        pending: dict[int, dict] = {}
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "timings" in event:
                out.usage.tok_per_second = (event["timings"] or {}).get("predicted_per_second", 0.0)
                out.usage.prompt_per_second = (event["timings"] or {}).get("prompt_per_second", 0.0)
            if event.get("usage"):
                out.usage.prompt_tokens = event["usage"].get("prompt_tokens", out.usage.prompt_tokens)
                out.usage.completion_tokens = event["usage"].get(
                    "completion_tokens", out.usage.completion_tokens
                )
            choice = (event.get("choices") or [{}])[0]
            if choice.get("finish_reason"):
                out.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                out.content += delta["content"]
                if on_token:
                    on_token(delta["content"])
            if delta.get("reasoning_content"):
                out.reasoning += delta["reasoning_content"]
                if on_reasoning:
                    on_reasoning(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                slot = pending.setdefault(tc.get("index", len(pending)), {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
        out.tool_calls = [
            ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"], arguments=slot["args"])
            for i, slot in sorted(pending.items())
        ]
        return out

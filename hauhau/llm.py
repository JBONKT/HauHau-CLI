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

import http.client
import json
import math
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
        except TypeError as exc:  # arguments arrived as an object, not a string
            raise LLMError(f"tool {self.name!r} got non-string arguments: {self.arguments!r}") from exc
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
        # Compact separators and no ASCII escaping: every separator is a token the
        # server has to prefill on each step of every turn (the whole transcript is
        # re-sent), and Swedish text doubles in size when escaped.
        data = (
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
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
        except (http.client.HTTPException, ValueError) as exc:
            # A malformed server_url ("http://127.0.0.1:notaport") raises
            # InvalidURL, which used to escape as a traceback out of main().
            raise LLMError(
                f"cannot use the server URL {self.base_url!r}: {exc}. "
                "Check server_url in config.toml."
            ) from exc

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
                raw = resp.read().decode("utf-8")
            try:
                return self._parse_full(json.loads(raw))
            except json.JSONDecodeError as exc:
                # A proxy error page or a non-llama server answers 200 with HTML;
                # surface it as LLMError so the CLI shows a panel, not a traceback.
                raise LLMError(f"backend returned non-JSON ({len(raw)} bytes): {raw[:200]!r}") from exc
        with self._open(req) as resp:
            return self._parse_stream(resp, on_token, on_reasoning)

    @staticmethod
    def _shape(data: Any) -> tuple[dict, dict]:
        """First choice and its message/delta, validated.

        A proxy error page or a different server version can answer with JSON
        of another shape; indexing it blindly raised KeyError/AttributeError out
        of chat() and killed the turn with a traceback instead of an LLMError
        the CLI can show.
        """
        if not isinstance(data, dict):
            raise LLMError(f"unexpected response from llama-server: {type(data).__name__}, not an object")
        choices = data.get("choices") or []
        if not isinstance(choices, list):
            raise LLMError("unexpected response from llama-server: 'choices' is not a list")
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict):
            raise LLMError("unexpected response from llama-server: choices[0] is not an object")
        body = choice.get("message") or choice.get("delta") or {}
        if not isinstance(body, dict):
            raise LLMError("unexpected response from llama-server: message is not an object")
        return choice, body

    @staticmethod
    def _rate(source: Any, key: str) -> float:
        """A finite float, or 0.0. Timings arrive as None when a server omits them."""
        if not isinstance(source, dict):
            return 0.0
        value = source.get(key, 0.0)
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return 0.0
        return rate if math.isfinite(rate) else 0.0

    @staticmethod
    def _args(function: Any) -> str:
        """Tool arguments as JSON text, as the API promises but does not always send."""
        if not isinstance(function, dict):
            return "{}"
        value = function.get("arguments")
        if isinstance(value, str):
            return value or "{}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return "{}"

    @staticmethod
    def _calls(raw: Any) -> list[ToolCall]:
        if not isinstance(raw, list):
            return []
        return [
            ToolCall(
                id=(tc.get("id") or f"call_{i}") if isinstance(tc, dict) else f"call_{i}",
                name=((tc.get("function") or {}).get("name") or "") if isinstance(tc, dict) else "",
                arguments=LlamaClient._args((tc.get("function") or {}) if isinstance(tc, dict) else {}),
            )
            for i, tc in enumerate(raw)
        ]

    @staticmethod
    def _tokens(source: Any, key: str, fallback: int = 0) -> int:
        """A non-negative int, or ``fallback``.

        Timings get this guard through :meth:`_rate`; usage needs the same one.
        A server that omits the fields sends ``null``, and ``None`` stored in
        ``Usage`` made the agent's running total raise TypeError mid-turn.
        """
        if not isinstance(source, dict):
            return fallback
        try:
            number = int(source.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return max(0, number)

    def _stream_event(
        self,
        out: Completion,
        event: dict,
        pending: dict[int, dict],
        on_token: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
    ) -> None:
        """Fold one SSE event into ``out``; ignore anything of another shape."""
        if "timings" in event:
            out.usage.tok_per_second = self._rate(event["timings"], "predicted_per_second")
            out.usage.prompt_per_second = self._rate(event["timings"], "prompt_per_second")
        usage = event.get("usage")
        if isinstance(usage, dict) and usage:
            out.usage.prompt_tokens = self._tokens(usage, "prompt_tokens", out.usage.prompt_tokens)
            out.usage.completion_tokens = self._tokens(
                usage, "completion_tokens", out.usage.completion_tokens
            )
        choices = event.get("choices") or []
        if not isinstance(choices, list):
            return
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict):
            return
        if choice.get("finish_reason"):
            out.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            return
        if delta.get("content"):
            out.content += delta["content"]
            if on_token:
                on_token(delta["content"])
        if delta.get("reasoning_content"):
            out.reasoning += delta["reasoning_content"]
            if on_reasoning:
                on_reasoning(delta["reasoning_content"])
        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                self._stream_call(pending, call)

    @staticmethod
    def _stream_call(pending: dict[int, dict], call: dict) -> None:
        """Tool calls arrive in pieces: id and name once, arguments by fragment."""
        slot = pending.setdefault(
            call.get("index", len(pending)), {"id": "", "name": "", "args": ""}
        )
        if call.get("id"):
            slot["id"] = call["id"]
        function = call.get("function") or {}
        if function.get("name"):
            slot["name"] += function["name"]
        if function.get("arguments"):
            slot["args"] += function["arguments"]

    def _parse_full(self, data: dict) -> Completion:
        choice, msg = self._shape(data)
        calls = self._calls(msg.get("tool_calls"))
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        if not isinstance(usage, dict):
            usage = {}
        if not isinstance(timings, dict):
            timings = {}
        return Completion(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "",
            usage=Usage(
                prompt_tokens=self._tokens(usage, "prompt_tokens"),
                completion_tokens=self._tokens(usage, "completion_tokens"),
                tok_per_second=self._rate(timings, "predicted_per_second"),
                prompt_per_second=self._rate(timings, "prompt_per_second"),
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
        seen_event = False
        preview = ""
        done = False
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            if not line.startswith("data:"):
                # A proxy error page or a JSON error body answers 200 with
                # something that is not a stream. Returning an empty Completion
                # there ended the turn with no answer and no error panel.
                preview = preview or line
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                done = True
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                seen_event = True
                self._stream_event(out, event, pending, on_token, on_reasoning)
        # Only markup counts as "not a stream": an SSE stream may still carry
        # comments and malformed lines, and those are skipped silently.
        if not seen_event and not done and preview[:1] in ("<", "{"):
            raise LLMError(f"backend answered 200 but sent no stream: {preview[:200]!r}")
        out.tool_calls = [
            ToolCall(id=slot["id"] or f"call_{i}", name=slot["name"], arguments=slot["args"])
            for i, slot in sorted(pending.items())
        ]
        return out

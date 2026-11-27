from __future__ import annotations

import json
import unittest
from unittest import mock

from hauhau.llm import LlamaClient, LLMError, ToolCall


class FakeResponse:
    """Stands in for urlopen(): iterable of raw SSE lines, usable as a context."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return b"".join(self.lines)


def sse(*events: dict) -> list[bytes]:
    return [f"data: {json.dumps(e)}\n".encode() for e in events] + [b"data: [DONE]\n"]


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class StreamParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LlamaClient("http://127.0.0.1:9")

    def parse(self, lines: list[bytes], on_token=None, on_reasoning=None):
        return self.client._parse_stream(FakeResponse(lines), on_token, on_reasoning)

    def test_content_accumulates(self) -> None:
        result = self.parse(sse(chunk({"content": "Hel"}), chunk({"content": "lo"}, "stop")))
        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.finish_reason, "stop")

    def test_tokens_are_forwarded(self) -> None:
        seen: list[str] = []
        self.parse(sse(chunk({"content": "a"}), chunk({"content": "b"})), on_token=seen.append)
        self.assertEqual(seen, ["a", "b"])

    def test_reasoning_is_separate_from_content(self) -> None:
        seen: list[str] = []
        result = self.parse(
            sse(chunk({"reasoning_content": "hmm"}), chunk({"content": "answer"})),
            on_reasoning=seen.append,
        )
        self.assertEqual(result.reasoning, "hmm")
        self.assertEqual(result.content, "answer")
        self.assertEqual(seen, ["hmm"])

    def test_tool_call_fragments_are_assembled(self) -> None:
        result = self.parse(
            sse(
                chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "shell"}}]}),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd"'}}]}),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": ':"ls"}'}}]}, "tool_calls"),
            )
        )
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "shell")
        self.assertEqual(result.tool_calls[0].parsed(), {"cmd": "ls"})

    def test_parallel_tool_calls_keep_their_index(self) -> None:
        result = self.parse(
            sse(
                chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "shell", "arguments": "{}"}}]}),
                chunk({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "grep", "arguments": "{}"}}]}),
            )
        )
        self.assertEqual([c.name for c in result.tool_calls], ["shell", "grep"])
        self.assertEqual([c.id for c in result.tool_calls], ["a", "b"])

    def test_usage_from_the_final_chunk(self) -> None:
        result = self.parse(
            sse(
                chunk({"content": "hi"}, "stop"),
                {"choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 8},
                 "timings": {"predicted_per_second": 26.7}},
            )
        )
        self.assertEqual(result.usage.completion_tokens, 8)
        self.assertEqual(result.usage.prompt_tokens, 17)
        self.assertAlmostEqual(result.usage.tok_per_second, 26.7)

    def test_malformed_and_unrelated_lines_are_skipped(self) -> None:
        result = self.parse([b": keep-alive\n", b"data: {broken\n", b"not-sse\n", b"data: [DONE]\n"])
        self.assertEqual(result.content, "")

    def test_done_terminates_reading(self) -> None:
        lines = [*sse(chunk({"content": "x"})), b'data: {"choices":[{"delta":{"content":"never"}}]}\n']
        self.assertEqual(self.parse(lines).content, "x")


class RequestPayload(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LlamaClient("http://127.0.0.1:9")
        self.payloads: list[dict] = []

    def run_chat(self, lines: list[bytes], **kwargs):
        def fake_request(path, payload=None):
            self.payloads.append(payload or {})
            return object()

        with mock.patch.object(self.client, "_request", fake_request), \
                mock.patch.object(self.client, "_open", lambda req: FakeResponse(lines)):
            return self.client.chat([{"role": "user", "content": "hi"}], **kwargs)

    def test_streaming_requests_usage(self) -> None:
        self.run_chat(sse(chunk({"content": "ok"}, "stop")), stream=True)
        self.assertEqual(self.payloads[0]["stream_options"], {"include_usage": True})

    def test_non_streaming_does_not(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        with mock.patch.object(self.client, "_request",
                               lambda path, payload=None: self.payloads.append(payload) or object()), \
                mock.patch.object(self.client, "_open", lambda req: FakeResponse([json.dumps(body).encode()])):
            self.client.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("stream_options", self.payloads[0])

    def test_thinking_flag_is_explicit(self) -> None:
        self.run_chat(sse(chunk({"content": "ok"}, "stop")), stream=True, think=False)
        self.assertEqual(self.payloads[0]["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_can_be_enabled(self) -> None:
        self.run_chat(sse(chunk({"content": "ok"}, "stop")), stream=True, think=True)
        self.assertEqual(self.payloads[0]["chat_template_kwargs"], {"enable_thinking": True})

    def test_tools_and_choice_are_sent(self) -> None:
        tools = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]
        self.run_chat(sse(chunk({"content": "ok"}, "stop")), stream=True, tools=tools)
        self.assertEqual(self.payloads[0]["tools"], tools)
        self.assertEqual(self.payloads[0]["tool_choice"], "auto")

    def test_max_tokens_is_forwarded(self) -> None:
        self.run_chat(sse(chunk({"content": "ok"}, "stop")), stream=True, max_tokens=321)
        self.assertEqual(self.payloads[0]["max_tokens"], 321)


class NonStreaming(unittest.TestCase):
    def test_full_response_is_parsed(self) -> None:
        body = {
            "model": "m.gguf",
            "choices": [{
                "message": {
                    "content": "done",
                    "tool_calls": [{"id": "c1", "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            "timings": {"predicted_per_second": 25.0, "prompt_per_second": 90.0},
        }
        client = LlamaClient("http://127.0.0.1:9")
        with mock.patch.object(client, "_request", lambda path, payload=None: object()), \
                mock.patch.object(client, "_open", lambda req: FakeResponse([json.dumps(body).encode()])):
            result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result.content, "done")
        self.assertEqual(result.tool_calls[0].name, "shell")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.usage.completion_tokens, 4)
        self.assertAlmostEqual(result.usage.prompt_per_second, 90.0)


class ToolCallArguments(unittest.TestCase):
    def test_parsed_arguments(self) -> None:
        self.assertEqual(ToolCall("1", "shell", '{"cmd":"ls"}').parsed(), {"cmd": "ls"})

    def test_missing_arguments_are_empty(self) -> None:
        self.assertEqual(ToolCall("1", "shell", "").parsed(), {})

    def test_malformed_arguments_raise(self) -> None:
        with self.assertRaises(LLMError):
            ToolCall("1", "shell", "{nope").parsed()

    def test_non_object_arguments_raise(self) -> None:
        with self.assertRaises(LLMError):
            ToolCall("1", "shell", "[1,2]").parsed()

class TestHostileStream(unittest.TestCase):
    """A 200 that is not a stream must fail, not end the turn with no answer."""

    def setUp(self) -> None:
        self.client = LlamaClient("http://127.0.0.1:9")

    def test_an_html_error_page_raises(self) -> None:
        with self.assertRaises(LLMError) as caught:
            self.client._parse_stream(FakeResponse([b"<html>proxy error</html>\n"]), None, None)
        self.assertIn("no stream", str(caught.exception))

    def test_a_json_error_body_raises(self) -> None:
        with self.assertRaises(LLMError):
            self.client._parse_stream(FakeResponse([b'{"error":"nope"}\n']), None, None)

    def test_an_empty_stream_is_not_an_error(self) -> None:
        self.assertEqual(self.client._parse_stream(FakeResponse([b"\n"]), None, None).content, "")

    def test_a_null_usage_does_not_store_none(self) -> None:
        result = self.client._parse_stream(
            FakeResponse(sse({"choices": [], "usage": {"prompt_tokens": None, "completion_tokens": None}})),
            None,
            None,
        )
        self.assertEqual((result.usage.prompt_tokens, result.usage.completion_tokens), (0, 0))
        self.assertIsInstance(result.usage.prompt_tokens, int)


class TestBadServerUrl(unittest.TestCase):
    def test_a_non_numeric_port_raises_llmerror_not_invalidurl(self) -> None:
        client = LlamaClient("http://127.0.0.1:notaport")
        with self.assertRaises(LLMError) as caught:
            client.chat([{"role": "user", "content": "hej"}])
        self.assertIn("server_url", str(caught.exception))
        self.assertFalse(client.health(), "a broken URL is not a healthy backend")



if __name__ == "__main__":
    unittest.main()


class TestHostilePayloads(unittest.TestCase):
    """A proxy error page or another server version must not crash a turn."""

    def setUp(self) -> None:
        self.client = LlamaClient("http://127.0.0.1:9")

    def test_choices_that_are_not_a_list_raise_llm_error(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            self.client._parse_full({"choices": {"message": {"content": "x"}}})
        self.assertIn("choices", str(ctx.exception).lower())

    def test_choice_that_is_not_an_object_raises_llm_error(self) -> None:
        with self.assertRaises(LLMError):
            self.client._parse_full({"choices": ["nope"]})

    def test_message_that_is_not_an_object_raises_llm_error(self) -> None:
        with self.assertRaises(LLMError):
            self.client._parse_full({"choices": [{"message": "nope"}]})

    def test_missing_pieces_are_simply_empty(self) -> None:
        self.assertEqual(self.client._parse_full({}).content, "")
        self.assertEqual(self.client._parse_full({"choices": []}).tool_calls, [])

    def test_none_and_nan_timings_become_zero(self) -> None:
        for value in (None, float("nan"), "fast", float("inf")):
            completion = self.client._parse_full(
                {"choices": [{"message": {"content": "x"}}], "timings": {"predicted_per_second": value}}
            )
            self.assertEqual(completion.usage.tok_per_second, 0.0)

    def test_usage_that_is_not_an_object_is_ignored(self) -> None:
        completion = self.client._parse_full({"choices": [{"message": {"content": "x"}}], "usage": []})
        self.assertEqual(completion.usage.completion_tokens, 0)

    def test_object_arguments_become_json_text(self) -> None:
        payload = {
            "choices": [
                {"message": {"tool_calls": [{"function": {"name": "shell", "arguments": {"cmd": "ls"}}}]}}
            ]
        }
        call = self.client._parse_full(payload).tool_calls[0]
        self.assertEqual(call.arguments, '{"cmd": "ls"}')
        self.assertEqual(call.parsed(), {"cmd": "ls"})

    def test_non_string_arguments_raise_llm_error_from_parsed(self) -> None:
        with self.assertRaises(LLMError):
            ToolCall(id="1", name="shell", arguments={"cmd": "ls"}).parsed()

    def test_stream_ignores_junk_events_but_keeps_the_text(self) -> None:
        chunks = [
            b'data: {"choices": {"not": "a list"}}\n',
            b"data: not json\n",
            b'data: {"choices": ["nope"]}\n',
            b'data: {"choices": [{"delta": {"content": "hi"}}]}\n',
            b"data: [DONE]\n",
        ]
        self.assertEqual(self.client._parse_stream(chunks, None, None).content, "hi")

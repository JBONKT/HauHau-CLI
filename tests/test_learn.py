from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hauhau.brain import Brain
from hauhau.brain.learn import Draft, Learner, build_transcript, durable, parse_drafts, worth_distilling
from hauhau.config import Config
from hauhau.llm import Completion, LLMError


class FakeClient:
    def __init__(self, replies: list[str] | None = None, error: Exception | None = None) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, **kwargs) -> Completion:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if self.error:
            raise self.error
        return Completion(content=self.replies.pop(0) if self.replies else "[]", finish_reason="stop")


class TestParseDrafts(unittest.TestCase):
    def test_plain_array(self) -> None:
        raw = '[{"kind":"preference","text":"The user prefers uv over pip.","tags":["python"],"importance":0.8}]'
        drafts = parse_drafts(raw)
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].kind, "preference")
        self.assertEqual(drafts[0].tags, "python")
        self.assertAlmostEqual(drafts[0].importance, 0.8)

    def test_fenced_and_chatty_output(self) -> None:
        raw = (
            "Sure!\n```json\n"
            '[{"kind":"semantic","text":"The repo lives in ~/Documents/HauHau-CLI."}]\n'
            "```\nHope that helps."
        )
        drafts = parse_drafts(raw)
        self.assertEqual(len(drafts), 1)
        self.assertIn("HauHau-CLI", drafts[0].text)

    def test_empty_array(self) -> None:
        self.assertEqual(parse_drafts("[]"), [])

    def test_garbage_returns_empty(self) -> None:
        self.assertEqual(parse_drafts("no json here"), [])
        self.assertEqual(parse_drafts(""), [])

    def test_short_texts_are_dropped(self) -> None:
        self.assertEqual(parse_drafts('[{"kind":"semantic","text":"too short"}]'), [])

    def test_bad_importance_defaults(self) -> None:
        drafts = parse_drafts('[{"kind":"semantic","text":"A durable fact about the machine.","importance":"NaN"}]')
        self.assertEqual(len(drafts), 1)
        self.assertAlmostEqual(drafts[0].importance, 0.5)

    def test_capped_at_eight(self) -> None:
        items = ",".join(f'{{"kind":"semantic","text":"fact number {i} about things"}}' for i in range(20))
        self.assertEqual(len(parse_drafts(f"[{items}]")), 8)

    def test_tags_list_is_flattened(self) -> None:
        drafts = parse_drafts('[{"kind":"semantic","text":"A durable fact about the machine.","tags":["a","b"]}]')
        self.assertEqual(drafts[0].tags, "a,b")


class TestTranscript(unittest.TestCase):
    def test_includes_each_role(self) -> None:
        messages = [
            {"role": "user", "content": "check disk"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "shell"}}]},
            {"role": "tool", "name": "shell", "content": "df output"},
            {"role": "assistant", "content": "You have 109 GB free."},
        ]
        text = build_transcript(messages)
        self.assertIn("[user] check disk", text)
        self.assertIn("[assistant -> shell]", text)
        self.assertIn("[tool shell] df output", text)
        self.assertIn("109 GB free", text)

    def test_tail_truncation(self) -> None:
        messages = [{"role": "user", "content": "x" * 500} for _ in range(50)]
        self.assertLessEqual(len(build_transcript(messages, max_chars=1000)), 1000)


class TestLearner(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(root / "data")
        self.brain = Brain(self.cfg)
        self.messages = [
            {"role": "user", "content": "when you run the test suite here, always use pytest with the -x flag"},
            {"role": "assistant", "content": "Understood - checking the existing test command first."},
            {"role": "tool", "name": "shell", "content": "pytest -x -q\n42 passed"},
            {"role": "assistant", "content": "Noted: pytest -x is what works here."},
        ]

    def tearDown(self) -> None:
        self.brain.close()
        self._tmp.cleanup()

    def test_distilled_memories_are_stored(self) -> None:
        payload = (
            '[{"kind":"preference","text":"The user wants pytest run with the -x flag here.",'
            '"tags":"testing","importance":0.8}]'
        )
        client = FakeClient([payload])
        learner = Learner(self.cfg, client, self.brain)
        drafts, written = learner.distill(self.messages)
        self.assertEqual(len(drafts), 1)
        self.assertEqual(len(written), 1)
        self.assertTrue(written[0][1], "first write should be new")
        hits = self.brain.recall("pytest flags", k=3)
        self.assertTrue(hits)
        self.assertIn("pytest", hits[0].text)

    def test_repeated_distillation_merges(self) -> None:
        payload = '[{"kind":"preference","text":"The user wants pytest run with the -x flag here."}]'
        learner = Learner(self.cfg, FakeClient([payload, payload]), self.brain)
        learner.distill(self.messages)
        _, written = learner.distill(self.messages)
        self.assertFalse(written[0][1], "second identical distillation must merge")
        self.assertEqual(self.brain.stats()["total"], 1)

    def test_llm_error_is_swallowed(self) -> None:
        learner = Learner(self.cfg, FakeClient(error=LLMError("down")), self.brain)
        self.assertEqual(learner.distill(self.messages), ([], []))

    def test_trivial_greeting_costs_no_model_pass(self) -> None:
        """A hello is not a session: no 20s exit delay to learn nothing."""
        client = FakeClient()
        learner = Learner(self.cfg, client, self.brain)
        drafts, written = learner.distill(
            [
                {"role": "user", "content": "hej"},
                {"role": "assistant", "content": "Hej! Vad kan jag hjälpa dig med?"},
            ]
        )
        self.assertEqual((drafts, written), ([], []))
        self.assertEqual(client.calls, [], "no model pass for a greeting")

    def test_worth_distilling_rules(self) -> None:
        greeting = [{"role": "user", "content": "hej"}, {"role": "assistant", "content": "Hej!"}]
        self.assertFalse(worth_distilling(greeting))
        self.assertFalse(worth_distilling([]))
        one_long = [
            {"role": "user", "content": "why is the build failing " + "x" * 400},
            {"role": "assistant", "content": "because of the cache"},
        ]
        self.assertTrue(worth_distilling(one_long))
        two_turns = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "thanks"},
        ]
        self.assertTrue(worth_distilling(two_turns))
        with_tool = [*greeting, {"role": "tool", "name": "shell", "content": "ok"}]
        self.assertTrue(worth_distilling(with_tool))

    def test_no_turns_yields_nothing(self) -> None:
        learner = Learner(self.cfg, FakeClient(), self.brain)
        self.assertEqual(learner.distill([{"role": "assistant", "content": "hi"}]), ([], []))

    def test_generic_draft_is_rejected(self) -> None:
        """The 2B model calls "echo MEM_OK worked" a procedure; it is noise."""
        payload = (
            '[{"kind":"procedural","text":"The user executes the command echo MEM_OK to confirm success."}]'
        )
        client = FakeClient([payload])
        drafts, written = Learner(self.cfg, client, self.brain).distill(self.messages)
        self.assertEqual(drafts, [])
        self.assertEqual(written, [])
        self.assertEqual(self.brain.stats()["total"], 0)

    def test_concrete_draft_survives_the_guard(self) -> None:
        payload = (
            '[{"kind":"procedural","text":"Ruff is configured with line length 120 in pyproject.toml."},'
            ' {"kind":"preference","text":"The user wants short answers."}]'
        )
        drafts, written = Learner(self.cfg, FakeClient([payload]), self.brain).distill(self.messages)
        self.assertEqual(len(drafts), 2, "path-bearing procedure and a plain preference both stay")
        self.assertEqual(len(written), 2)

    def test_prompt_lists_already_known_memories(self) -> None:
        self.brain.remember("semantic", "The model lives in /Users/johan/modeller/gguf.")
        client = FakeClient(["[]"])
        Learner(self.cfg, client, self.brain).distill(self.messages)
        prompt = client.calls[0]["messages"][0]["content"]
        self.assertIn("ALREADY REMEMBERED", prompt)
        self.assertIn("/Users/johan/modeller/gguf", prompt)
        self.assertIn("do not restate", prompt)

    def test_prompt_marks_empty_memory(self) -> None:
        client = FakeClient(["[]"])
        Learner(self.cfg, client, self.brain).distill(self.messages)
        self.assertIn("(nothing yet)", client.calls[0]["messages"][0]["content"])

    def test_disabled_brain_is_a_noop(self) -> None:
        self.cfg.brain.enabled = False
        learner = Learner(self.cfg, FakeClient(), None)
        self.assertEqual(learner.distill(self.messages), ([], []))


class TestDurableKindValidation(unittest.TestCase):
    """The specificity guard must not be bypassable by a kind label."""

    def test_unknown_kind_fails_closed(self) -> None:
        noise = "The user executes echo MEM_OK to confirm success."
        for kind in ("procedural", "procedure", "Semantic", "SEMANTIC"):
            self.assertFalse(durable(Draft(kind, noise), ("hauhau", "johan")), kind)

    def test_real_machine_facts_are_kept(self) -> None:
        for text in (
            "The CI provider for this repository is GitHub Actions.",
            "The user's machine is an M1 MacBook Air with eight gigabytes of memory.",
            "The user's editor of choice is Emacs with the evil keybindings.",
            "The staging deploy is triggered by the release branch.",
        ):
            self.assertTrue(durable(Draft("semantic", text), ("hauhau", "johan")), text)

    def test_a_plain_preference_is_still_kept(self) -> None:
        self.assertTrue(durable(Draft("preference", "The user wants answers in Swedish.")))

    def test_parse_drafts_normalises_the_kind(self) -> None:
        drafts = parse_drafts('[{"kind": "Procedure", "text": "Run make test before committing."}]')
        self.assertEqual(drafts[0].kind, "semantic")

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hauhau.evolve.prompts import PromptStore, evolve_prompt, validate
from hauhau.evolve.suite import DEFAULT_TASKS, SuiteResult, Task, _judge, available, run_suite
from hauhau.evolve.tune import Candidate, hill_climb
from hauhau.llm import Completion, LLMError, ToolCall


class FakeClient:
    """Replies are keyed by the task text so a suite run is reproducible."""

    def __init__(self, tool: str | None = None, text: str = "", error: Exception | None = None) -> None:
        self.tool = tool
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, **kwargs) -> Completion:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if self.error:
            raise self.error
        calls = [ToolCall(id="c1", name=self.tool, arguments="{}")] if self.tool else []
        return Completion(content=self.text, tool_calls=calls, finish_reason="stop")


class TestJudge(unittest.TestCase):
    def test_tool_task_passes_on_the_expected_tool(self) -> None:
        task = Task("t", "list files", ("list_dir", "shell"))
        result = run_suite(
            FakeClient(tool="list_dir"), (task,), system_prompt="x", tools=None
        )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.failures, [])

    def test_tool_task_fails_on_the_wrong_tool(self) -> None:
        task = Task("t", "list files", ("list_dir",))
        result = run_suite(FakeClient(tool="write_file"), (task,), system_prompt="x", tools=None)
        self.assertEqual(result.score, 0.0)
        self.assertIn("write_file", result.failures[0][1])

    def test_tool_task_fails_when_it_only_talks(self) -> None:
        task = Task("t", "list files", ("list_dir",))
        result = run_suite(FakeClient(text="I would list them"), (task,), system_prompt="x", tools=None)
        self.assertEqual(result.score, 0.0)
        self.assertIn("prose", result.failures[0][1])

    def test_text_task_rejects_a_tool_call(self) -> None:
        task = Task("t", "reply pong", contains=("pong",))
        result = run_suite(FakeClient(tool="shell", text="pong"), (task,), system_prompt="x", tools=None)
        self.assertEqual(result.score, 0.0)
        self.assertIn("instead of answering", result.failures[0][1])

    def test_text_task_matches_case_insensitively(self) -> None:
        task = Task("t", "reply pong", contains=("pong",))
        result = run_suite(FakeClient(text="PONG"), (task,), system_prompt="x", tools=None)
        self.assertEqual(result.score, 1.0)

    def test_wrong_arguments_fail_a_tool_task(self) -> None:
        """Picking shell is not enough if the command is not the right one."""
        task = Task("t", "how much disk is free", ("shell",), args_contains=("df",))
        good = Completion(
            content="", tool_calls=[ToolCall(id="c1", name="shell", arguments='{"cmd":"df -h"}')]
        )

        def client(completion):
            class C:
                def chat(self, *_a, **_k):
                    return completion

            return C()

        self.assertEqual(run_suite(client(good), (task,), system_prompt="x", tools=None).score, 1.0)
        wrong = Completion(
            content="", tool_calls=[ToolCall(id="c1", name="shell", arguments='{"cmd":"echo hi"}')]
        )
        result = run_suite(client(wrong), (task,), system_prompt="x", tools=None)
        self.assertEqual(result.score, 0.0)
        self.assertIn("wrong command", result.failures[0][1])

    def test_summary_labels_the_score(self) -> None:
        result = SuiteResult(passed=3, total=12)
        self.assertEqual(result.summary(), "3/12 tasks (25%)")
        self.assertEqual(SuiteResult().summary(), "0/0 tasks (0%)")


class TestAvailability(unittest.TestCase):
    def test_tasks_needing_missing_tools_are_dropped(self) -> None:
        kept = available(DEFAULT_TASKS, ["shell", "read_file"])
        names = {t.name for t in kept}
        self.assertNotIn("store-fact", names, "rag_write is unavailable with the brain off")
        self.assertIn("run-tests", names)
        self.assertTrue(all(not t.tools_any or "shell" in t.tools_any or "read_file" in t.tools_any for t in kept))

    def test_llm_error_propagates(self) -> None:
        with self.assertRaises(LLMError):
            run_suite(FakeClient(error=LLMError("down")), DEFAULT_TASKS[:1], system_prompt="x", tools=None)


class TestPromptStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "prompts"
        self.store = PromptStore(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_numbers_from_one_and_round_trips(self) -> None:
        self.assertEqual(self.store.versions(), [])
        first = self.store.add("a" * 300)
        second = self.store.add("b" * 320, parent=1, score=0.75)
        self.assertEqual((first.number, second.number), (1, 2))
        self.assertEqual(self.store.text(2), "b" * 320)
        self.assertEqual(len(self.store.text(2)), 320, "trailing newline is not part of the prompt")
        self.assertEqual(second.score, 0.75)

    def test_active_pointer(self) -> None:
        self.assertIsNone(self.store.active())
        version = self.store.add("x" * 300)
        self.store.activate(version.number)
        self.assertEqual(self.store.active().number, 1)
        with self.assertRaises(FileNotFoundError):
            self.store.activate(99)

    def test_corrupt_index_lines_are_skipped(self) -> None:
        self.store.add("x" * 300)
        with self.store.index_path.open("a", encoding="utf-8") as fh:
            fh.write("{not json}\n")
        self.assertEqual(len(self.store.versions()), 1)

    def test_missing_file_for_indexed_version_is_ignored(self) -> None:
        version = self.store.add("x" * 300)
        version.path.unlink()
        self.assertEqual(self.store.versions(), [])

    def test_empty_prompt_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add("   ")


class TestValidate(unittest.TestCase):
    def test_short_and_long_are_rejected(self) -> None:
        self.assertIn("too short", validate("short"))
        self.assertIn("too long", validate("y" * 7000))

    def test_stray_brace_is_rejected(self) -> None:
        # The agent formats the personality with name=..., so this would crash
        # every turn after adoption.
        self.assertIn("{name} template", validate("z" * 250 + " {oops}"))

    def test_floor_is_proportional_to_the_current_prompt(self) -> None:
        # A 199-char rewrite of a 161-char prompt was rejected by a fixed floor.
        current = "x" * 400
        self.assertIsNone(validate("y" * 200, current=current))
        self.assertIn("too short", validate("y" * 120, current=current))
        self.assertIsNone(validate("z" * 80))

    def test_plain_name_placeholder_is_fine(self) -> None:
        self.assertIsNone(validate("You are {name}, a terminal agent. " + "k" * 250))


class TestEvolvePrompt(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(Path(self._tmp.name) / "prompts")
        self.current = "You are {name}. Be careful. " + "c" * 250

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, scores: list[tuple[int, int]]):
        calls: list[str] = []

        def run(text: str) -> SuiteResult:
            passed, total = scores.pop(0)
            calls.append(text)
            result = SuiteResult(passed=passed, total=total)
            if passed < total:
                result.failures = [("run-tests", "answered in prose")]
            return result

        return run, calls

    def test_improvement_is_adopted_and_versioned(self) -> None:
        candidate = "You are {name}. Always act. " + "d" * 250
        run, _calls = self._run([(4, 12), (7, 12)])
        result = evolve_prompt(FakeClient(text=candidate), self.store, run, self.current)
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.version, 1)
        self.assertEqual(self.store.active().number, 1)
        self.assertEqual(self.store.text(1).strip(), candidate)

    def test_regression_is_not_adopted(self) -> None:
        run, _ = self._run([(7, 12), (5, 12)])
        result = evolve_prompt(FakeClient(text="w" * 400), self.store, run, self.current)
        self.assertFalse(result.accepted)
        self.assertIn("no improvement", result.reason)
        self.assertEqual(self.store.versions(), [], "a losing candidate is not even stored")

    def test_equal_score_is_not_adopted(self) -> None:
        run, _ = self._run([(6, 12), (6, 12)])
        result = evolve_prompt(FakeClient(text="q" * 400), self.store, run, self.current)
        self.assertFalse(result.accepted)

    def test_perfect_suite_skips_the_model_entirely(self) -> None:
        client = FakeClient(text="unused")
        run, calls = self._run([(12, 12)])
        result = evolve_prompt(client, self.store, run, self.current)
        self.assertFalse(result.accepted)
        self.assertIn("already passes", result.reason)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(calls), 1)

    def test_nonsense_candidate_never_reaches_the_suite(self) -> None:
        run, calls = self._run([(4, 12)])
        result = evolve_prompt(FakeClient(text="too short"), self.store, run, self.current)
        self.assertFalse(result.accepted)
        self.assertIn("candidate rejected", result.reason)
        self.assertEqual(len(calls), 1, "only the baseline ran")

    def test_identical_prompt_is_rejected(self) -> None:
        run, _ = self._run([(4, 12)])
        result = evolve_prompt(FakeClient(text=self.current), self.store, run, self.current)
        self.assertFalse(result.accepted)
        self.assertIn("same prompt", result.reason)

    def test_rewrite_failure_is_reported(self) -> None:
        run, _ = self._run([(4, 12)])
        result = evolve_prompt(FakeClient(error=LLMError("down")), self.store, run, self.current)
        self.assertFalse(result.accepted)
        self.assertIn("rewrite failed", result.reason)


class TestVersionNumbers(unittest.TestCase):
    """A rollback point must never be overwritten."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lost_index_entry_does_not_reissue_a_live_number(self) -> None:
        self.store.add("prompt one " * 25)
        self.store.add("prompt two " * 25)
        self.store.add("PRECIOUS v3 prompt " * 20)
        lines = self.store.index_path.read_text().splitlines()
        # Simulate a truncated/edited index: the v3 line is gone but v3.md is not.
        self.store.index_path.write_text(lines[0] + "\n" + lines[1] + "\n{oops}\n")
        version = self.store.add("brand new rewrite " * 20)
        self.assertEqual(version.number, 4)
        self.assertIn("PRECIOUS", (self.store.versions_dir / "v3.md").read_text())

    def test_orphan_version_file_is_not_clobbered(self) -> None:
        self.store.versions_dir.mkdir(parents=True, exist_ok=True)
        (self.store.versions_dir / "v7.md").write_text("orphan\n")
        version = self.store.add("fresh prompt " * 25)
        self.assertEqual(version.number, 8)
        self.assertEqual((self.store.versions_dir / "v7.md").read_text(), "orphan\n")

    def test_a_hand_edited_score_does_not_crash_the_listing(self) -> None:
        self.store.add("prompt one " * 25)
        lines = self.store.index_path.read_text().splitlines()
        self.store.index_path.write_text(lines[0].replace('"score": null', '"score": "0.9"') + "\n")
        versions = self.store.versions()
        self.assertEqual(len(versions), 1)
        # A string score used to reach "/prompt" and raise on f"{score:.0%}".
        self.assertEqual(versions[0].score, 0.9)
        self.assertEqual(f"{versions[0].score:.0%}", "90%")


class TestProseTask(unittest.TestCase):
    def test_the_natural_answer_passes(self) -> None:
        task = Task("no-tool-answer", "what does ls do?", contains=("list", "lists"))
        for answer in ("The ls command lists directory contents.", "ls lists files and directories."):
            self.assertTrue(_judge(task, Completion(content=answer))[0], answer)


class TestJudgeIsNotSubstring(unittest.TestCase):
    def test_a_longer_number_does_not_pass(self) -> None:
        task = Task("arithmetic", "What is 1+1? Answer with the number only.", contains=("2",))
        self.assertFalse(_judge(task, Completion(content="12"))[0])
        self.assertTrue(_judge(task, Completion(content="2"))[0])

    def test_negated_answer_does_not_pass(self) -> None:
        task = Task("arithmetic", "What is 1+1? Answer with the number only.", contains=("2",))
        self.assertFalse(_judge(task, Completion(content="It is not 2."))[0])

    def test_prose_still_passes_a_word_task(self) -> None:
        task = Task("exact-reply", "Reply with exactly: pong", contains=("pong",), exact=True)
        self.assertTrue(_judge(task, Completion(content="pong"))[0])
        self.assertTrue(_judge(task, Completion(content="PONG!"))[0])
        self.assertFalse(_judge(task, Completion(content="I will not reply pong."))[0])

    def test_an_extension_still_matches_a_filename(self) -> None:
        # Requiring a left word boundary made ".py" unmatchable in "mod.py".
        task = Task("find-files", "list python files", contains=(".py",))
        self.assertTrue(_judge(task, Completion(content="the file is mod.py"))[0])
        self.assertFalse(_judge(task, Completion(content="there is no python here"))[0])

    def test_a_tool_argument_is_matched_as_a_word(self) -> None:
        task = Task("list-files", "list files", tools_any=("shell",), args_contains=("ls", "."))
        wrong = Completion(content="", tool_calls=[ToolCall(id="c", name="shell", arguments='{"cmd": "false"}')])
        right = Completion(content="", tool_calls=[ToolCall(id="c", name="shell", arguments='{"cmd": "ls ."}')])
        self.assertFalse(_judge(task, wrong)[0])
        self.assertTrue(_judge(task, right)[0])


class TestHillClimbAdoption(unittest.TestCase):
    def test_the_winning_trial_is_the_one_marked_adopted(self) -> None:
        options = [Candidate("model.n_threads", 2, "a"), Candidate("model.n_threads", 2, "b")]
        scores = iter([10.0, 20.0])
        result = hill_climb(10.0, options, lambda candidate: next(scores), max_trials=2)
        adopted = [trial for trial in result.trials if trial.adopted]
        self.assertEqual(len(adopted), 1)
        self.assertEqual(adopted[0].score, 20.0)


if __name__ == "__main__":
    unittest.main()

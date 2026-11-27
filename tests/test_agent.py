from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hauhau.agent import (
    UI,
    Agent,
    asks_instead_of_acting,
    commands_to_run,
    says_it_cannot,
)
from hauhau.config import Config
from hauhau.llm import Completion, ToolCall, Usage
from hauhau.tools.base import Registry, Tool, fail, ok


class FakeClient:
    """Scripted stand-in for LlamaClient."""

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = list(completions)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, **kwargs) -> Completion:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if not self.completions:
            return Completion(content="(ran out of scripted replies)", finish_reason="stop")
        return self.completions.pop(0)


def tool_call(name: str = "echo", args: str = '{"text":"hi"}', call_id: str = "c1") -> Completion:
    return Completion(tool_calls=[ToolCall(id=call_id, name=name, arguments=args)], finish_reason="tool_calls")


def reply(text: str) -> Completion:
    return Completion(content=text, finish_reason="stop", usage=Usage(completion_tokens=5, tok_per_second=27.0))


def make_registry(log: list) -> Registry:
    def echo(args):
        log.append(args.get("text"))
        return ok(f"echoed {args.get('text')}")

    return Registry([
        Tool("echo", "Echo text back.", {"type": "object", "properties": {"text": {"type": "string"}},
                                           "required": ["text"]}, echo),
        Tool("boom", "Always raises.", {"type": "object", "properties": {}},
             lambda args: (_ for _ in ()).throw(RuntimeError("kaboom"))),
        Tool("denied", "Always fails cleanly.", {"type": "object", "properties": {}},
             lambda args: fail("nope")),
    ])


class SpyUI(UI):
    """Records what the CLI would have shown."""

    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.tokens: list[str] = []

    def status(self, text: str) -> None:
        self.statuses.append(text)

    def token(self, text: str) -> None:
        self.tokens.append(text)


class FakeAudit:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def log(self, event: str, **fields) -> None:
        self.events.append({"event": event, **fields})


class AgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.cfg.tools.workspace = str(self.root)
        self.cfg.brain.enabled = False
        self.log: list = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_agent(self, client, ui=None, **cfg_overrides) -> Agent:
        for key, value in cfg_overrides.items():
            setattr(self.cfg.agent, key, value)
        return Agent(self.cfg, client, make_registry(self.log), brain=None, ui=ui or UI())


class TestToolLoop(AgentTestCase):
    def test_single_tool_call_then_answer(self) -> None:
        client = FakeClient([tool_call(), reply("done: hi")])
        agent = self.make_agent(client)
        text = agent.run_turn("say hi")

        self.assertEqual(text, "done: hi")
        self.assertEqual(self.log, ["hi"])
        roles = [m["role"] for m in agent.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertEqual(agent.messages[1]["tool_calls"][0]["function"]["name"], "echo")
        self.assertEqual(agent.messages[2]["tool_call_id"], "c1")
        self.assertIn("echoed hi", agent.messages[2]["content"])

    def test_tool_message_shape_matches_openai(self) -> None:
        client = FakeClient([tool_call(), reply("ok")])
        agent = self.make_agent(client)
        agent.run_turn("go")
        tool_msg = agent.messages[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["name"], "echo")
        self.assertIn("tool_call_id", tool_msg)

    def test_tools_are_offered_on_first_call(self) -> None:
        client = FakeClient([reply("no tools needed")])
        agent = self.make_agent(client)
        agent.run_turn("hello")
        self.assertIsNotNone(client.calls[0]["tools"])
        names = [t["function"]["name"] for t in client.calls[0]["tools"]]
        self.assertIn("echo", names)

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        multi = Completion(
            tool_calls=[
                ToolCall(id="a", name="echo", arguments='{"text":"one"}'),
                ToolCall(id="b", name="echo", arguments='{"text":"two"}'),
            ],
            finish_reason="tool_calls",
        )
        client = FakeClient([multi, reply("both done")])
        agent = self.make_agent(client)
        agent.run_turn("do both")
        self.assertEqual(self.log, ["one", "two"])
        self.assertEqual([m["role"] for m in agent.messages].count("tool"), 2)

    def test_unknown_tool_reports_available_tools(self) -> None:
        client = FakeClient([tool_call(name="nope"), reply("adapted")])
        agent = self.make_agent(client)
        agent.run_turn("try")
        self.assertIn("unknown tool", agent.messages[2]["content"])
        self.assertIn("echo", agent.messages[2]["content"])

    def test_crashing_tool_does_not_kill_the_turn(self) -> None:
        client = FakeClient([tool_call(name="boom"), reply("recovered")])
        agent = self.make_agent(client)
        text = agent.run_turn("boom")
        self.assertEqual(text, "recovered")
        self.assertIn("kaboom", agent.messages[2]["content"])

    def test_failed_tool_result_is_passed_through(self) -> None:
        client = FakeClient([tool_call(name="denied"), reply("fine")])
        agent = self.make_agent(client)
        agent.run_turn("x")
        self.assertIn("nope", agent.messages[2]["content"])

    def test_malformed_arguments_are_reported_not_raised(self) -> None:
        client = FakeClient([tool_call(args="{not json"), reply("handled")])
        agent = self.make_agent(client)
        text = agent.run_turn("x")
        self.assertEqual(text, "handled")
        self.assertIn("malformed JSON", agent.messages[2]["content"])

    def test_step_budget_forces_final_answer_without_tools(self) -> None:
        client = FakeClient([tool_call(call_id="1"), tool_call(call_id="2"), reply("forced")])
        agent = self.make_agent(client, max_steps=2)
        text = agent.run_turn("loop please")
        self.assertEqual(text, "forced")
        self.assertEqual(len(client.calls), 3)
        self.assertIsNotNone(client.calls[1]["tools"])
        self.assertIsNone(client.calls[2]["tools"], "final call must withhold tools")


class TestTruncation(AgentTestCase):
    def test_prose_uses_the_configured_max_tokens(self) -> None:
        client = FakeClient([reply("short")])
        agent = self.make_agent(client)
        agent.run_turn("hi")
        self.assertEqual(client.calls[0]["max_tokens"], self.cfg.generation.max_tokens)

    def test_tool_turn_uses_the_same_budget_as_prose(self) -> None:
        # Regression: tool turns were capped at 512, which silently truncated
        # any answer longer than ~400 words mid-sentence.
        client = FakeClient([tool_call(), reply("done")])
        agent = self.make_agent(client)
        agent.run_turn("go")
        for call in client.calls:
            self.assertEqual(call["max_tokens"], self.cfg.generation.max_tokens)

    def test_truncated_tool_call_is_not_executed(self) -> None:
        cut = Completion(
            tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text":"partial')],
            finish_reason="length",
        )
        client = FakeClient([cut, reply("retried smaller")])
        agent = self.make_agent(client)
        agent.run_turn("write a lot")
        self.assertEqual(self.log, [], "a cut-off call must not run with guessed arguments")
        self.assertIn("cut off", agent.messages[2]["content"])
        self.assertIn("NOT run", agent.messages[2]["content"])

    def test_cut_off_answer_is_announced(self) -> None:
        ui = SpyUI()
        client = FakeClient([Completion(content="This sentence stops", finish_reason="length")])
        agent = self.make_agent(client, ui=ui)
        agent.run_turn("write a long essay")
        self.assertTrue(any("cut off" in s for s in ui.statuses), ui.statuses)
        self.assertTrue(any("max_tokens" in s for s in ui.statuses), ui.statuses)

    def test_complete_answer_raises_no_warning(self) -> None:
        ui = SpyUI()
        agent = self.make_agent(FakeClient([reply("all done")]), ui=ui)
        agent.run_turn("hi")
        self.assertEqual(ui.statuses, [])


class TestCompression(AgentTestCase):
    def test_short_conversation_is_untouched(self) -> None:
        client = FakeClient([])
        agent = self.make_agent(client)
        agent.messages = [{"role": "user", "content": "hi"}]
        self.assertFalse(agent.compress_if_needed())

    def test_long_conversation_is_summarised(self) -> None:
        client = FakeClient([reply("- user asked about disks\n- we checked df")])
        agent = self.make_agent(client, context_budget_tokens=60, keep_recent_turns=1)
        agent.messages = [
            {"role": "user", "content": "a" * 200},
            {"role": "assistant", "content": "b" * 200},
            {"role": "user", "content": "c" * 200},
            {"role": "assistant", "content": "d" * 200},
        ]
        self.assertTrue(agent.compress_if_needed())
        self.assertIn("df", agent.summary)
        self.assertLess(len(agent.messages), 4)
        self.assertEqual(agent.messages[0]["content"], "c" * 200)

    def test_oversized_summary_is_folded(self) -> None:
        """A summary that outgrew the budget used to block compression forever."""
        client = FakeClient([reply("- we discussed disks\n- df showed 40 GB free")])
        agent = self.make_agent(client, context_budget_tokens=600, keep_recent_turns=1)
        agent.summary = "old note about the machine. " * 400  # ~3k tokens on its own
        agent.messages = [
            {"role": "user", "content": "a" * 200},
            {"role": "assistant", "content": "b" * 200},
            {"role": "user", "content": "c" * 200},
            {"role": "assistant", "content": "d" * 200},
        ]
        self.assertTrue(agent.compress_if_needed())
        self.assertLessEqual(len(agent.summary), agent.summary_cap_tokens() * 4 + 40)
        self.assertIn("df", agent.summary)
        # Settles: a second pass has nothing left to do, so no model thrash.
        self.assertFalse(agent.compress_if_needed())

    def test_fold_summary_falls_back_when_the_model_returns_nothing(self) -> None:
        agent = self.make_agent(FakeClient([reply("")]), context_budget_tokens=600)
        folded = agent._fold_summary("x" * 4000, 200)
        self.assertLess(len(folded), 4000)
        self.assertIn("earlier notes dropped", folded)

    def test_never_cuts_between_tool_call_and_result(self) -> None:
        agent = self.make_agent(FakeClient([]))
        agent.messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "out"},
        ]
        self.assertEqual(agent._safe_split(2), 0, "trimming here would orphan the tool result")

    def test_keeps_the_requested_turns(self) -> None:
        # Regression: the first boundary at or after the target was taken, which
        # kept one message where `keep` promised several.
        agent = self.make_agent(FakeClient([]))
        agent.messages = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "assistant", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "e"},
            {"role": "assistant", "content": "f"},
            {"role": "assistant", "content": "g"},
            {"role": "assistant", "content": "h"},
            {"role": "assistant", "content": "i"},
        ]
        keep = 4
        cut = agent._safe_split(keep)
        self.assertEqual(cut, 5)
        self.assertGreaterEqual(len(agent.messages) - cut, keep)

    def test_makes_progress_when_no_late_boundary_exists(self) -> None:
        # Regression: with a small budget the tail can be one tool-loop turn, so
        # no user message sits at the split point. Returning 0 there meant the
        # transcript was never trimmed again and grew past the budget forever.
        agent = self.make_agent(FakeClient([]))
        agent.messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
            {"role": "tool", "tool_call_id": "t", "content": "out"},
            {"role": "assistant", "content": "4"},
        ]
        self.assertEqual(agent._safe_split(2), 2, "must fall back to the earliest boundary")

    def test_no_boundary_means_no_cut(self) -> None:
        agent = self.make_agent(FakeClient([]))
        agent.messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
            {"role": "tool", "tool_call_id": "t", "content": "out"},
        ]
        self.assertEqual(agent._safe_split(2), 0)

    def test_cuts_at_user_boundary(self) -> None:
        agent = self.make_agent(FakeClient([]))
        agent.messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "out"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "answer"},
        ]
        self.assertEqual(agent._safe_split(2), 3)
        self.assertEqual(agent.messages[3]["role"], "user")


class TestPrompt(AgentTestCase):
    def test_system_prompt_has_workspace_and_date(self) -> None:
        agent = self.make_agent(FakeClient([]))
        prompt = agent.system_prompt("query")
        self.assertIn(str(self.root), prompt)
        self.assertIn("Today:", prompt)
        self.assertIn(self.cfg.agent.name, prompt)

    def test_memory_block_is_injected_when_present(self) -> None:
        agent = self.make_agent(FakeClient([]))

        class FakeBrain:
            def context_block(self, query):
                return "What you remember: the user runs macOS."

        agent.brain = FakeBrain()
        self.cfg.brain.enabled = True
        self.assertIn("runs macOS", agent.system_prompt("what os"))


class TestDetectors(unittest.TestCase):
    def test_questions_are_detected(self) -> None:
        for text in ("Vill du att jag listar diskarna?", "  Ska jag köra den?  "):
            self.assertTrue(asks_instead_of_acting(text), text)

    def test_answers_are_not_questions(self) -> None:
        self.assertFalse(asks_instead_of_acting("Klart, diskarna är listade."))
        self.assertFalse(asks_instead_of_acting(""))
        long_answer = "Här är en lång förklaring. " * 10 + "Visst?"
        self.assertFalse(asks_instead_of_acting(long_answer), "a real answer that ends in ?")

    def test_swedish_modal_word_order_is_recognised(self) -> None:
        for text in ("Det kan jag inte.", "Det kan jag tyvärr inte just nu.",
                     "Jag kan inte hitta filen.", "Jag vet inte hur man gör.",
                     "Jag har inte tillgång till den mappen."):
            self.assertTrue(says_it_cannot(text), text)

    def test_pasted_commands_are_detected(self) -> None:
        for text in ("Kör det här:\n```bash\ndu -sh * | sort -rh\n```",
                     "```sh\nls -la\n```",
                     "```zsh\ndf -h\n```"):
            self.assertTrue(commands_to_run(text), text)

    def test_prose_and_other_fences_are_not_commands(self) -> None:
        for text in ("Klart, disken är full.", "```python\nprint(1)\n```",
                     "```\nplain fence\n```", ""):
            self.assertFalse(commands_to_run(text), text)

    def test_ordinary_answers_are_not_refusals(self) -> None:
        for text in ("Klart, jag hittade filen.", "jag kan det här nu", ""):
            self.assertFalse(says_it_cannot(text), text)


class TestAutonomy(AgentTestCase):
    """The bot must act, not hand the question back."""

    def test_rules_are_in_the_prompt_by_default(self) -> None:
        agent = self.make_agent(FakeClient([]))
        prompt = agent.system_prompt("hi")
        self.assertIn("ask only as a last resort", prompt)
        self.assertIn("note_lesson", prompt)

    def test_rules_can_be_turned_off(self) -> None:
        agent = self.make_agent(FakeClient([]), autonomy=False)
        self.assertNotIn("ask only as a last resort", agent.system_prompt("hi"))

    def test_a_question_is_answered_by_acting(self) -> None:
        client = FakeClient([
            reply("Vill du att jag listar diskarna?"),
            tool_call(name="echo", args='{"text":"df -h"}'),
            reply("Klart: diskarna är listade."),
        ])
        ui = SpyUI()
        agent = self.make_agent(client, ui=ui)
        text = agent.run_turn("kolla diskarna")

        self.assertEqual(text, "Klart: diskarna är listade.")
        self.assertEqual(agent.nudges_this_turn, 1)
        self.assertEqual(len(client.calls), 3, "one extra step, not a new turn")
        self.assertIn("instead of acting", client.calls[1]["messages"][0]["content"])
        self.assertEqual(self.log, ["df -h"], "the nudge made it use a tool")
        self.assertIn("working out what was meant", " ".join(ui.statuses))

    def test_the_nudge_is_audited(self) -> None:
        audit = FakeAudit()
        client = FakeClient([reply("Vill du att jag kollar?"), reply("Klart.")])
        agent = Agent(self.cfg, client, make_registry(self.log), brain=None, audit=audit)
        agent.run_turn("kolla läget")

        self.assertEqual([e["event"] for e in audit.events], ["nudge"])
        self.assertEqual(audit.events[0]["kind"], "ambiguous")
        self.assertEqual(audit.events[0]["count"], 1)

    def test_saying_it_cannot_triggers_the_stuck_nudge(self) -> None:
        client = FakeClient([
            reply("Det kan jag inte."),
            reply("Nu löste jag det: paketet heter ripgrep."),
        ])
        agent = self.make_agent(client)
        text = agent.run_turn("hitta ripgrep i hemkatalogen")

        self.assertEqual(text, "Nu löste jag det: paketet heter ripgrep.")
        self.assertEqual(agent.nudges_this_turn, 1)
        self.assertIn("You said you cannot do this", client.calls[1]["messages"][0]["content"])

    def test_a_pasted_command_is_nudged_into_running_it(self) -> None:
        audit = FakeAudit()
        client = FakeClient([
            reply("Låt oss kolla.\n```bash\ndu -sh /Users/johan/* | sort -rh | head\n```"),
            tool_call(name="echo", args='{"text":"du -sh"}'),
            reply("Största mappen är ~/Documents."),
        ])
        agent = Agent(self.cfg, client, make_registry(self.log), brain=None, audit=audit)
        text = agent.run_turn("vad tar upp disken")

        self.assertEqual(text, "Största mappen är ~/Documents.")
        self.assertEqual(self.log, ["du -sh"])
        self.assertEqual(audit.events[0]["kind"], "act")
        self.assertIn("instead of running it", client.calls[1]["messages"][0]["content"])

    def test_a_question_after_a_failed_tool_call_is_kept(self) -> None:
        client = FakeClient([
            tool_call(name="denied", args="{}"),
            reply("Filen gick inte att läsa - ska jag försöka med sudo?"),
        ])
        agent = self.make_agent(client)
        text = agent.run_turn("läs filen")

        self.assertEqual(text, "Filen gick inte att läsa - ska jag försöka med sudo?")
        self.assertEqual(agent.nudges_this_turn, 0)
        self.assertEqual(len(client.calls), 2, "a real obstacle must reach the user")

    def test_a_question_after_a_successful_tool_call_is_nudged(self) -> None:
        client = FakeClient([
            tool_call(name="echo", args='{"text":"a"}'),
            reply("Vilken katalog vill du att jag tittar i?"),
            reply("Klart: /Users/johan/Documents är störst."),
        ])
        agent = self.make_agent(client)
        text = agent.run_turn("vad tar upp disken")

        self.assertEqual(text, "Klart: /Users/johan/Documents är störst.")
        self.assertEqual(agent.nudges_this_turn, 1)

    def test_nudges_are_capped_at_two(self) -> None:
        client = FakeClient([reply("Vill du att jag gör det?"), reply("Vill du att jag gör det?"),
                             reply("Vill du att jag gör det?")])
        agent = self.make_agent(client)
        text = agent.run_turn("gör det")

        self.assertEqual(agent.nudges_this_turn, 2)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(text, "Vill du att jag gör det?")

    def test_autonomy_off_returns_the_question_at_once(self) -> None:
        client = FakeClient([reply("Vill du att jag listar diskarna?")])
        agent = self.make_agent(client, autonomy=False)
        text = agent.run_turn("kolla diskarna")

        self.assertEqual(text, "Vill du att jag listar diskarna?")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(agent.nudges_this_turn, 0)

    def test_the_nudge_does_not_leak_into_history_or_the_next_turn(self) -> None:
        client = FakeClient([reply("Vill du att jag kollar?"),
                             reply("Klart."),
                             reply("Klart igen.")])
        agent = self.make_agent(client)
        agent.run_turn("kolla läget")
        agent.run_turn("igen")

        for msg in agent.messages:
            self.assertNotIn("instead of acting", str(msg.get("content") or ""))
        self.assertNotIn("instead of acting", client.calls[2]["messages"][0]["content"])


class TestDetectorFalsePositives(unittest.TestCase):
    """A detector that fires on ordinary prose costs a step and misleads."""

    def test_an_answer_with_a_closing_offer_is_not_a_question(self) -> None:
        for text in ("Klart, filen är borta. Vill du att jag återställer den?",
                     "Disken är full. Ska jag rensa cachen?",
                     "Jag hittade 3 filer. Vilken vill du se?"):
            self.assertFalse(asks_instead_of_acting(text), text)

    def test_only_question_sentences_are_a_hand_back(self) -> None:
        self.assertTrue(asks_instead_of_acting("Vill du att jag listar diskarna?"))
        self.assertTrue(asks_instead_of_acting("Vill du att jag kollar? Eller ska jag hoppa över?"))

    def test_a_quoted_error_is_not_the_agent_saying_it_cannot(self) -> None:
        for text in ("Felet var 'unable to resolve host'; jag rättade DNS-posten.",
                     "The build said unable to find crate, then I added it to Cargo.toml.",
                     "`unable to` betyder att den inte hittade värden."):
            self.assertFalse(says_it_cannot(text), text)

    def test_advice_containing_kan_inte_is_not_a_refusal(self) -> None:
        text = "Klart. Du kan köra `make` om den inte redan är installerad."
        self.assertFalse(says_it_cannot(text))

    def test_untagged_fences_and_inline_spans_count_as_commands(self) -> None:
        for text in ("```\ndu -sh *\n```", "Kör bara `du -sh *` så ser du det.", "`pmset -g`"):
            self.assertTrue(commands_to_run(text), text)

    def test_names_in_backticks_are_not_commands(self) -> None:
        for text in ("`note_lesson` skriver till minnet.", "se `config.toml`"):
            self.assertFalse(commands_to_run(text), text)


class TestTurnIntegrity(AgentTestCase):
    """A turn must never leave the transcript in a shape the backend rejects."""

    def test_an_interrupted_tool_call_still_answers_the_call(self) -> None:
        def interrupted(_args):
            raise KeyboardInterrupt

        agent = self.make_agent(FakeClient([]))
        agent.registry = Registry(
            [Tool("boom", "Interrupts.", {"type": "object", "properties": {}}, interrupted)]
        )
        agent.messages.append({"role": "user", "content": "go"})
        agent._record_assistant(
            Completion(
                tool_calls=[ToolCall(id="c1", name="boom", arguments="{}")], finish_reason="tool_calls"
            )
        )

        with self.assertRaises(KeyboardInterrupt):
            agent._execute(ToolCall(id="c1", name="boom", arguments="{}"))

        self.assertEqual(agent.messages[-1]["role"], "tool", "the announced call must be answered")
        self.assertEqual(agent.messages[-1]["tool_call_id"], "c1")
        self.assertIn("interrupted", agent.messages[-1]["content"])

    def test_auditing_non_string_arguments_does_not_raise(self) -> None:
        audit = FakeAudit()
        client = FakeClient([
            Completion(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                       finish_reason="tool_calls"),
            reply("done"),
        ])
        agent = Agent(self.cfg, client, make_registry(self.log), brain=None, audit=audit)
        self.assertEqual(agent.run_turn("go"), "done")
        self.assertEqual([e["event"] for e in audit.events], ["tool"])

    def test_a_failed_tool_stops_every_nudge(self) -> None:
        client = FakeClient([
            tool_call(name="denied", args="{}"),
            reply("Jag kan inte läsa filen - den kräver root."),
        ])
        agent = self.make_agent(client)
        text = agent.run_turn("läs filen")

        self.assertEqual(text, "Jag kan inte läsa filen - den kräver root.")
        self.assertEqual(agent.nudges_this_turn, 0, "the obstacle is real: show it")

    def test_the_budget_answer_gets_no_nudge_it_cannot_obey(self) -> None:
        client = FakeClient([reply("Vill du att jag kollar?"), reply("Ingen aning.")])
        agent = self.make_agent(client, max_steps=1)
        text = agent.run_turn("kolla")

        self.assertEqual(text, "Ingen aning.")
        self.assertNotIn("instead of acting", client.calls[1]["messages"][0]["content"])
        self.assertIsNone(agent._nudge)


class TestPersistence(AgentTestCase):
    def test_persist_and_load_roundtrip(self) -> None:
        agent = self.make_agent(FakeClient([reply("hello")]))
        agent.run_turn("hi")
        path = self.root / "sessions" / "s1.json"
        agent.persist(path)

        fresh = self.make_agent(FakeClient([]))
        self.assertTrue(fresh.load(path))
        self.assertEqual(fresh.messages, agent.messages)
        self.assertEqual(fresh.turn_count(), 1)

    def test_load_missing_file_returns_false(self) -> None:
        agent = self.make_agent(FakeClient([]))
        self.assertFalse(agent.load(self.root / "nope.json"))

    def test_reset_clears_context(self) -> None:
        agent = self.make_agent(FakeClient([reply("x")]))
        agent.run_turn("hi")
        agent.reset()
        self.assertEqual(agent.messages, [])
        self.assertEqual(agent.summary, "")


if __name__ == "__main__":
    unittest.main()


class TestSessionShape(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        cfg = Config()
        cfg.brain.enabled = False
        self.agent = Agent(cfg, FakeClient([]), Registry([]), brain=None)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, payload) -> Path:
        path = self.root / "s.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    def test_a_good_session_round_trips(self) -> None:
        path = self.root / "good.json"
        self.agent.messages = [{"role": "user", "content": "hi"}]
        self.agent.summary = "some summary"
        self.agent.persist(path)
        other = Agent(Config(), FakeClient([]), Registry([]), brain=None)
        self.assertTrue(other.load(path))
        self.assertEqual(other.messages, self.agent.messages)
        self.assertEqual(other.summary, "some summary")

    def test_rejects_messages_that_are_not_a_list(self) -> None:
        for payload in ({"messages": {"a": 1}}, {"messages": "hello"}, {"messages": [], "summary": 5}):
            self.assertFalse(self.agent.load(self._write(payload)), payload)
        self.assertEqual(self.agent.messages, [])

    def test_rejects_a_top_level_list(self) -> None:
        self.assertFalse(self.agent.load(self._write([1, 2, 3])))

    def test_rejects_a_message_without_a_role(self) -> None:
        self.assertFalse(self.agent.load(self._write({"messages": [{"content": "x"}]})))

    def test_rejects_broken_json(self) -> None:
        self.assertFalse(self.agent.load(self._write("{not json")))


class TestPersonalityBraces(unittest.TestCase):
    def test_an_unknown_placeholder_does_not_crash_every_turn(self) -> None:
        cfg = Config()
        cfg.brain.enabled = False
        cfg.agent.personality = "You are {name}. Reply about {topic}."
        agent = Agent(cfg, FakeClient([]), Registry([]), brain=None)
        prompt = agent.system_prompt("hi")
        self.assertIn("You are Hauhau", prompt)
        self.assertIn("{topic}", prompt, "the text is kept, braces and all")

"""Task suggestions: the deterministic layers and the model round-trip."""

from __future__ import annotations

import unittest

from hauhau import suggest


class CommandCompletionTestCase(unittest.TestCase):
    def test_a_started_command_is_completed(self):
        result = suggest.complete_command("/mod")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "/model")
        self.assertEqual(result.completion, "el")
        self.assertEqual(result.source, "command")

    def test_a_finished_command_has_nothing_to_ghost(self):
        self.assertIsNone(suggest.complete_command("/model"))

    def test_an_unknown_command_is_left_alone(self):
        self.assertIsNone(suggest.complete_command("/nope"))

    def test_a_command_with_arguments_is_not_completed(self):
        self.assertIsNone(suggest.complete_command("/model lf"))


class ContinuationTestCase(unittest.TestCase):
    def test_the_sentence_is_finished_from_the_last_word(self):
        result = suggest.continue_task("hur mycket disk")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "hur mycket disk är det kvar och vad tar mest plats?")
        self.assertEqual(result.source, "keyword")

    def test_the_completion_is_the_rest_of_the_line_only(self):
        result = suggest.continue_task("kolla loggen")
        self.assertEqual(result.completion, " och säg vad som faktiskt hände")

    def test_a_word_from_the_middle_does_not_trigger(self):
        # "mail" appears, but the sentence has moved past it.
        self.assertIsNone(suggest.continue_task("vilka mail ska jag svara på"))

    def test_a_finished_sentence_is_left_alone(self):
        self.assertIsNone(suggest.continue_task("är det klart?"))
        self.assertIsNone(suggest.continue_task("kör testerna."))

    def test_a_long_thought_does_not_get_padded(self):
        self.assertIsNone(suggest.continue_task("jag undrar vad som hände med den där filen egentligen"))

    def test_too_short_to_be_worth_it(self):
        self.assertIsNone(suggest.continue_task("di"))

    def test_a_slash_command_is_not_a_sentence(self):
        self.assertIsNone(suggest.continue_task("/status"))

    def test_a_multi_word_trigger_may_appear_anywhere(self):
        result = suggest.continue_task("vad vet du")
        self.assertIsNotNone(result)
        self.assertTrue(result.text.startswith("vad vet du om mig"))


class ShouldAskTestCase(unittest.TestCase):
    def test_short_input_is_not_worth_a_model_call(self):
        self.assertFalse(suggest.should_ask("hur my"))

    def test_a_typed_sentence_is(self):
        self.assertTrue(suggest.should_ask("sortera mina bilder efter datum"))

    def test_a_finished_sentence_is_not(self):
        self.assertFalse(suggest.should_ask("sortera mina bilder efter datum."))
        self.assertFalse(suggest.should_ask("vilka filer är störst?"))

    def test_commands_are_not_sent_to_the_model(self):
        self.assertFalse(suggest.should_ask("/tune 3"))


class ParseCompletionTestCase(unittest.TestCase):
    def test_quotes_are_stripped(self):
        self.assertEqual(suggest.parse_completion('" finns det kvar?"', "disk"), " finns det kvar?")

    def test_an_echo_of_the_prompt_is_removed(self):
        self.assertEqual(suggest.parse_completion("disk finns det kvar", "disk"), " finns det kvar")

    def test_a_pure_echo_is_rejected(self):
        self.assertIsNone(suggest.parse_completion("disk", "disk"))

    def test_an_empty_reply_is_rejected(self):
        self.assertIsNone(suggest.parse_completion("   ", "disk"))

    def test_only_the_first_line_is_used(self):
        self.assertEqual(suggest.parse_completion("är det kvar\nand more", "disk"), " är det kvar")

    def test_a_runaway_reply_is_clipped(self):
        reply = "x" * 400
        completion = suggest.parse_completion(reply, "disk")
        self.assertLessEqual(len(completion), suggest.MAX_COMPLETION_CHARS)
        self.assertTrue(completion.endswith("…"))


class Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeClient:
    def __init__(self, content: str = "", raises: bool = False) -> None:
        self.content = content
        self.raises = raises
        self.seen: list[dict] = []

    def chat(self, messages, tools, **kwargs):
        self.seen.append({"messages": messages, "tools": tools, **kwargs})
        if self.raises:
            raise RuntimeError("backend down")
        return Reply(self.content)


class ProposeTestCase(unittest.TestCase):
    def test_it_appends_the_completion_to_what_was_typed(self):
        client = FakeClient('" finns det kvar?"')
        result = suggest.propose(client, "disk")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "disk finns det kvar?")
        self.assertEqual(result.source, "ai")

    def test_a_broken_backend_returns_none_instead_of_raising(self):
        self.assertIsNone(suggest.propose(FakeClient(raises=True), "disk"))

    def test_the_request_is_small_and_greedy_free(self):
        client = FakeClient(" och sen")
        suggest.propose(client, "sortera bilder", tools=["shell", "read_file"])
        call = client.seen[0]
        self.assertLessEqual(call["max_tokens"], 64)
        self.assertLessEqual(call["temperature"], 0.5)
        prompt = call["messages"][0]["content"]
        self.assertIn("sortera bilder", prompt)
        self.assertIn("shell", prompt)


if __name__ == "__main__":
    unittest.main()

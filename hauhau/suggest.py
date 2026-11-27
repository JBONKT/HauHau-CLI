"""Task suggestions for the prompt line, without any UI dependency.

Two layers, cheapest first:

* **Continuations** - a table of the tasks this agent actually does well, keyed
  by the words that introduce them. Typing "hur mycket disk" finishes the
  sentence, because the suggestion is the *rest* of it, not a different task.
  This is instant and deterministic, so it costs nothing on every keystroke.
* **The model** - for anything the table does not recognise, the local model is
  asked to finish the sentence in a dozen words. It is throttled and gated (see
  :func:`should_ask`): a 2B model at ~30 tok/s cannot answer per keystroke, and
  a suggestion that arrives after you have typed past it is worse than none.

Both return only the *completion*, never a rewritten prompt: Textual's ghost
text is rendered after the caret, so a suggestion that does not literally extend
what was typed cannot be shown - and would fight the user's own wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# How short a value may be before it is worth a model round-trip. Below this the
# completion is guesswork that the user has already out-typed.
MIN_AI_CHARS = 10
# The longest completion the model may return; the ghost line is one line.
MAX_COMPLETION_CHARS = 90

# (triggers, completion, why). Every completion starts with a space and is a
# whole phrase: it is appended after the user's words, so it must not glue itself
# onto the end of a word ("vilka mail" + "en jag inte..." reads as "vilka mailen").
CONTINUATIONS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("disk", "plats", "utrymme", "lagring"), " är det kvar och vad tar mest plats?", "disk"),
    (("mail", "mejl", "inkorg"), " jag inte svarat på den här veckan?", "mail"),
    (("möte", "möten", "kalender", "agenda", "bokat"), " har jag framöver och vad borde jag läsa innan?", "kalender"),
    (("påminn", "reminder"), " har jag den här veckan som jag missat?", "påminnelser"),
    (("sammanfatta", "summera", "sammanfattning"), " vad som hänt i veckan och vad som väntar härnäst", "sammanfatta"),
    (("vad vet du", "kommer du ihåg", "minns du"), " om mig och hur jag brukar vilja ha saker?", "minne"),
    (("git", "commit", "branch"), " — visa mig senaste commiten och vad den ändrade", "git"),
    (("test", "pytest"), " och fixa det som fallerar", "tester"),
    (("logg", "krasch", "krash", "fel"), " och säg vad som faktiskt hände", "logg"),
    (("varför",), " — ta reda på det med en enda avgörande kontroll", "felsök"),
    (("hitta", "finns", "var ligger"), " — sök i filsystemet och visa var", "filer"),
    (("skriv", "skapa", "bygg"), " — och verifiera att resultatet fungerar", "bygg"),
    (("översätt", "förklara", "sammanfatta den"), " det kort och konkret", "text"),
    (("hej", "morgon", "god morgon"), " — vad har jag framför mig idag?", "brief"),
)

# Slash commands worth completing, grouped by what the user is likely to want.
COMMAND_HINTS: tuple[tuple[str, str], ...] = (
    ("/status", "model, context, memory, session"),
    ("/model", "list local models, or switch"),
    ("/mem", "what the agent remembers"),
    ("/learn", "distil this session into memory"),
    ("/sync", "pull mail, calendar, messages into the journal"),
    ("/journal", "ask the journal directly"),
    ("/brief", "the morning brief"),
    ("/tune", "hill-climb the server flags"),
    ("/eval", "score tool choice against the suite"),
    ("/evolve", "rewrite the prompt, keep it only if it scores"),
    ("/perf", "measure real tokens/second"),
    ("/context", "what is in the window right now"),
    ("/compact", "summarise the conversation"),
    ("/audit", "every command and self-edit"),
    ("/help", "all commands"),
)


@dataclass(frozen=True)
class Suggestion:
    """A ghost-text completion plus where it came from."""

    text: str  # the whole line: typed value + completion
    completion: str  # only what would be appended
    source: str  # "command" | "keyword" | "ai"

    @property
    def label(self) -> str:
        return self.source


def _words(value: str) -> str:
    return value.casefold()


def complete_command(value: str) -> Suggestion | None:
    """Finish a ``/command`` the user has started typing."""
    stem = value.strip()
    if not stem.startswith("/") or " " in stem:
        return None
    matches = [pair for pair in COMMAND_HINTS if pair[0].startswith(stem.casefold())]
    if not matches:
        return None
    name = matches[0][0]
    if name == stem:
        return None  # already complete: nothing to ghost
    return Suggestion(text=name, completion=name[len(stem) :], source="command")


def continue_task(value: str) -> Suggestion | None:
    """Finish a sentence the user has started, from the continuation table.

    Only the *last word* may trigger a single-word entry: a completion is glued
    to the end of the line, so matching a word from the middle would append a
    clause to a sentence that has already moved past it ("vilka mail ska jag
    svara på" must not become "vilka mail ska jag svara på jag inte svarat på").
    """
    text = value.rstrip()
    if len(text) < 3 or text.startswith("/"):
        return None
    if text.endswith(("?", ".", "!", ":", ",")):
        return None
    words = text.split()
    if len(words) > 5:
        return None  # already a full thought; a completion would only pad it
    last = words[-1].casefold()
    lowered = text.casefold()
    for triggers, completion, _why in CONTINUATIONS:
        for trigger in triggers:
            matched = trigger in lowered if " " in trigger else trigger in last
            if not matched:
                continue
            rest = completion if completion.startswith(" ") else " " + completion
            return Suggestion(text=text + rest, completion=rest, source="keyword")
    return None


def should_ask(value: str, *, min_chars: int = MIN_AI_CHARS) -> bool:
    """Whether a model round-trip is worth it for this prefix."""
    text = value.strip()
    if len(text) < min_chars:
        return False
    if text.startswith("/"):
        return False
    # Already a whole sentence ending in punctuation: the user is done, not stuck.
    return not text.endswith((".", "?", "!", ":"))


def build_prompt(value: str, *, tools: list[str] | None = None) -> str:
    """The instruction sent to the local model for a completion."""
    known = ", ".join(tools or [])
    return (
        "You are completing a half-typed request to a local CLI agent that has a "
        "shell, a filesystem, a memory and a journal of the user's mail, calendar "
        "and messages.\n"
        f"The agent's tools: {known or 'shell, files, memory, journal'}.\n"
        "The user has typed this much:\n"
        f"<typed>{value}</typed>\n"
        "Finish their sentence in at most 12 words. Answer with ONLY the "
        "continuation - no quotes, no explanation, no repetition of what they "
        "already typed. It must read as the natural end of their sentence."
    )


_STRIP = re.compile(r"^[\s\"'`»«*-]+|[\s\"'`«»]+$")


def parse_completion(reply: str, value: str) -> str | None:
    """Turn a model reply into a completion, or ``None`` if it is unusable."""
    line = (reply or "").strip().splitlines()
    if not line:
        return None
    text = _STRIP.sub("", line[0].strip())
    # A model that echoes the prompt would double it in the ghost line.
    if not text or text.casefold() == value.strip().casefold():
        return None
    if text.casefold().startswith(value.strip().casefold()):
        text = text[len(value.strip()) :].lstrip()
    if not text:
        return None
    if text[0] not in " .,;:—-":
        text = " " + text
    # Clipped last, so the leading space and the ellipsis both fit the budget.
    if len(text) > MAX_COMPLETION_CHARS:
        text = text[: MAX_COMPLETION_CHARS - 1].rstrip() + "…"
    return text


def propose(client, value: str, *, tools: list[str] | None = None) -> Suggestion | None:
    """Ask the local model to finish the sentence. Blocking; call off the loop."""
    prompt = build_prompt(value, tools=tools)
    messages = [{"role": "user", "content": prompt}]
    try:
        reply = client.chat(
            messages,
            None,
            think=False,
            max_tokens=48,
            temperature=0.3,
        )
    except Exception:
        return None  # a suggestion must never break the prompt line
    completion = parse_completion(getattr(reply, "content", "") or "", value)
    if not completion:
        return None
    return Suggestion(text=value.rstrip() + completion, completion=completion, source="ai")

"""Interactive REPL."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from rich.box import ROUNDED
from rich.panel import Panel
from rich.text import Text

from . import config as config_mod
from .agent import Agent
from .audit import Audit
from .brain import Brain
from .brain.learn import Learner, worth_distilling
from .evolve import (
    DEFAULT_TASKS,
    PromptStore,
    available,
    benchmark,
    evolve_prompt,
    hill_climb,
    run_suite,
)
from .evolve import (
    candidates as tune_candidates,
)
from .evolve import (
    record as record_tuning,
)
from .evolve.tune import set_value
from .llm import LLMError
from .profiler import apply_recommendations, probe, recommend
from .server import ServerManager
from .tools import build_registry
from .ui import (
    ACCENT,
    COMMANDS,
    ERR,
    MUTED,
    OK,
    WARN,
    Progress,
    RichUI,
    SlashCompleter,
    banner,
    data_table,
    error_panel,
    help_panel,
    kv_table,
    make_console,
    make_prompt_session,
    nearest_command,
    note,
    prompt_glyph,
    short_name,
    toolbar_text,
)
from .util import clip_text, estimate_messages_tokens, human_bytes, snapshot_file


def apply_active_prompt(cfg, prompts: PromptStore) -> int | None:
    """Adopt the stored prompt version, if there is one. Returns its number.

    The version store wins over ``config.toml``: the config keeps the text that
    was last applied for transparency, but only the store has the history and
    the rollback.
    """
    active = prompts.active()
    if active is None:
        return None
    try:
        stored = prompts.text(active.number)
    except FileNotFoundError:
        return None
    if not stored:
        return None
    cfg.agent.personality = stored
    return active.number


class Repl:
    def __init__(
        self, cfg, client, manager, brain, learner, agent, audit, ui, session_path,
        prompts: PromptStore | None = None, profile=None,
    ) -> None:
        self.cfg = cfg
        self.prompts = prompts or PromptStore(cfg.data_dir / "evolve" / "prompts")
        self.profile = profile
        self.client = client
        self.manager = manager
        self.brain = brain
        self.learner = learner
        self.agent = agent
        self.audit = audit
        self.ui = ui
        self.session_path = session_path
        self.console = ui.console
        self.running = True

    # --- helpers -------------------------------------------------------
    def _table(self, title: str, rows: list[tuple[str, str]]) -> None:
        kv_table(self.console, title, rows)

    # --- commands ------------------------------------------------------
    def handle(self, line: str) -> None:
        parts = line.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        handler = {
            "/help": self.cmd_help,
            "/status": self.cmd_status,
            "/show": self.cmd_show,
            "/?": self.cmd_help,
            "/tools": self.cmd_tools,
            "/mem": self.cmd_mem,
            "/compact": self.cmd_compact,
            "/prune": self.cmd_prune,
            "/remember": self.cmd_remember,
            "/forget": self.cmd_forget,
            "/learn": self.cmd_learn,
            "/eval": self.cmd_eval,
            "/tune": self.cmd_tune,
            "/evolve": self.cmd_evolve,
            "/prompt": self.cmd_prompt,
            "/model": self.cmd_model,
            "/perf": self.cmd_perf,
            "/config": self.cmd_config,
            "/audit": self.cmd_audit,
            "/context": self.cmd_context,
            "/clear": self.cmd_clear,
            "/save": self.cmd_save,
            "/quit": self.cmd_quit,
            "/exit": self.cmd_quit,
        }.get(command)
        if handler is None:
            hint = nearest_command(command)
            tail = f" — did you mean {hint}?" if hint else " — /help lists them"
            error_panel(self.console, f"unknown command {command}{tail}")
            return
        handler(arg)

    def cmd_help(self, arg: str) -> None:
        help_panel(self.console, arg)

    def cmd_status(self, _arg: str) -> None:
        """One screen, so a user does not have to ask four questions."""
        status = self.manager.status()
        stats = self.brain.stats() if self.brain else None
        active = self.prompts.active_number()
        rows = [
            ("model", short_name(os.path.basename(os.path.expanduser(status["gguf"]))) if status["gguf"] else "(none)"),
            ("backend", f"{status['url']} · {'managed' if status['managed'] else 'external'}"),
            ("context", f"{status['n_ctx']} tokens · {status['n_threads']} threads"),
            (
                "memory",
                f"{stats['total']} memories · vectors {'on' if stats['embeddings_enabled'] else 'off'}"
                if stats
                else "off (--no-brain)",
            ),
            ("prompt", f"v{active} (evolved)" if active else "from config.toml"),
            ("workspace", str(self.cfg.workspace)),
            (
                "session",
                f"{self.session_path.name} · {self.agent.turn_count()} turns · "
                f"{'saved' if self.session_path.is_file() else 'not saved yet'}",
            ),
            ("config", str(self.cfg.source) if self.cfg.source else "(built-in defaults)"),
        ]
        if self.agent.last_completion and self.agent.last_completion.usage.tok_per_second:
            rows.append(("last speed", f"{self.agent.last_completion.usage.tok_per_second:.1f} tok/s"))
        self._table("Status", rows)

    def cmd_show(self, arg: str) -> None:
        """Replay a tool result in full: the one-line preview loses error detail."""
        collect = getattr(self.ui, "recent_results", None)
        entries = collect() if callable(collect) else []
        if not entries:
            note(self.console, "no tool output in this session yet")
            return
        try:
            index = int(arg) if arg.strip() else 1
        except ValueError:
            note(self.console, "usage: /show [n] — 1 is the most recent", WARN)
            return
        if not 1 <= index <= len(entries):
            note(self.console, f"only {len(entries)} tool results are kept; /show 1 is the latest", WARN)
            return
        name, result = entries[index - 1]
        body = clip_text(result.content or "(no output)", self.cfg.tools.max_output_chars)
        heading = f"{name} {'✓' if result.ok else '✗'} · {index} of {len(entries)}"
        self.console.print(
            Panel(
                Text(body, style="" if result.ok else ERR),
                title=Text(heading, style=f"bold {ACCENT if result.ok else ERR}"),
                title_align="left",
                border_style=MUTED if result.ok else ERR,
                box=ROUNDED,
                expand=False,
            )
        )

    def cmd_tools(self, _arg: str) -> None:
        data_table(
            self.console,
            "Tools",
            ["name", "description"],
            [[t.name, t.description.splitlines()[0]] for t in self.agent.registry.tools()],
        )

    def cmd_mem(self, arg: str) -> None:
        if self.brain is None:
            note(self.console, "memory is disabled (--no-brain)", WARN)
            return
        if arg.strip():
            hits = self.brain.recall(arg.strip(), k=10)
            if not hits:
                note(self.console, f"nothing remembered about {arg.strip()!r}")
                return
            data_table(
                self.console,
                f"memory: {arg.strip()}",
                ["id", "kind", "score", "text"],
                [[str(h.id), h.kind, f"{h.score:.2f}", h.text] for h in hits],
                no_wrap=3,
            )
            return
        stats = self.brain.stats()
        rows = [
            ("total", str(stats["total"])),
            ("by kind", ", ".join(f"{k}={v}" for k, v in stats["by_kind"].items()) or "-"),
            ("with vectors", str(stats["with_embeddings"])),
            ("database", f"{stats['db']} ({human_bytes(stats['db_bytes'])})"),
        ]
        if stats.get("embeddings_error"):
            rows.append(("vector error", stats["embeddings_error"][:120]))
        self._table("Memory", rows)

    def cmd_compact(self, _arg: str) -> None:
        if self.brain is None:
            note(self.console, "memory is disabled (--no-brain)", WARN)
            return
        before = self.brain.stats()["total"]
        if before > 300:
            with Progress(self.console, f"comparing {before} memories…"):
                absorbed = self.brain.store.compact()
        else:
            absorbed = self.brain.store.compact()
        after = self.brain.stats()["total"]
        note(self.console, f"merged {absorbed} duplicate memories ({before} → {after})", OK)

    def cmd_prune(self, arg: str) -> None:
        if self.brain is None:
            note(self.console, "memory is disabled (--no-brain)", WARN)
            return
        try:
            min_weight = float(arg) if arg.strip() else 0.02
        except ValueError:
            note(self.console, "usage: /prune [min_weight]", WARN)
            return
        before = self.brain.stats()["total"]
        removed = self.brain.store.prune(min_weight=min_weight)
        after = self.brain.stats()["total"]
        note(
            self.console,
            f"pruned {removed} stale memories ({before} → {after})",
            OK if removed else WARN,
        )

    def cmd_remember(self, arg: str) -> None:
        if self.brain is None or not arg.strip():
            note(self.console, "usage: /remember <text>", WARN)
            return
        row_id, created = self.brain.remember("semantic", arg.strip(), "", 0.7)
        verb = "stored" if created else "merged into"
        note(self.console, f"{verb} memory #{row_id}", OK)

    def cmd_forget(self, arg: str) -> None:
        if self.brain is None:
            note(self.console, "memory is disabled (--no-brain)", WARN)
            return
        try:
            row_id = int(arg.strip())
        except ValueError:
            note(self.console, "usage: /forget <id>", WARN)
            return
        removed = self.brain.store.forget(row_id)
        note(self.console, "forgotten" if removed else "no such memory", OK if removed else WARN)

    def cmd_learn(self, _arg: str) -> None:
        if self.brain is None:
            note(self.console, "memory is disabled (--no-brain)", WARN)
            return
        if not self.agent.messages:
            note(self.console, "nothing to learn from yet")
            return
        if not worth_distilling(self.agent.messages):
            note(self.console, "this session is too thin to learn from (one short turn, no tool use)", WARN)
            return
        with Progress(self.console, "distilling this session…") as progress:
            try:
                drafts, written = self.learner.distill(self.agent.messages)
            except LLMError as exc:
                error_panel(self.console, f"learn failed: {exc}")
                return
            progress.stop()
        if not drafts:
            note(self.console, "nothing durable found")
            return
        for draft, (row_id, created) in zip(drafts, written, strict=False):
            mark = "new" if created else "merged"
            self.console.print(Text(f"  [{draft.kind}/{mark} #{row_id}] {draft.text}", style="green"), markup=False)

    # --- auto-tuning and prompt evolution -------------------------------
    def _eval_tasks(self):
        return available(DEFAULT_TASKS, self.agent.registry.names())

    def _eval_prompt_text(self, text: str) -> str:
        """The personality is the part under test, not the per-session context."""
        return text.format(name=self.cfg.agent.name)

    def _save_config(self, what: str) -> Path:
        """Snapshot the config, then write it. Never edit it behind the user's back."""
        target = self.cfg.source or (self.cfg.data_dir / "config.toml")
        copy = snapshot_file(target, self.cfg.snapshots_dir)
        if copy:
            self.audit.log("snapshot", path=str(target), copy=str(copy), why=what)
        return config_mod.save(self.cfg, target)

    def cmd_eval(self, _arg: str) -> None:
        tasks = self._eval_tasks()
        if not tasks:
            note(self.console, "no eval tasks apply to the current tool set", WARN)
            return
        seen = {"n": 0}

        with Progress(self.console, f"playing {len(tasks)} eval tasks…") as progress:

            def on_task(name: str) -> None:
                seen["n"] += 1
                progress.update(f"task {seen['n']}/{len(tasks)}: {name}")

            try:
                result = run_suite(
                    self.client,
                    tasks,
                    system_prompt=self._eval_prompt_text(self.cfg.agent.personality),
                    tools=self.agent.registry.schemas(),
                    progress=on_task,
                )
            except LLMError as exc:
                error_panel(self.console, f"eval failed: {exc}")
                return
            progress.step(f"scored {result.summary()} in {result.seconds:.1f}s")
        self.audit.log("eval", score=round(result.score, 3), passed=result.passed, total=result.total)
        self._table("Eval", [("score", result.summary()), ("seconds", f"{result.seconds:.1f}")])
        if result.failures:
            data_table(
                self.console,
                "Failed tasks",
                ["task", "why"],
                [[name, detail] for name, detail in result.failures],
                no_wrap=1,
            )

    def cmd_tune(self, arg: str) -> None:
        if not self.cfg.model.manage_server:
            note(
                self.console,
                "the backend is external (manage_server = false); the tuner cannot restart it",
                WARN,
            )
            return
        try:
            budget = int(arg) if arg.strip() else 4
        except ValueError:
            note(self.console, "usage: /tune [max_trials]", WARN)
            return
        cores = (self.profile.cores if self.profile else 0) or os.cpu_count() or 4
        options = tune_candidates(self.cfg, cores)
        if not options:
            note(self.console, "nothing to try: the config already has the candidate values")
            return
        with Progress(self.console, "benchmarking the current config…") as progress:
            try:
                baseline = benchmark(self.client)
            except LLMError as exc:
                error_panel(self.console, f"benchmark failed: {exc}")
                return
            progress.step(f"baseline {baseline:.1f} tok/s · trying up to {budget} values")
            original = {
                key: getattr(self.cfg.model, key.split(".", 1)[1]) for key in ("model.n_threads", "model.n_ctx")
            }
            self.audit.log("tune.start", baseline=round(baseline, 2), trials=budget)

            def measure(option) -> float:
                progress.update(f"restarting for {option}…")
                set_value(self.cfg, option.key, option.value)
                self.audit.log("tune.trial", setting=option.key, value=option.value)
                self.manager.restart()
                progress.update(f"benchmarking {option}…")
                return benchmark(self.client)

            def on_trial(trial) -> None:
                progress.step(f"  {trial.candidate} → {trial.score:.1f} tok/s")

            try:
                result = hill_climb(baseline, options, measure, max_trials=budget, on_trial=on_trial)
            except LLMError as exc:
                error_panel(self.console, f"tuning aborted: {exc}")
                result = None
            finally:
                # Whatever happened, the running server must match the chosen values.
                for key, value in original.items():
                    set_value(self.cfg, key, value)
                if result is not None:
                    for chosen in result.adopted:
                        set_value(self.cfg, chosen.key, chosen.value)
                try:
                    self.manager.restart()
                except LLMError as exc:
                    # Tuning must never take the session down with it.
                    note(self.console, f"backend restart failed: {exc} - run /model to check it", WARN)

        if result is None:
            return
        rows = [
            (trial.candidate, f"{trial.score:.1f} tok/s", "✓ adopted" if trial.adopted else "kept")
            for trial in result.trials
        ]
        if rows:
            data_table(self.console, "Tuning trials", ["setting", "speed", "result"], rows, no_wrap=2)
        if result.adopted:
            target = self._save_config("tune")
            record_tuning(self.cfg.data_dir / "evolve" / "tuning.jsonl", result, adopted_config=str(target))
            self.audit.log("tune.adopt", best=round(result.best, 2), settings=[str(c) for c in result.adopted])
            note(
                self.console,
                f"adopted {', '.join(str(c) for c in result.adopted)}: "
                f"{baseline:.1f} → {result.best:.1f} tok/s, saved to {target}",
                OK,
            )
        else:
            note(
                self.console,
                f"kept the current settings ({baseline:.1f} tok/s). {result.stopped or 'no candidate was faster'}",
                WARN,
            )

    def _apply_prompt(self, text: str, why: str) -> None:
        self.cfg.agent.personality = text
        target = self._save_config(why)
        self.audit.log("prompt.apply", why=why, config=str(target), chars=len(text))

    def cmd_evolve(self, _arg: str) -> None:
        tasks = self._eval_tasks()
        if not tasks:
            note(self.console, "no eval tasks apply to the current tool set", WARN)
            return
        store = self.prompts
        if not store.versions():
            seed = store.add(self.cfg.agent.personality, score=None)
            store.activate(seed.number)
            note(self.console, f"stored the current prompt as v{seed.number} (rollback point)")
        schemas = self.agent.registry.schemas()

        def run(text: str):
            return run_suite(
                self.client, tasks, system_prompt=self._eval_prompt_text(text), tools=schemas
            )

        with Progress(self.console, "scoring the current prompt…") as progress:
            try:
                result = evolve_prompt(
                    self.client,
                    store,
                    run,
                    self.cfg.agent.personality,
                    progress=progress.update,
                )
            except LLMError as exc:
                error_panel(self.console, f"evolution failed: {exc}")
                return
            progress.step(f"current {result.baseline:.0%} · candidate {result.candidate:.0%}")
        self.audit.log(
            "evolve.prompt",
            accepted=result.accepted,
            baseline=round(result.baseline, 3),
            candidate=round(result.candidate, 3),
            version=result.version,
        )
        if result.accepted:
            self._apply_prompt(result.text, "evolve")
            note(self.console, f"{result.reason}; the new prompt is active", OK)
        else:
            note(self.console, f"kept the current prompt: {result.reason}", WARN)

    def cmd_prompt(self, arg: str) -> None:
        store = self.prompts
        versions = store.versions()
        if not versions:
            note(self.console, "no prompt versions yet — run /evolve to store v1", WARN)
            return
        active = store.active_number()
        if not arg.strip():
            data_table(
                self.console,
                "Prompt versions",
                ["version", "score", "parent", "chars", "active"],
                [
                    [
                        v.label,
                        "-" if v.score is None else f"{v.score:.0%}",
                        "-" if v.parent is None else f"v{v.parent}",
                        str(len(store.text(v.number))),
                        "● active" if v.number == active else "",
                    ]
                    for v in versions
                ],
                no_wrap=1,
            )
            return
        try:
            number = int(arg.strip().lstrip("vV"))
        except ValueError:
            note(self.console, "usage: /prompt [vN]", WARN)
            return
        try:
            version = store.activate(number)
            text = store.text(number)
        except FileNotFoundError as exc:
            note(self.console, str(exc), WARN)
            return
        self._apply_prompt(text, f"prompt v{version.number}")
        note(self.console, f"activated prompt v{version.number} ({len(text)} chars)", OK)

    def cmd_model(self, _arg: str) -> None:
        status = self.manager.status()
        rows = [
            ("model", status["gguf"] or "(none)"),
            ("server", f"{status['url']} ({'managed' if status['managed'] else 'external'})"),
            ("context", f"{status['n_ctx']} tokens"),
            ("threads", str(status["n_threads"])),
            ("gpu layers", str(status["n_gpu_layers"])),
            ("think", str(self.cfg.generation.think)),
        ]
        if self.agent.last_completion and self.agent.last_completion.usage.tok_per_second:
            rows.append(("last speed", f"{self.agent.last_completion.usage.tok_per_second:.1f} tok/s"))
        self._table("Backend", rows)

    def cmd_perf(self, _arg: str) -> None:
        note(self.console, "benchmarking…")
        try:
            started = time.monotonic()
            warm = self.client.chat(
                [{"role": "user", "content": "Reply with exactly: ok"}],
                None,
                think=False,
                max_tokens=16,
                temperature=0.0,
            )
            latency = time.monotonic() - started
            run = self.client.chat(
                [{"role": "user", "content": "Write the numbers 1 to 40 separated by commas."}],
                None,
                think=False,
                max_tokens=200,
                temperature=0.0,
            )
        except LLMError as exc:
            error_panel(self.console, f"benchmark failed: {exc}")
            return
        self._table(
            "Performance",
            [
                ("short-call latency", f"{latency:.2f}s"),
                ("generation", f"{run.usage.tok_per_second:.1f} tok/s"),
                ("prompt eval", f"{run.usage.prompt_per_second:.1f} tok/s"),
                ("tokens", f"{run.usage.completion_tokens} completion"),
                ("sample", (run.content or warm.content)[:60].replace("\n", " ")),
            ],
        )

    def cmd_config(self, _arg: str) -> None:
        self._table(
            "Config",
            [
                ("file", str(self.cfg.source) if self.cfg.source else "(built-in defaults)"),
                ("gguf", self.cfg.model.gguf or "(none)"),
                ("data dir", str(self.cfg.data_dir)),
                ("workspace", str(self.cfg.workspace)),
                ("shell confirm", str(self.cfg.tools.shell_confirm)),
                ("denylist", ", ".join(self.cfg.tools.denylist) or "(empty - everything allowed)"),
                ("max steps", str(self.cfg.agent.max_steps)),
                ("context budget", f"{self.cfg.agent.context_budget_tokens} tokens"),
                ("brain", "on" if self.cfg.brain.enabled else "off"),
            ],
        )

    def cmd_audit(self, arg: str) -> None:
        try:
            count = int(arg) if arg.strip() else 10
        except ValueError:
            count = 10
        entries = self.audit.tail(count)
        if not entries:
            note(self.console, "audit log is empty")
            return
        for entry in entries:
            when = str(entry.get("ts", ""))[11:19]
            event = entry.get("event", "?")
            detail = {k: v for k, v in entry.items() if k not in ("ts", "session", "event")}
            self.console.print(Text(f"{when} {event} {detail}", style="dim"), soft_wrap=True, markup=False)

    def cmd_context(self, _arg: str) -> None:
        used = estimate_messages_tokens(self.agent.messages)
        self._table(
            "Context",
            [
                ("messages", str(len(self.agent.messages))),
                ("turns", str(self.agent.turn_count())),
                ("estimated tokens", f"{used} / {self.cfg.agent.context_budget_tokens}"),
                ("summarised", "yes" if self.agent.summary else "no"),
            ],
        )

    def cmd_clear(self, _arg: str) -> None:
        self.agent.reset()
        note(self.console, "conversation cleared", OK)

    def cmd_save(self, _arg: str) -> None:
        self.agent.persist(self.session_path)
        note(self.console, f"saved to {self.session_path}", OK)

    def cmd_quit(self, _arg: str) -> None:
        self.running = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hauhau",
        description="Conversational, self-learning CLI agent on a local Gemma GGUF.",
    )
    parser.add_argument("prompt", nargs="*", help="run a single turn and exit")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--model", help="path to a .gguf file (overrides config)")
    parser.add_argument("--server-url", help="reuse an existing llama-server here")
    parser.add_argument("--no-server", action="store_true", help="never start llama-server ourselves")
    parser.add_argument("--no-brain", action="store_true", help="disable persistent memory")
    parser.add_argument("--confirm", action="store_true", help="ask before every shell command")
    parser.add_argument("--think", action="store_true", help="enable the model's reasoning output")
    parser.add_argument("--data-dir", help="where memory, logs and sessions live")
    parser.add_argument("--max-steps", type=int, help="max tool steps per turn")
    parser.add_argument("--profile", action="store_true", help="print hardware profile and exit")
    parser.add_argument("--resume", nargs="?", const="last", help="resume the last (or named) session")
    parser.add_argument(
        "--yolo", action="store_true", help="accepted for compatibility; auto-run is already the default"
    )
    return parser


def _apply_args(cfg, args) -> None:
    if args.model:
        cfg.model.gguf = os.path.expanduser(args.model)
    if args.server_url:
        cfg.model.server_url = args.server_url
        cfg.model.manage_server = False
    if args.no_server:
        cfg.model.manage_server = False
    if args.no_brain:
        cfg.brain.enabled = False
    if args.confirm:
        cfg.tools.shell_confirm = True
    if args.think:
        cfg.generation.think = True
    if args.data_dir:
        cfg.logging.data_dir = args.data_dir
    if args.max_steps:
        cfg.agent.max_steps = args.max_steps


def _session_path(cfg, resume: str | None) -> tuple[Path, str | None]:
    """Resolve the session file, returning (path, warning).

    An unknown --resume id must not become a new file named after the id: that
    wrote an extensionless stray into the sessions directory.
    """
    cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
    fresh = cfg.sessions_dir / f"{time.strftime('%Y%m%dT%H%M%S')}.json"
    if not resume:
        return fresh, None
    if resume == "last":
        existing = sorted(cfg.sessions_dir.glob("*.json"))
        return (existing[-1], None) if existing else (fresh, None)
    candidate = cfg.sessions_dir / (resume if resume.endswith(".json") else f"{resume}.json")
    if candidate.is_file():
        return candidate, None
    return fresh, f"no session named {resume!r} - starting a new one"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = make_console()
    ui = RichUI(console)

    try:
        cfg, warnings = config_mod.load(args.config)
    except FileNotFoundError as exc:
        error_panel(console, str(exc))
        return 2
    for warning in warnings:
        note(console, f"config: {warning}", WARN)
    _apply_args(cfg, args)

    profile = probe()
    if args.profile:
        kv_table(
            console,
            "Hardware",
            [(key, str(value)) for key, value in profile.as_dict().items() if key != "notes"],
        )
        size = 0
        if cfg.model.gguf:
            gguf_path = os.path.expanduser(cfg.model.gguf)
            size = os.path.getsize(gguf_path) if os.path.isfile(gguf_path) else 0
        for name, (value, reason) in recommend(profile, size).items():
            note(console, f"  {name} = {value}: {reason}")
        return 0

    if cfg.source is None:
        changes = apply_recommendations(cfg, profile)
        if changes:
            note(console, "auto-tuned for this machine:")
            for change in changes:
                note(console, f"  {change}")

    session_path, session_warning = _session_path(cfg, args.resume)
    session_id = session_path.stem
    audit = Audit(cfg.audit_path, session_id, cfg.logging.audit)
    manager = ServerManager(cfg, audit)

    try:
        client = manager.ensure()
    except LLMError as exc:
        error_panel(console, str(exc))
        audit.close()
        manager.stop()
        return 1

    brain = None
    if cfg.brain.enabled:
        try:
            brain = Brain(cfg, audit)
        except Exception as exc:  # memory must never block startup
            note(console, f"memory disabled: {exc}", WARN)

    def confirm(cmd: str) -> bool:
        try:
            answer = pt_prompt(f"run? {cmd[:120]} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")

    def revive_backend() -> bool:
        """Make sure a backend exists after an interrupt or a failed turn."""
        if not cfg.model.manage_server:
            return False
        try:
            manager.ensure()
            return True
        except LLMError:
            return False

    prompts = PromptStore(cfg.data_dir / "evolve" / "prompts")
    adopted = apply_active_prompt(cfg, prompts)
    if adopted is not None and console.is_terminal:
        note(console, f"system prompt v{adopted} active")

    registry = build_registry(cfg, audit, brain, confirm if cfg.tools.shell_confirm else None)
    agent = Agent(cfg, client, registry, brain, audit, ui)
    learner = Learner(cfg, client, brain, audit)
    if session_warning:
        note(console, session_warning, WARN)
    if args.resume and session_path.exists() and agent.load(session_path):
        note(console, f"resumed {agent.turn_count()} turns from {session_path.name}")

    # --- banner --------------------------------------------------------
    stats = brain.stats() if brain else {"total": 0}
    if console.is_terminal:  # a boxed banner is noise when piped
        banner(
            console,
            name=cfg.agent.name,
            model=os.path.basename(os.path.expanduser(cfg.model.gguf)),
            ctx=cfg.model.n_ctx,
            threads=cfg.model.n_threads,
            ram=profile.ram_gb,
            memories=stats["total"],
            url=cfg.model.server_url,
            workspace=str(cfg.workspace),
            prompt=f"v{adopted}" if adopted else "",
        )

    repl = Repl(
        cfg, client, manager, brain, learner, agent, audit, ui, session_path,
        prompts=prompts, profile=profile,
    )

    def finish() -> None:
        """Learn from the session, then shut down - in that order, always.

        The shutdown lives in ``finally`` because a failure in the learning pass
        or in the session save used to skip ``manager.stop()`` and leave a
        llama-server running after the CLI had exited.
        """
        try:
            if brain is not None and agent.messages and worth_distilling(agent.messages):
                if console.is_terminal:
                    note(console, "distilling this session into memory…")
                try:
                    _, written = learner.distill(agent.messages)
                    if console.is_terminal:
                        if written:
                            new = sum(1 for _, created in written if created)
                            note(
                                console,
                                f"learned {len(written)} memories ({new} new, {len(written) - new} merged)",
                            )
                        else:
                            note(console, "nothing durable to keep from this session")
                except KeyboardInterrupt:  # Ctrl-C during the pass: skip it
                    if console.is_terminal:
                        note(console, "skipped learning this session", WARN)
                except Exception as exc:  # learning must never block shutdown
                    audit.log("learn.error", error=f"{exc.__class__.__name__}: {exc}")
            if agent.messages:
                try:
                    agent.persist(session_path)
                except OSError as exc:
                    if console.is_terminal:
                        note(console, f"could not save the session: {exc}", WARN)
        finally:
            audit.close()
            manager.stop()

    # --- one-shot ------------------------------------------------------
    if args.prompt:
        try:
            agent.run_turn(" ".join(args.prompt))
            ui.newline()
        except LLMError as exc:
            error_panel(console, str(exc))
        finally:
            finish()
        return 0

    # --- interactive ---------------------------------------------------
    history_file = cfg.data_dir / "history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    def toolbar() -> str:
        active_version = prompts.active_number()
        return toolbar_text(
            ctx=cfg.model.n_ctx,
            memories=brain.stats()["total"] if brain else 0,
            ram=profile.ram_gb,
            shell_mode="confirm" if cfg.tools.shell_confirm else "auto",
            prompt=f"v{active_version}" if active_version else "",
        )

    def completions(text: str) -> list[tuple[str, str]]:
        """Argument completion for the commands whose arguments the CLI knows."""
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        if command == "/help":
            # No leading slash: completing "ev" must yield "/help evolve", not
            # "/help /evolve". cmd_help normalises either form.
            return [(name.lstrip("/"), description) for name, description in COMMANDS]
        if command == "/prompt":
            rows = []
            for version in prompts.versions():
                score = f"{version.score:.0%}" if version.score is not None else "baseline"
                rows.append((f"v{version.number}", score))
            return rows
        if command in ("/show", "/audit") and not argument:
            return [("1", "most recent"), ("2", "one back"), ("3", "two back")]
        return []

    session = make_prompt_session(history_file, toolbar, SlashCompleter(completions))
    try:
        while repl.running:
            try:
                line = session.prompt(prompt_glyph())
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                repl.handle(text)
                continue
            try:
                agent.run_turn(text)
                ui.newline()
            except LLMError as exc:
                ui.interrupt()
                error_panel(console, str(exc))
                if revive_backend():
                    note(console, "backend restarted - ask again", WARN)
            except KeyboardInterrupt:
                ui.interrupt()
                note(console, "interrupted", WARN)
                revive_backend()
    finally:
        finish()
        if console.is_terminal:
            note(console, "bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())

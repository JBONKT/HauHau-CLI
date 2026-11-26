"""Interactive REPL."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from rich.text import Text

from . import config as config_mod
from .agent import Agent
from .audit import Audit
from .brain import Brain
from .brain.learn import Learner
from .llm import LLMError
from .profiler import apply_recommendations, probe, recommend
from .server import ServerManager
from .tools import build_registry
from .ui import (
    OK,
    WARN,
    RichUI,
    banner,
    data_table,
    error_panel,
    help_panel,
    kv_table,
    make_console,
    make_prompt_session,
    note,
    prompt_glyph,
    toolbar_text,
)
from .util import estimate_messages_tokens, human_bytes


class Repl:
    def __init__(self, cfg, client, manager, brain, learner, agent, audit, ui, session_path) -> None:
        self.cfg = cfg
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
            "/?": self.cmd_help,
            "/tools": self.cmd_tools,
            "/mem": self.cmd_mem,
            "/compact": self.cmd_compact,
            "/remember": self.cmd_remember,
            "/forget": self.cmd_forget,
            "/learn": self.cmd_learn,
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
            error_panel(self.console, f"unknown command {command} — try /help")
            return
        handler(arg)

    def cmd_help(self, _arg: str) -> None:
        help_panel(self.console)

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
        absorbed = self.brain.store.compact()
        after = self.brain.stats()["total"]
        note(self.console, f"merged {absorbed} duplicate memories ({before} → {after})", OK)

    def cmd_remember(self, arg: str) -> None:
        if self.brain is None or not arg.strip():
            note(self.console, "usage: /remember <text>", WARN)
            return
        row_id, created = self.brain.remember("semantic", arg.strip(), "", 0.7)
        verb = "stored" if created else "merged into"
        note(self.console, f"{verb} memory #{row_id}", OK)

    def cmd_forget(self, arg: str) -> None:
        if self.brain is None:
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
        note(self.console, "distilling this session…")
        try:
            drafts, written = self.learner.distill(self.agent.messages)
        except LLMError as exc:
            error_panel(self.console, f"learn failed: {exc}")
            return
        if not drafts:
            note(self.console, "nothing durable found")
            return
        for draft, (row_id, created) in zip(drafts, written, strict=False):
            mark = "new" if created else "merged"
            self.console.print(Text(f"  [{draft.kind}/{mark} #{row_id}] {draft.text}", style="green"), markup=False)

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


def _session_path(cfg, resume: str | None) -> Path:
    cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        if resume == "last":
            existing = sorted(cfg.sessions_dir.glob("*.json"))
            return existing[-1] if existing else cfg.sessions_dir / f"{time.strftime('%Y%m%dT%H%M%S')}.json"
        candidate = cfg.sessions_dir / f"{resume}.json"
        return candidate if candidate.exists() else cfg.sessions_dir / f"{resume}"
    return cfg.sessions_dir / f"{time.strftime('%Y%m%dT%H%M%S')}.json"


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

    session_path = _session_path(cfg, args.resume)
    session_id = session_path.stem
    audit = Audit(cfg.audit_path, session_id, cfg.logging.audit)
    manager = ServerManager(cfg, audit)

    try:
        client = manager.ensure()
    except LLMError as exc:
        error_panel(console, str(exc))
        audit.close()
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

    registry = build_registry(cfg, audit, brain, confirm if cfg.tools.shell_confirm else None)
    agent = Agent(cfg, client, registry, brain, audit, ui)
    learner = Learner(cfg, client, brain, audit)
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
        )

    repl = Repl(cfg, client, manager, brain, learner, agent, audit, ui, session_path)

    def finish() -> None:
        if brain is not None and agent.messages:
            try:
                _, written = learner.distill(agent.messages)
                if written and console.is_terminal:
                    new = sum(1 for _, created in written if created)
                    note(
                        console,
                        f"learned {len(written)} memories ({new} new, {len(written) - new} merged)",
                    )
            except LLMError:
                pass
        if agent.messages:
            agent.persist(session_path)
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
        return toolbar_text(
            ctx=cfg.model.n_ctx,
            memories=brain.stats()["total"] if brain else 0,
            ram=profile.ram_gb,
            shell_mode="confirm" if cfg.tools.shell_confirm else "auto",
        )

    session = make_prompt_session(history_file, toolbar)
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
            except KeyboardInterrupt:
                ui.interrupt()
                note(console, "interrupted", WARN)
    finally:
        finish()
        if console.is_terminal:
            note(console, "bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())

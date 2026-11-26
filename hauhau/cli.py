"""Interactive REPL."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import config as config_mod
from .agent import Agent, UI
from .audit import Audit
from .brain import Brain
from .brain.learn import Learner
from .llm import LLMError
from .profiler import apply_recommendations, probe, recommend
from .server import ServerManager
from .tools import ToolResult, build_registry
from .util import estimate_messages_tokens, human_bytes

HELP = """\
[bold]Slash commands[/bold]
  /help              this list
  /tools             tools the model can call
  /mem [query]       memory stats, or search memories
  /compact           merge near-duplicate memories
  /remember <text>   store a fact yourself
  /forget <id>       delete a memory by id
  /learn             distil this session into long-term memory now
  /model             backend and model status
  /perf              measure real tokens/second
  /config            effective configuration
  /audit [n]         tail the audit log
  /context           token usage of the current conversation
  /clear             forget the current conversation (memory stays)
  /save              persist the session now
  /quit              leave (runs a /learn pass first)
"""


class RichUI(UI):
    """Streams tokens to the terminal without Rich re-parsing model output."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._open = False
        self._thinking = False

    def _close_thinking(self) -> None:
        if self._thinking:
            self.console.print()
            self._thinking = False

    def token(self, text: str) -> None:
        self._close_thinking()
        self._open = True
        self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def reasoning(self, text: str) -> None:
        if not self._thinking:
            if self._open:
                self.console.print()
                self._open = False
            self.console.print(Text("thinking…", style="dim italic"))
            self._thinking = True
        self.console.print(Text(text, style="dim italic"), end="", soft_wrap=True)

    def newline(self) -> None:
        if self._open or self._thinking:
            self.console.print()
        self._open = self._thinking = False

    def status(self, text: str) -> None:
        self.newline()
        self.console.print(Text(f"· {text}", style="dim"))

    def tool_call(self, name: str, args: str) -> None:
        self.newline()
        self.console.print(Text(f"⚙ {name} {args}", style="cyan"), soft_wrap=True, markup=False)

    def tool_result(self, name: str, result: ToolResult) -> None:
        lines = [ln for ln in (result.content or "").strip().splitlines() if ln.strip()]
        # The shell tool prefixes "exit=N"; show the actual output instead.
        body = [ln for ln in lines if not ln.startswith("exit=")] or lines
        preview = body[0][:200] if body else "(empty)"
        extra = f" (+{len(body) - 1} lines)" if len(body) > 1 else ""
        self.console.print(
            Text(f"  ↳ {preview}{extra}", style="green" if result.ok else "red"),
            soft_wrap=True,
            markup=False,
        )


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
        table = Table(title=title, title_justify="left", show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(table)

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
            self.console.print(Text(f"unknown command {command}; try /help", style="red"))
            return
        handler(arg)

    def cmd_help(self, _arg: str) -> None:
        self.console.print(HELP, markup=True)

    def cmd_tools(self, _arg: str) -> None:
        table = Table(title="Tools", title_justify="left")
        table.add_column("name", style="cyan", no_wrap=True)
        table.add_column("description", overflow="fold")
        for tool in self.agent.registry.tools():
            table.add_row(tool.name, tool.description.splitlines()[0])
        self.console.print(table)

    def cmd_mem(self, arg: str) -> None:
        if self.brain is None:
            self.console.print(Text("memory is disabled (--no-brain)", style="yellow"))
            return
        if arg.strip():
            hits = self.brain.recall(arg.strip(), k=10)
            if not hits:
                self.console.print(Text(f"nothing remembered about {arg.strip()!r}", style="dim"))
                return
            table = Table(title=f"memory: {arg.strip()}", title_justify="left")
            table.add_column("id", style="dim", no_wrap=True)
            table.add_column("kind", style="cyan", no_wrap=True)
            table.add_column("score", no_wrap=True)
            table.add_column("text", overflow="fold")
            for hit in hits:
                table.add_row(str(hit.id), hit.kind, f"{hit.score:.2f}", hit.text)
            self.console.print(table)
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
            self.console.print(Text("memory is disabled (--no-brain)", style="yellow"))
            return
        before = self.brain.stats()["total"]
        absorbed = self.brain.store.compact()
        after = self.brain.stats()["total"]
        self.console.print(
            Text(f"merged {absorbed} duplicate memories ({before} -> {after})", style="green")
        )

    def cmd_remember(self, arg: str) -> None:
        if self.brain is None or not arg.strip():
            self.console.print(Text("usage: /remember <text>", style="yellow"))
            return
        row_id, created = self.brain.remember("semantic", arg.strip(), "", 0.7)
        verb = "stored" if created else "merged into"
        self.console.print(Text(f"{verb} memory #{row_id}", style="green"))

    def cmd_forget(self, arg: str) -> None:
        if self.brain is None:
            return
        try:
            row_id = int(arg.strip())
        except ValueError:
            self.console.print(Text("usage: /forget <id>", style="yellow"))
            return
        removed = self.brain.store.forget(row_id)
        self.console.print(Text("forgotten" if removed else "no such memory", style="green" if removed else "yellow"))

    def cmd_learn(self, _arg: str) -> None:
        if self.brain is None:
            self.console.print(Text("memory is disabled (--no-brain)", style="yellow"))
            return
        if not self.agent.messages:
            self.console.print(Text("nothing to learn from yet", style="dim"))
            return
        self.console.print(Text("distilling this session…", style="dim"))
        try:
            drafts, written = self.learner.distill(self.agent.messages)
        except LLMError as exc:
            self.console.print(Text(f"learn failed: {exc}", style="red"))
            return
        if not drafts:
            self.console.print(Text("nothing durable found", style="dim"))
            return
        for draft, (row_id, created) in zip(drafts, written):
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
        self.console.print(Text("benchmarking…", style="dim"))
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
            self.console.print(Text(f"benchmark failed: {exc}", style="red"))
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
            self.console.print(Text("audit log is empty", style="dim"))
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
        self.console.print(Text("conversation cleared", style="green"))

    def cmd_save(self, _arg: str) -> None:
        self.agent.persist(self.session_path)
        self.console.print(Text(f"saved to {self.session_path}", style="green"))

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
    parser.add_argument("--yolo", action="store_true", help="accepted for compatibility; auto-run is already the default")
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
    console = Console()
    ui = RichUI(console)

    try:
        cfg, warnings = config_mod.load(args.config)
    except FileNotFoundError as exc:
        console.print(Text(str(exc), style="red"))
        return 2
    for warning in warnings:
        console.print(Text(f"config: {warning}", style="yellow"))
    _apply_args(cfg, args)

    profile = probe()
    if args.profile:
        rows = list(profile.as_dict().items())
        table = Table(title="Hardware", title_justify="left", show_header=False, box=None)
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        for key, value in rows:
            if key == "notes":
                continue
            table.add_row(key, str(value))
        console.print(table)
        size = 0
        if cfg.model.gguf:
            gguf_path = os.path.expanduser(cfg.model.gguf)
            size = os.path.getsize(gguf_path) if os.path.isfile(gguf_path) else 0
        for name, (value, reason) in recommend(profile, size).items():
            console.print(Text(f"{name} = {value}: {reason}", style="dim"))
        return 0

    if cfg.source is None:
        changes = apply_recommendations(cfg, profile)
        if changes:
            console.print(Text("auto-tuned for this machine:", style="dim"))
            for change in changes:
                console.print(Text(f"  {change}", style="dim"))

    session_path = _session_path(cfg, args.resume)
    session_id = session_path.stem
    audit = Audit(cfg.audit_path, session_id, cfg.logging.audit)
    manager = ServerManager(cfg, audit)

    try:
        client = manager.ensure()
    except LLMError as exc:
        console.print(Text(str(exc), style="red"))
        audit.close()
        return 1

    brain = None
    if cfg.brain.enabled:
        try:
            brain = Brain(cfg, audit)
        except Exception as exc:  # noqa: BLE001 - memory must never block startup
            console.print(Text(f"memory disabled: {exc}", style="yellow"))

    def confirm(cmd: str) -> bool:
        try:
            answer = pt_prompt(f"run? {cmd[:120]} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")

    registry = build_registry(cfg, audit, brain, confirm if cfg.tools.shell_confirm else None)
    agent = Agent(cfg, client, registry, brain, audit, ui)
    learner = Learner(cfg, client, brain, audit)
    if args.resume and session_path.exists():
        if agent.load(session_path):
            console.print(Text(f"resumed {agent.turn_count()} turns from {session_path.name}", style="dim"))

    # --- banner --------------------------------------------------------
    stats = brain.stats() if brain else {"total": 0}
    console.print(
        Text(
            f"{cfg.agent.name} · {os.path.basename(os.path.expanduser(cfg.model.gguf)) or 'unknown model'} · "
            f"{cfg.model.n_ctx} ctx · {profile.ram_gb:g} GB RAM · {stats['total']} memories",
            style="bold",
        ),
        soft_wrap=True,
    )
    console.print(
        Text(f"{cfg.model.server_url} · workspace {cfg.workspace} · /help for commands", style="dim"),
        soft_wrap=True,
    )

    repl = Repl(cfg, client, manager, brain, learner, agent, audit, ui, session_path)

    def finish() -> None:
        if brain is not None and agent.messages:
            try:
                _, written = learner.distill(agent.messages)
                if written:
                    new = sum(1 for _, created in written if created)
                    console.print(
                        Text(f"learned {len(written)} memories ({new} new, {len(written) - new} merged)", style="dim")
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
            console.print(Text(str(exc), style="red"))
        finally:
            finish()
        return 0

    # --- interactive ---------------------------------------------------
    history_file = cfg.data_dir / "history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
    )
    try:
        while repl.running:
            try:
                line = session.prompt("you › ")
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
                console.print(Text(f"error: {exc}", style="red"))
            except KeyboardInterrupt:
                ui.newline()
                console.print(Text("interrupted", style="yellow"))
    finally:
        finish()
        console.print(Text("bye", style="dim"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

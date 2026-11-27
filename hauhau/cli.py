"""Interactive REPL."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt as pt_prompt
from rich.box import ROUNDED
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import config as config_mod
from .agent import Agent
from .audit import Audit
from .brain import Brain
from .brain.learn import Learner, worth_distilling
from .evolve import (
    DEFAULT_TASKS,
    TUNABLE,
    PromptStore,
    available,
    benchmark,
    benchmark_prompt,
    evolve_prompt,
    hill_climb,
    metric_for,
    run_suite,
)
from .evolve import (
    candidates as tune_candidates,
)
from .evolve import (
    record as record_tuning,
)
from .evolve.tune import ctx_floor, get_value, log_tail, rejected_tokens, set_value
from .llm import LLMError
from .models import ModelError, display_name
from .models import catalog as model_catalog
from .models import resolve as resolve_model
from .models import switch as switch_model
from .profiler import apply_recommendations, probe, recommend
from .server import ServerManager
from .skills import discover as discover_skills
from .skills import index as skill_index
from .skills import nearest as skill_nearest
from .skills import skill_dirs
from .tools import build_registry
from .tui import ScreenApp, ScreenUI
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
        prompts: PromptStore | None = None, profile=None, journal=None, skills=None,
    ) -> None:
        self.cfg = cfg
        self.prompts = prompts or PromptStore(cfg.data_dir / "evolve" / "prompts")
        self.profile = profile
        self.client = client
        self.manager = manager
        self.brain = brain
        self.journal = journal
        self.skills = skills or []
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
            "/skills": self.cmd_skills,
            "/mem": self.cmd_mem,
            "/compact": self.cmd_compact,
            "/prune": self.cmd_prune,
            "/remember": self.cmd_remember,
            "/forget": self.cmd_forget,
            "/learn": self.cmd_learn,
            "/sync": self.cmd_sync,
            "/journal": self.cmd_journal,
            "/brief": self.cmd_brief,
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

    def cmd_skills(self, arg: str) -> None:
        """`/skills` lists, `/skills <name>` shows one, `/skills reload` re-scans."""
        wanted = arg.strip().lstrip("/")
        if wanted.lower() in ("reload", "scan", "rescan"):
            self._reload_skills()
            return
        if wanted:
            skill = skill_index(self.skills).get(wanted)
            if skill is None:
                close = skill_nearest(wanted, self.skills)
                hint = f" - did you mean /skills {close}?" if close else ""
                note(self.console, f"no skill named {wanted!r}{hint}", WARN)
                return
            tools = f" · needs {', '.join(skill.allowed_tools)}" if skill.allowed_tools else ""
            self.console.print(
                Panel(
                    Markdown(skill.body or "(empty body)"),
                    title=f"{skill.name}{tools}",
                    subtitle=str(skill.folder),
                    border_style=ACCENT,
                    box=ROUNDED,
                )
            )
            return
        if not self.skills:
            scanned = ", ".join(str(d) for d in skill_dirs(self.cfg))
            note(self.console, f"no skills installed - folders scanned: {scanned}")
            note(self.console, "drop a folder with a SKILL.md there, then run /skills reload")
            return
        data_table(
            self.console,
            "Skills",
            ["name", "description", "tools", "folder"],
            [
                [
                    s.name,
                    s.description.splitlines()[0],
                    ", ".join(s.allowed_tools) or "-",
                    str(s.folder),
                ]
                for s in self.skills
            ],
        )
        note(self.console, "/skills <name> shows one · /skills reload re-scans the folders")

    def _reload_skills(self) -> None:
        """Re-scan the skill folders: a skill added while the app runs is usable
        on the next turn, because the prompt block and the tool both re-read."""
        found, warnings = discover_skills(skill_dirs(self.cfg))
        for warning in warnings:
            note(self.console, f"skills: {warning}", WARN)
        before = {s.name for s in self.skills}
        after = {s.name for s in found}
        self.skills = found
        self.agent.skills = found
        added, removed = sorted(after - before), sorted(before - after)
        parts = []
        if added:
            parts.append(f"added {', '.join(added)}")
        if removed:
            parts.append(f"removed {', '.join(removed)}")
        note(self.console, f"{len(found)} skills" + (f" ({'; '.join(parts)})" if parts else ""))


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
            # Trials *per key*, not in total: with several keys tunable, a single
            # shared budget would only ever reach the first one. Bare /tune uses
            # a hardware-aware default (12 on 8 GB) so flag groups are reachable.
            from .profiler import default_tune_trials

            budget = int(arg) if arg.strip() else default_tune_trials(self.profile)
        except ValueError:
            note(self.console, "usage: /tune [trials-per-key]", WARN)
            return
        profile = self.profile
        cores = (profile.cores if profile else 0) or os.cpu_count() or 4
        ram_gb = profile.ram_gb if profile else 0.0
        has_gpu = bool(profile and (profile.metal or "nvidia" in profile.gpu.lower()))
        options = tune_candidates(self.cfg, cores, ram_gb=ram_gb, has_gpu=has_gpu)
        if not options:
            note(self.console, "nothing to try: the config already has the candidate values")
            return
        groups: dict[str, list] = {}
        for option in options:
            groups.setdefault(option.key, []).append(option)
        # Every tunable value, so the finally-block can restore all of them.
        original = {key: get_value(self.cfg, key) for key in TUNABLE}
        trials: list = []
        adopted: list = []
        stopped = ""
        baseline_note = ""
        with Progress(self.console, "benchmarking the current config…") as progress:
            self.audit.log("tune.start", trials=budget, keys=sorted(groups))

            def measure(option) -> float:
                set_value(self.cfg, option.key, option.value)
                self.audit.log("tune.trial", setting=option.key, value=option.value)
                if option.key.startswith("model."):
                    # Server flags and model settings only exist after a restart.
                    progress.update(f"restarting for {option}…")
                    log = self.cfg.data_dir / "logs" / "server.log"
                    before = log.stat().st_size if log.is_file() else 0
                    self.manager.restart()
                    refused = rejected_tokens(log_tail(log, before), _flag_tokens(option))
                    if refused:
                        # llama-server started, declined the flag and measured
                        # something else - exactly how --cache-reuse "won" a
                        # trial while the log said it was disabled.
                        note(
                            self.console,
                            f"{option}: llama-server refused {', '.join(refused)}, trial discarded",
                            WARN,
                        )
                        return 0.0
                    progress.update(f"benchmarking {option}…")
                    return benchmark(self.client)
                progress.update(f"benchmarking {option} (prompt speed)…")
                return benchmark_prompt(self.client)

            try:
                for key, group in groups.items():
                    metric = metric_for(key)
                    try:
                        base = benchmark(self.client) if metric == "gen" else benchmark_prompt(self.client)
                    except LLMError as exc:
                        # One key without timings must not stop the others.
                        error_panel(self.console, f"baseline benchmark failed for {key}: {exc}")
                        stopped = f"{key} baseline failed: {exc}"
                        continue
                    piece = f"{key}={base:.1f} tok/s"
                    baseline_note = f"{baseline_note} · {piece}" if baseline_note else f"{piece}"
                    progress.step(f"{key}: baseline {base:.1f} tok/s · up to {budget} values")
                    try:
                        result = hill_climb(base, group, measure, max_trials=budget)
                    except LLMError as exc:
                        error_panel(self.console, f"tuning {key} aborted: {exc}")
                        stopped = f"{key} aborted: {exc}"
                        break
                    trials.extend(result.trials)
                    adopted.extend(result.adopted)
                    if result.stopped:
                        stopped = result.stopped
            finally:
                # Whatever happened, the running server must match the chosen values.
                for key, value in original.items():
                    set_value(self.cfg, key, value, allow_below_floor=True)
                for chosen in adopted:
                    set_value(self.cfg, chosen.key, chosen.value)
                try:
                    self.manager.restart()
                except (LLMError, OSError) as exc:
                    # Tuning must never take the session down with it.
                    note(self.console, f"backend restart failed: {exc} - run /model to check it", WARN)

        if not trials:
            note(self.console, f"nothing was measured. {stopped}", WARN)
            return
        rows = [
            (trial.candidate, f"{trial.score:.1f} tok/s", "✓ adopted" if trial.adopted else "kept")
            for trial in trials
        ]
        data_table(self.console, "Tuning trials", ["setting", "speed", "result"], rows, no_wrap=2)
        if adopted:
            from .evolve.tune import TuneResult

            merged = TuneResult(baseline=0.0, best=0.0, trials=trials, adopted=adopted, stopped=stopped)
            target = self._save_config("tune")
            record_tuning(self.cfg.data_dir / "evolve" / "tuning.jsonl", merged, adopted_config=str(target))
            self.audit.log("tune.adopt", settings=[str(c) for c in adopted])
            note(
                self.console,
                f"adopted {', '.join(str(c) for c in adopted)} ({baseline_note.strip()}) - saved to {target}",
                OK,
            )
        else:
            note(
                self.console,
                f"kept the current settings ({baseline_note.strip()}). {stopped or 'no candidate was faster'}",
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
                    tries=3,
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
            reason=result.reason,
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

    def cmd_model(self, arg: str) -> None:
        """/model lists the local .gguf files; /model <name> switches backend.

        Changing the model means restarting llama-server, so the config is only
        saved once the new server is healthy - a model that fails to load leaves
        both the running session and config.toml as they were.
        """
        if arg.strip():
            self._switch_model(arg.strip())
            return
        entries = model_catalog(self.cfg)
        if entries:
            rows = [
                [
                    str(index),
                    entry.name,
                    entry.size_text,
                    "✓ active" if entry.active else "",
                ]
                for index, entry in enumerate(entries, start=1)
            ]
            data_table(self.console, "Models", ["#", "name", "size", "in use"], rows, no_wrap=2)
        else:
            note(self.console, "no .gguf files found - drop one in ~/modeller/gguf", WARN)
        status = self.manager.status()
        self._table(
            "Backend",
            [
                ("model", status["gguf"] or "(none)"),
                ("server", f"{status['url']} ({'managed' if status['managed'] else 'external'})"),
                ("context", f"{status['n_ctx']} tokens"),
                ("threads", str(status["n_threads"])),
                ("gpu layers", str(status["n_gpu_layers"])),
                ("think", str(self.cfg.generation.think)),
                ("last speed", self._last_speed()),
            ],
        )
        note(self.console, "/model <number|name> switches - the backend restarts")

    def _last_speed(self) -> str:
        completion = getattr(self.agent, "last_completion", None)
        if completion is None or not completion.usage.tok_per_second:
            return "-"
        return f"{completion.usage.tok_per_second:.1f} tok/s"

    def _switch_model(self, query: str) -> None:
        try:
            entry = resolve_model(self.cfg, query)
        except ModelError as exc:
            error_panel(self.console, str(exc))
            return
        if entry.active:
            note(self.console, f"{entry.name} is already the active model")
            return
        old_path = self.cfg.model.gguf
        old_args = list(self.cfg.model.extra_args)
        old_ctx, new_ctx = switch_model(self.cfg, entry, ctx_floor=ctx_floor(self.cfg))
        if not self.cfg.model.manage_server:
            # Without a managed server the caller restarts by hand; save anyway so
            # the next start picks the model up.
            target = self._save_config("model switch")
            self.audit.log("model.switch", model=str(entry.path), ctx=new_ctx, config=str(target))
            note(
                self.console,
                f"config now points at {entry.name}; the backend is external, restart it yourself",
                WARN,
            )
            return
        note(self.console, f"loading {entry.name} ({entry.size_text}, ctx {new_ctx})…")
        try:
            self.manager.restart()
        except (LLMError, OSError) as exc:
            # Put the old values back so a failed load cannot leave a config that
            # points at a model the machine cannot run.
            self.cfg.model.gguf = old_path
            self.cfg.model.n_ctx = old_ctx
            self.cfg.model.extra_args = old_args
            error_panel(self.console, f"{entry.name} did not start: {exc}")
            note(self.console, "kept the previous model - config.toml was not changed", WARN)
            try:
                self.manager.restart()
            except (LLMError, OSError):
                note(self.console, "the previous backend is not running either; check /model", WARN)
            return
        target = self._save_config("model switch")
        self.audit.log("model.switch", model=str(entry.path), ctx=new_ctx, config=str(target))
        note(
            self.console,
            f"{entry.name} is live · ctx {new_ctx} · {self._last_speed()} · saved to {target.name}",
            OK,
        )

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

    # --- journal -------------------------------------------------------
    def _require_journal(self):
        if self.journal is None:
            error_panel(
                self.console,
                "the journal is unavailable: either [journal] enabled = false in config.toml, "
                "or it failed to open at startup (a warning was printed then). "
                "Set enabled = true and restart to use it.",
            )
            return None
        return self.journal

    def cmd_sync(self, arg: str) -> None:
        """Pull mails, messages, calls, events and reminders into the journal."""
        journal = self._require_journal()
        if journal is None:
            return
        from .ingest import FETCHERS, sync

        wanted = [name.strip() for name in arg.replace(",", " ").split() if name.strip()]
        unknown = [name for name in wanted if name not in FETCHERS]
        if unknown:
            error_panel(self.console, f"unknown source {unknown[0]!r} — pick from {', '.join(FETCHERS)}")
            return
        with Progress(self.console, "syncing…") as progress:
            report = sync(journal, wanted or None, journal_cfg=self.cfg.journal, progress=progress.update)
        rows: list[tuple[str, str]] = []
        failed: list[str] = []
        for name, result in report.items():
            if "error" in result:
                failed.append(f"{name}: {result['error']}")
                rows.append((name, f"[red]failed[/red] {result['error'][:90]}"))
            else:
                rows.append((name, f"{result['added']} new of {result['fetched']} seen"))
        self.audit.log("sync", report=report)
        data_table(self.console, "journal sync", ["source", "result"], [list(r) for r in rows])
        if failed:
            error_panel(self.console, failed[0])

    def cmd_journal(self, arg: str) -> None:
        """Search the journal, or show what it holds."""
        journal = self._require_journal()
        if journal is None:
            return
        query = arg.strip()
        if not query:
            stats = journal.stats()
            by_source = ", ".join(f"{k} {v}" for k, v in dict(stats["by_source"]).items()) or "empty"
            newest = stats["newest"]
            last = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(float(newest)))
                if newest
                else "never"
            )
            self._table(
                "journal",
                [
                    ("events", str(stats["total"])),
                    ("by source", by_source),
                    ("newest", last),
                    ("database", f"{journal.path} ({journal.path.stat().st_size // 1024} KB)"),
                ],
            )
            return
        rows = journal.search(query, k=20)
        if not rows:
            note(self.console, f"nothing in the journal matches {query!r}", WARN)
            return
        data_table(
            self.console,
            f"{len(rows)} match(es) for {query!r}",
            ["when", "event"],
            [[row.when, f"{row.kind} {row.title or row.body}"[:100]] for row in rows],
        )

    def cmd_brief(self, _arg: str) -> None:
        """Today's digest: what arrived, what is due, what is next."""
        journal = self._require_journal()
        if journal is None:
            return
        from .brief import compose

        digest = compose(journal, self.cfg)
        self.audit.log("brief", where="repl")
        self.console.print(digest)


def _flag_tokens(option) -> list[str]:
    """The server flags a trial candidate asks for: the part that can be refused."""
    value = getattr(option, "value", None)
    if getattr(option, "key", "") != "model.extra_args" or not isinstance(value, list):
        return []
    return [token for token in value if isinstance(token, str) and token.startswith("-")]


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
    parser.add_argument(
        "--plain",
        action="store_true",
        help="streaming terminal interface instead of the full-screen one (automatic when piped)",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="widget dashboard (chat, live stats, scrolling) instead of the full-screen frame",
    )
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
    if args.max_steps is not None:
        cfg.agent.max_steps = max(1, args.max_steps)


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
    name = resume if resume.endswith(".json") else f"{resume}.json"
    candidate = cfg.sessions_dir / name
    # Check containment on the resolved path (symlinks included) but keep the
    # unresolved one: /var is a symlink to /private/var on macOS, and a session
    # path that suddenly reads "/private/var/..." is a surprise in the logs.
    root = cfg.sessions_dir.resolve()
    try:
        inside = root in candidate.resolve().parents
    except OSError:
        inside = False
    if not inside:
        # --resume ../../notes would otherwise load, and later overwrite, any
        # .json file the user can reach from the data directory.
        return fresh, f"refusing --resume {resume!r}: it points outside {root}"
    if candidate.is_file():
        return candidate, None
    return fresh, f"no session named {resume!r} - starting a new one"


def _wants_full_screen() -> bool:
    """The full-screen interface needs a real terminal on both ends."""
    if os.environ.get("TERM") in ("", "dumb"):
        return False
    try:
        if not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


class _StartupError(Exception):
    """A startup step that ends the run with a panel and an exit code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _pick_interface(args) -> tuple[Any, Any, Any, str]:
    """Choose the interface, returning (console, term_console, ui, mode).

    ``mode`` is "dashboard", "screen" or "plain". The interface owns the terminal
    when it is not plain, so its console replaces the streaming one; the original
    is kept for the parting line that has to survive the interface shutting down.
    """
    console = make_console()
    ui = RichUI(console)
    term_console = console
    full_screen = _wants_full_screen() and not args.plain
    if full_screen and args.tui:
        try:
            from .dashboard import DashboardUI
            from .dashboard import available as dashboard_available
        except ImportError as exc:
            raise _StartupError(2, str(exc)) from exc
        if not dashboard_available():  # pragma: no cover - import already checked
            raise _StartupError(
                2, "the dashboard needs textual: .venv/bin/pip install 'hauhau[tui]'"
            )
        dashboard_ui = DashboardUI()
        return dashboard_ui.console, term_console, dashboard_ui, "dashboard"
    if full_screen:
        screen_ui = ScreenUI()
        return screen_ui.console, term_console, screen_ui, "screen"
    return console, term_console, ui, "plain"


def _load_config(args, console):
    try:
        cfg, warnings = config_mod.load(args.config)
    except (FileNotFoundError, config_mod.ConfigError) as exc:
        raise _StartupError(2, str(exc)) from exc
    for warning in warnings:
        note(console, f"config: {warning}", WARN)
    _apply_args(cfg, args)
    return cfg


def _profile_report(console, cfg, profile) -> None:
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


class Runtime:
    """A started session: backend, memory, journal, agent and the REPL.

    ``main`` only sequences it. The startup order matters - the backend first
    (nothing works without it), then memory, then the prompt version that becomes
    part of the system prompt, then the journal and the tools - and the shutdown
    that must follow it whatever happens is here in one piece.
    """

    def __init__(self, args, cfg, profile, console, term_console, ui, mode: str) -> None:
        self.args = args
        self.cfg = cfg
        self.profile = profile
        self.console = console
        self.term_console = term_console
        self.ui = ui
        self.mode = mode
        self.session_path, self.session_warning = _session_path(cfg, args.resume)
        self.audit = Audit(cfg.audit_path, self.session_path.stem, cfg.logging.audit)
        self.manager = ServerManager(cfg, self.audit)
        self.prompts = PromptStore(cfg.data_dir / "evolve" / "prompts")
        self.adopted: int | None = None
        self.brain = None
        self.journal = None
        self.skills = []
        self.client = None
        self.registry = None
        self.agent = None
        self.learner = None
        self.repl = None
        self._panel_cache: dict[str, Any] = {"at": 0.0, "memories": "-", "journal": "-"}
        self._memory_cache: dict[str, Any] = {"at": 0.0, "text": ""}

    # --- startup -------------------------------------------------------
    def start(self) -> None:
        self.client = self._open_backend()
        self.brain = self._open_brain()
        self.adopted = apply_active_prompt(self.cfg, self.prompts)
        if self.adopted is not None and self.console.is_terminal:
            note(self.console, f"system prompt v{self.adopted} active")
        self.journal = self._open_journal()
        self.skills = self._open_skills()
        self.registry = build_registry(
            self.cfg,
            self.audit,
            self.brain,
            self.confirm if self.cfg.tools.shell_confirm else None,
            journal=self.journal,
            skills=self.skills,
        )
        self.agent = Agent(self.cfg, self.client, self.registry, self.brain, self.audit, self.ui, skills=self.skills)
        self.learner = Learner(self.cfg, self.client, self.brain, self.audit)
        if self.session_warning:
            note(self.console, self.session_warning, WARN)
        if self.args.resume and self.session_path.exists() and self.agent.load(self.session_path):
            note(self.console, f"resumed {self.agent.turn_count()} turns from {self.session_path.name}")
        self._banner()
        self.repl = Repl(
            self.cfg, self.client, self.manager, self.brain, self.learner, self.agent, self.audit,
            self.ui, self.session_path, prompts=self.prompts, profile=self.profile,
            journal=self.journal, skills=self.skills,
        )

    def _open_skills(self):
        """SKILL.md folders; a broken skill warns, it never blocks startup."""
        if not self.cfg.skills.enabled:
            return []
        found, warnings = discover_skills(skill_dirs(self.cfg))
        for warning in warnings:
            note(self.console, f"skills: {warning}", WARN)
        return found

    def _open_backend(self):
        try:
            return self.manager.ensure()
        except LLMError as exc:
            self.audit.close()
            self.manager.stop()
            raise _StartupError(1, str(exc)) from exc

    def _open_brain(self):
        if not self.cfg.brain.enabled:
            return None
        try:
            return Brain(self.cfg, self.audit)
        except Exception as exc:  # memory must never block startup
            note(self.console, f"memory disabled: {exc}", WARN)
            return None

    def _open_journal(self):
        if not self.cfg.journal.enabled:
            return None
        try:
            from .journal import Journal

            return Journal(self.cfg.journal_path)
        except Exception as exc:  # the journal must never block startup
            note(self.console, f"journal disabled: {exc}", WARN)
            return None

    def _banner(self) -> None:
        """The stream interface's header; the full-screen ones draw their own."""
        if not (self.console.is_terminal and self.mode == "plain"):
            return
        try:
            stats = self.brain.stats() if self.brain else {"total": 0}
        except Exception:  # a broken brain.db must not block the CLI
            stats = {"total": 0}
        banner(
            self.console,
            name=self.cfg.agent.name,
            model=os.path.basename(os.path.expanduser(self.cfg.model.gguf)),
            ctx=self.cfg.model.n_ctx,
            threads=self.cfg.model.n_threads,
            ram=self.profile.ram_gb,
            memories=stats["total"],
            url=self.cfg.model.server_url,
            workspace=str(self.cfg.workspace),
            prompt=f"v{self.adopted}" if self.adopted else "",
        )

    # --- callbacks the interfaces use ----------------------------------
    def confirm(self, cmd: str) -> bool:
        try:
            answer = pt_prompt(f"run? {cmd[:120]} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")

    def revive_backend(self) -> bool:
        """Make sure a backend exists after a failed turn."""
        if not self.cfg.model.manage_server:
            return False
        try:
            self.manager.ensure()
            return True
        except LLMError:
            return False

    def restart_backend(self) -> bool:
        """Abort the in-flight answer by restarting the backend Hauhau owns.

        Only a restart drops the generation llama-server is busy with: ensure()
        returns a healthy server untouched, so Ctrl-C reported "the backend was
        restarted" while the answer kept streaming.
        """
        if not self.cfg.model.manage_server:
            return False
        try:
            self.manager.restart()
            return True
        except (LLMError, OSError):
            return False

    def submit(self, text: str) -> None:
        """One entered line: either a slash command or a chat turn."""
        if text.startswith("/"):
            self._run_command(text)
            return
        try:
            self.agent.run_turn(text)
            self.ui.newline()
        except LLMError as exc:
            self.ui.interrupt()
            error_panel(self.console, str(exc))
            if self.revive_backend():
                note(self.console, "backend restarted - ask again", WARN)
        except KeyboardInterrupt:
            self.ui.interrupt()
            note(self.console, "interrupted", WARN)
            self.revive_backend()

    def _run_command(self, text: str) -> None:
        try:
            self.repl.handle(text)
        except Exception as exc:  # a command bug must not end the session
            command = text.split(maxsplit=1)[0]
            self.audit.log(
                "command.error", command=command, error=f"{exc.__class__.__name__}: {exc}"
            )
            error_panel(self.console, f"{command} failed: {exc}")

    def dashboard_stats(self) -> dict:
        """Side-panel values for the dashboard, two aggregates cached for 2 s."""
        now = time.monotonic()
        if now - self._panel_cache["at"] > 2.0:
            self._panel_cache["at"] = now
            if self.brain is not None:
                try:
                    self._panel_cache["memories"] = f"{self.brain.stats()['total']} mem"
                except Exception:
                    self._panel_cache["memories"] = "-"
            if self.journal is not None:
                try:
                    self._panel_cache["journal"] = f"{self.journal.stats()['total']} events"
                except Exception:
                    self._panel_cache["journal"] = "-"
        active = self.prompts.active_number()
        return {
            "name": self.cfg.agent.name,
            "model": display_name(
                Path(os.path.expanduser(self.cfg.model.gguf or "model.gguf")).stem, width=22
            ),
            "backend": self.cfg.model.server_url,
            "ctx": self.ctx_label(),
            "threads": f"{self.cfg.model.n_threads}t",
            "prompt": f"v{active}" if active else "config",
            "memories": self._panel_cache["memories"],
            "journal": self._panel_cache["journal"],
            "tuner": "managed" if self.cfg.model.manage_server else "external",
            "session": self.session_path.stem[:16],
            "workspace": str(self.cfg.workspace),
        }

    def screen_stats(self) -> dict:
        """Header values for the fixed frame; the memory count is on a timer."""
        now = time.monotonic()
        if self.brain is not None and now - self._memory_cache["at"] > 2.0:
            self._memory_cache["at"] = now
            try:
                self._memory_cache["text"] = f"{self.brain.stats()['total']} mem"
            except Exception:
                self._memory_cache["text"] = ""
        return {
            "name": self.cfg.agent.name,
            "model": display_name(
                Path(os.path.expanduser(self.cfg.model.gguf or "model.gguf")).stem, width=18
            ),
            "ctx": self.ctx_label(),
            "threads": f"{self.cfg.model.n_threads}t",
            "ram": f"{self.profile.ram_gb:g} GB",
            "workspace": short_name(str(self.cfg.workspace), width=16),
            "memories": self._memory_cache["text"],
            "state": "thinking…" if self.ui.busy else "ready",
        }

    def ctx_label(self) -> str:
        ctx = self.cfg.model.n_ctx
        return f"{ctx // 1024}k" if ctx >= 1024 else str(ctx)

    def toolbar(self) -> str:
        active = self.prompts.active_number()
        counts = None
        if self.brain is not None:
            try:
                counts = self.brain.stats()["total"]
            except Exception:  # a broken brain.db must not break the prompt line
                counts = 0
        return toolbar_text(
            ctx=self.cfg.model.n_ctx,
            memories=counts or 0,
            ram=self.profile.ram_gb,
            shell_mode="confirm" if self.cfg.tools.shell_confirm else "auto",
            prompt=f"v{active}" if active else "",
        )

    def completions(self, text: str) -> list[tuple[str, str]]:
        """Argument completion for the commands whose arguments the CLI knows."""
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        if command == "/help":
            # No leading slash: completing "ev" must yield "/help evolve", not
            # "/help /evolve". cmd_help normalises either form.
            return [(name.lstrip("/"), description) for name, description in COMMANDS]
        if command == "/model":
            return [
                (str(index), entry.name)
                for index, entry in enumerate(model_catalog(self.cfg), start=1)
                if not entry.active
            ]
        if command == "/prompt":
            rows = []
            for version in self.prompts.versions():
                score = f"{version.score:.0%}" if version.score is not None else "baseline"
                rows.append((f"v{version.number}", score))
            return rows
        if command in ("/show", "/audit") and not argument:
            return [("1", "most recent"), ("2", "one back"), ("3", "two back")]
        return []

    # --- shutdown ------------------------------------------------------
    def finish(self) -> None:
        """Learn from the session, then shut down - in that order, always.

        The shutdown lives in ``finally`` because a failure in the learning pass
        or in the session save used to skip ``manager.stop()`` and leave a
        llama-server running after the CLI had exited.
        """
        try:
            self._learn()
            self._persist()
        finally:
            self._close()

    def _learn(self) -> None:
        if self.brain is None or not self.agent.messages or not worth_distilling(self.agent.messages):
            return
        if self.console.is_terminal:
            note(self.console, "distilling this session into memory…")
        try:
            _, written = self.learner.distill(self.agent.messages)
        except KeyboardInterrupt:  # Ctrl-C during the pass: skip it
            if self.console.is_terminal:
                note(self.console, "skipped learning this session", WARN)
            return
        except Exception as exc:  # learning must never block shutdown
            self.audit.log("learn.error", error=f"{exc.__class__.__name__}: {exc}")
            return
        if not self.console.is_terminal:
            return
        if written:
            new = sum(1 for _, created in written if created)
            note(
                self.console,
                f"learned {len(written)} memories ({new} new, {len(written) - new} merged)",
            )
        else:
            note(self.console, "nothing durable to keep from this session")

    def _persist(self) -> None:
        if not self.agent.messages:
            return
        try:
            self.agent.persist(self.session_path)
        except OSError as exc:
            if self.console.is_terminal:
                note(self.console, f"could not save the session: {exc}", WARN)

    def _close(self) -> None:
        # sqlite kept its handle (and its WAL) open until the interpreter exited;
        # close it with everything else, and stop the backend we started.
        if self.journal is not None:
            try:
                self.journal.close()
            except Exception as exc:  # pragma: no cover - best effort
                self.audit.log("journal.close_error", error=str(exc))
        if self.brain is not None:
            try:
                self.brain.close()
            except Exception as exc:  # pragma: no cover - best effort
                self.audit.log("brain.close_error", error=str(exc))
        self.audit.close()
        self.manager.stop()

    # --- running -------------------------------------------------------
    def one_shot(self, text: str) -> int:
        """Answer one prompt and exit; a script must see a failure as nonzero."""
        if not text:
            error_panel(self.console, "the prompt is empty")
            self.finish()
            return 2
        failed = False
        try:
            self.agent.run_turn(text)
            self.ui.newline()
        except LLMError as exc:
            error_panel(self.console, str(exc))
            failed = True
        finally:
            self.finish()
        return 1 if failed else 0

    def run(self) -> int:
        if self.mode == "dashboard":
            return self._run_dashboard()
        if self.mode == "screen":
            return self._run_screen()
        return self._run_plain()

    def _run_dashboard(self) -> int:
        from . import suggest as suggest_mod
        from .dashboard import HauhauDashboard

        tool_names = self.agent.registry.names()

        def propose(prefix: str) -> str | None:
            """One model round-trip for a completion; runs on a worker thread."""
            suggestion = suggest_mod.propose(self.client, prefix, tools=tool_names)
            self.audit.log("suggest.propose", prefix=prefix[:60], ok=suggestion is not None)
            return suggestion.text if suggestion else None

        window = HauhauDashboard(
            self.ui,
            submit=self.submit,
            restart_backend=self.restart_backend,
            should_exit=lambda: not self.repl.running,
            stats=self.dashboard_stats,
            propose=propose,
        )
        try:
            window.run()
        finally:
            self.finish()
            note(self.term_console, "bye")
        return 0

    def _run_screen(self) -> int:
        screen = ScreenApp(
            self.ui,
            submit=self.submit,
            restart_backend=self.restart_backend,
            should_exit=lambda: not self.repl.running,
            stats=self.screen_stats,
        )
        try:
            screen.run()
        finally:
            self.finish()
            note(self.term_console, "bye")
        return 0

    def _run_plain(self) -> int:
        history_file = self.cfg.data_dir / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        session = make_prompt_session(
            history_file, self.toolbar, SlashCompleter(self.completions)
        )
        try:
            while self.repl.running:
                try:
                    line = session.prompt(prompt_glyph())
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    # A slash command can run for minutes (/tune, /eval, /sync) and
                    # Ctrl-C during one used to end the session, while the same key
                    # only interrupts a chat turn.
                    self.submit(text)
                except KeyboardInterrupt:
                    note(self.console, "interrupted", WARN)
                    continue
        finally:
            self.finish()
            if self.console.is_terminal:
                note(self.console, "bye")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    interface = None
    try:
        interface = _pick_interface(args)
        cfg = _load_config(args, interface[0])
    except _StartupError as stop:
        error_panel(make_console() if interface is None else interface[0], stop.message)
        return stop.code
    console = interface[0]

    profile = probe()
    if args.profile:
        _profile_report(console, cfg, profile)
        return 0

    if cfg.source is None:
        changes = apply_recommendations(cfg, profile)
        if changes:
            note(console, "auto-tuned for this machine:")
            for change in changes:
                note(console, f"  {change}")

    runtime = Runtime(args, cfg, profile, console, interface[1], interface[2], interface[3])
    try:
        runtime.start()
    except _StartupError as stop:
        error_panel(console, stop.message)
        return stop.code

    if args.prompt:
        return runtime.one_shot(" ".join(args.prompt).strip())
    return runtime.run()




if __name__ == "__main__":
    sys.exit(main())

"""Configuration: dataclasses, TOML load/save, and GGUF discovery.

Runtime state (brain db, audit log, sessions) lives under ``data_dir`` and is
deliberately separate from the repo so it is git-ignored and survives upgrades.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

GGUF_SEARCH_DIRS = ("~/modeller/gguf", "~/.cache/llama.cpp", "~/models", "~/.cache/hauhau/models")

DEFAULT_PERSONALITY = """\
You are {name}, a CLI agent running locally on the user's machine. You have a
shell, a filesystem, and a persistent memory. Be direct and concrete: do the
work with tools instead of describing what you would do. Never claim a command
succeeded unless you ran it and saw the output. Prefer one decisive check over
several speculative ones. Keep replies short - the user is watching a terminal."""


@dataclass
class ModelConfig:
    gguf: str = ""
    server_url: str = "http://127.0.0.1:8080"
    manage_server: bool = True
    llama_server: str = "llama-server"
    n_ctx: int = 8192
    n_threads: int = 4
    n_gpu_layers: int = 99
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: float = 120.0


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    think: bool = False


@dataclass
class AgentConfig:
    name: str = "Hauhau"
    max_steps: int = 12
    context_budget_tokens: int = 5000
    keep_recent_turns: int = 6
    personality: str = DEFAULT_PERSONALITY


@dataclass
class ToolConfig:
    shell_confirm: bool = False
    shell_timeout: float = 120.0
    denylist: list[str] = field(default_factory=list)
    max_output_chars: int = 8000
    workspace: str = ""


@dataclass
class BrainConfig:
    enabled: bool = True
    top_k: int = 5
    half_life_days: float = 30.0
    dedupe_jaccard: float = 0.9
    dedupe_containment: float = 0.8
    inject_max_chars: int = 1500
    embeddings_url: str = ""
    embeddings_model: str = "embeddinggemma-300m"


@dataclass
class LoggingConfig:
    data_dir: str = "~/.hauhau"
    audit: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    source: Path | None = None

    # --- derived paths -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(os.path.expanduser(self.logging.data_dir))

    @property
    def audit_path(self) -> Path:
        return self.data_dir / "logs" / "audit.jsonl"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "brain" / "brain.db"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / ".snapshots"

    @property
    def workspace(self) -> Path:
        return Path(os.path.expanduser(self.tools.workspace or "~"))


def find_gguf() -> str:
    """Best-effort discovery of a local model when config leaves gguf empty."""
    override = os.environ.get("HAUHAU_GGUF")
    if override:
        return override
    candidates: list[Path] = []
    for entry in GGUF_SEARCH_DIRS:
        base = Path(os.path.expanduser(entry))
        if base.is_dir():
            candidates.extend(base.glob("*.gguf"))
    if not candidates:
        return ""
    return str(max(candidates, key=lambda p: p.stat().st_size))


def _apply(obj: Any, table: dict[str, Any], origin: str) -> list[str]:
    """Copy known keys from a parsed TOML table onto a dataclass instance."""
    unknown: list[str] = []
    known = {f.name: f for f in fields(obj)}
    for key, value in table.items():
        if key not in known:
            unknown.append(f"{origin}{key}")
            continue
        setattr(obj, key, value)
    return unknown


def load(path: str | os.PathLike[str] | None = None) -> tuple[Config, list[str]]:
    """Load config. Returns (config, warnings). Missing file yields defaults."""
    cfg = Config()
    warnings: list[str] = []

    target: Path | None = None
    if path is not None:
        target = Path(os.path.expanduser(str(path)))
        if not target.is_file():
            raise FileNotFoundError(f"config file not found: {target}")
    else:
        for candidate in (Path("config.toml"), Path(os.path.expanduser("~/.hauhau/config.toml"))):
            if candidate.is_file():
                target = candidate
                break

    if target is not None:
        with target.open("rb") as fh:
            data = tomllib.load(fh)
        for section, value in data.items():
            if not hasattr(cfg, section) or not is_dataclass(getattr(cfg, section)):
                warnings.append(f"unknown config section [{section}]")
                continue
            if not isinstance(value, dict):
                warnings.append(f"[{section}] must be a table")
                continue
            warnings.extend(_apply(getattr(cfg, section), value, f"{section}."))
        cfg.source = target

    if not cfg.model.gguf:
        cfg.model.gguf = find_gguf()
    return cfg, warnings


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    # Multi-line strings read better as TOML triple-quoted blocks.
    if "\n" in text:
        return f'"""\n{text}"""'
    return f'"{text}"'


def dumps(cfg: Config) -> str:
    """Serialise the config back to TOML (scalars, string lists, nested tables)."""
    lines = ["# Hauhau configuration (generated)", ""]
    for f in fields(cfg):
        if f.name == "source":
            continue
        section = getattr(cfg, f.name)
        lines.append(f"[{f.name}]")
        for sf in fields(section):
            lines.append(f"{sf.name} = {_fmt(getattr(section, sf.name))}")
        lines.append("")
    return "\n".join(lines)


def save(cfg: Config, path: str | os.PathLike[str] | None = None) -> Path:
    target = Path(os.path.expanduser(str(path))) if path else (cfg.source or Path("config.toml"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(cfg), encoding="utf-8")
    return target

"""Configuration: dataclasses, TOML load/save, and GGUF discovery.

Runtime state (brain db, audit log, sessions) lives under ``data_dir`` and is
deliberately separate from the repo so it is git-ignored and survives upgrades.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
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


def _describe(default: Any) -> str:
    if isinstance(default, bool):
        return "a boolean"
    if isinstance(default, int):
        return "an integer"
    if isinstance(default, float):
        return "a number"
    if isinstance(default, str):
        return "a string"
    if isinstance(default, list):
        return "a list of strings"
    return type(default).__name__


def _type_ok(default: Any, value: Any) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        # nan < 0.1 is False, so a NaN half-life sailed past every range check
        # and turned decay, scores and sorting into NaN.
        return math.isfinite(float(value))
    if isinstance(default, int):
        # bool is a subclass of int, but `max_steps = true` is always a mistake.
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    return True


class ConfigError(Exception):
    """A config file that exists but cannot be understood."""


def _apply(obj: Any, table: dict[str, Any], origin: str) -> list[str]:
    """Copy known keys onto a dataclass instance, checking each value's type.

    Without this, `extra_args = "not-a-list"` was accepted and then expanded
    character by character into the llama-server command line, and
    `max_steps = "many"` raised TypeError in the middle of a turn - a failure
    far from the typo that caused it.
    """
    warnings: list[str] = []
    known = {f.name: f for f in fields(obj)}
    for key, value in table.items():
        if key not in known:
            warnings.append(f"unknown setting {origin}{key}")
            continue
        spec = known[key]
        default = spec.default if spec.default is not MISSING else (
            spec.default_factory() if spec.default_factory is not MISSING else None
        )
        if not _type_ok(default, value):
            warnings.append(
                f"{origin}{key} must be {_describe(default)}; ignoring {value!r}"
            )
            continue
        if isinstance(default, float) and isinstance(value, int):
            value = float(value)
        setattr(obj, key, value)
    return warnings


def _sanity(cfg: Config) -> list[str]:
    """Clamp values whose bad range would only surface as a crash later."""
    warnings: list[str] = []
    checks = [
        ("model.n_ctx", 256),
        ("model.n_threads", 1),
        ("model.startup_timeout", 1.0),
        ("tools.shell_timeout", 0.1),
        ("tools.max_output_chars", 200),
        ("agent.max_steps", 1),
        ("agent.context_budget_tokens", 200),
        ("agent.keep_recent_turns", 1),
        ("brain.top_k", 1),
        ("brain.half_life_days", 0.1),
        ("brain.inject_max_chars", 100),
    ]
    ceilings = [
        ("brain.top_k", 50, "retrieval beyond 50 rows costs more prompt than it returns"),
        ("agent.max_steps", 50, "a turn that needs 50 tool calls is a bug, not a plan"),
    ]
    for dotted, maximum, why in ceilings:
        section, name = dotted.split(".")
        current = getattr(getattr(cfg, section), name)
        if current > maximum:
            warnings.append(f"{dotted} must be <= {maximum}; using {maximum} ({why})")
            setattr(getattr(cfg, section), name, maximum)
    for dotted, minimum in checks:
        section, name = dotted.split(".")
        current = getattr(getattr(cfg, section), name)
        if current < minimum:
            warnings.append(f"{dotted} must be >= {minimum}; using {minimum}")
            setattr(getattr(cfg, section), name, minimum)
    if cfg.model.server_url and not cfg.model.server_url.startswith(("http://", "https://")):
        warnings.append(
            f"model.server_url should start with http:// (got {cfg.model.server_url!r}); "
            "using http://127.0.0.1:8080"
        )
        cfg.model.server_url = "http://127.0.0.1:8080"
    return warnings


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
        try:
            with target.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            # A typo in config.toml used to escape as a raw TOMLDecodeError and
            # print a stack trace instead of saying which line is wrong.
            raise ConfigError(f"{target} is not valid TOML: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read {target}: {exc}") from exc
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
    warnings.extend(_sanity(cfg))
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


_SECTION = re.compile(r"^\s*\[([A-Za-z0-9_]+)\]\s*$")
_KEY = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*)=")


def _patch(text: str, cfg: Config) -> str:
    """Rewrite the values of an existing config, keeping every comment.

    The tuner adopts a setting by saving the file again. Regenerating it from
    scratch dropped the comments that explain *why* each value is what it is -
    the only documentation the file has - so values are edited in place and
    anything the dataclasses do not own (prose, comments, ordering) survives.
    """
    out: list[str] = []
    section: str | None = None
    written: set[tuple[str, str]] = set()
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        header = _SECTION.match(line)
        if header:
            section = header.group(1)
            out.append(line)
            index += 1
            continue
        key = _KEY.match(line)
        owner = getattr(cfg, section, None) if section else None
        if key is None or owner is None or not is_dataclass(owner):
            out.append(line)
            index += 1
            continue
        name = key.group(2)
        known = {f.name for f in fields(owner)}
        if name not in known:
            out.append(line)
            index += 1
            continue
        value = getattr(owner, name)
        written.add((section, name))
        rest = line.split("=", 1)[1].lstrip()
        if rest.startswith('"""') and '"""' not in rest[3:]:
            # A multi-line string: replace the whole block, not just its first line.
            index += 1
            while index < len(lines) and '"""' not in lines[index]:
                index += 1
            index += 1
            out.append(f"{key.group(1)}{name} = {_fmt(value)}")
            continue
        out.append(f"{key.group(1)}{name} = {_fmt(value)}")
        index += 1

    for f in fields(cfg):
        if f.name == "source" or not is_dataclass(getattr(cfg, f.name)):
            continue
        section_obj = getattr(cfg, f.name)
        missing = [sf for sf in fields(section_obj) if (f.name, sf.name) not in written]
        if not missing:
            continue
        block = [f"{sf.name} = {_fmt(getattr(section_obj, sf.name))}" for sf in missing]
        at = _section_end(out, f.name)
        if at is None:
            out.extend(["", f"[{f.name}]", *block])
        else:
            out[at:at] = block
    return "\n".join(out) + "\n"


def _section_end(lines: list[str], name: str) -> int | None:
    """The index just past the last setting line of ``[name]``."""
    start = None
    for i, line in enumerate(lines):
        header = _SECTION.match(line)
        if header and header.group(1) == name:
            start = i + 1
            break
    if start is None:
        return None
    end = start
    for i in range(start, len(lines)):
        if _SECTION.match(lines[i]):
            break
        if _KEY.match(lines[i]):
            end = i + 1
    return end


def save(cfg: Config, path: str | os.PathLike[str] | None = None) -> Path:
    default = cfg.source or (cfg.data_dir / "config.toml")
    target = Path(os.path.expanduser(str(path))) if path else default
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    target.write_text(_patch(existing, cfg) if existing else dumps(cfg), encoding="utf-8")
    cfg.source = target
    return target

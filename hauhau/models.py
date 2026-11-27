"""The model catalog: which .gguf files this machine has, and switching between them.

A local agent is only as good as the model under it, and the right model depends
on the task - a 1.7B for a quick journal question, a 4B for a multi-step shell
job. Switching therefore has to be one command, not an edit of ``config.toml``
and a restart.

Two things are remembered per file, in ``<data_dir>/models.json``:

* ``n_ctx`` - a 1.7B with cheap KV cache can afford 32k while a 4B cannot, so
  carrying one context length across models either wastes RAM or rejects
  prompts. Whatever the tuner last adopted for a file is restored with it.
* ``extra_args`` - server flags that helped *this* model (a different cache type,
  a rope setting) are usually wrong for another.

The registry is a sidecar rather than a second config table because
``config.toml`` is schema-validated against the dataclasses: a free-form table of
models would fail type checks or be dropped on save.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import GGUF_SEARCH_DIRS, Config

# Quantisation and container suffixes that say nothing about *which* model it is.
# Matched against the whole stem before tokenising, because "Q4_K_M" is three
# tokens once "-" and "_" split it.
_QUANT_TAIL = re.compile(
    r"(?i)[-_.](?:q\d+(?:[-_][a-z0-9]+)*|iq\d+(?:[-_][a-z0-9]+)*|bf16|f16|f32|fp16|nvfp4)$"
)
_NOISE = {"gguf", "uncensored", "instruct", "it", "chat", "base", "hf", "model"}


class ModelError(Exception):
    """A model name that does not match, or matches more than one file."""


@dataclass(frozen=True)
class Entry:
    path: Path
    name: str
    size: int
    active: bool

    @property
    def size_text(self) -> str:
        gb = self.size / (1024**3)
        return f"{gb:.1f} GB" if gb >= 0.1 else f"{self.size / 1024**2:.0f} MB"


def _stem_tokens(stem: str) -> list[str]:
    trimmed = _QUANT_TAIL.sub("", stem)
    # Only "-" and "_" separate fields: a "." is part of a version ("1.7B"), and
    # splitting on it produced "LFM3-1-7B" in the header.
    tokens = [t for t in re.split(r"[-_]", trimmed) if t]
    while tokens and tokens[-1].lower() in _NOISE:
        tokens.pop()
    return tokens


def display_name(stem: str, width: int = 26) -> str:
    tokens = _stem_tokens(stem)
    if not tokens:
        return stem[:width]
    name = tokens[0]
    for token in tokens[1:]:
        if len(name) + len(token) + 1 > width:
            break
        name = f"{name}-{token}"
    if len(name) > width:
        name = name[: width - 1] + "…"
    return name


def _search_dirs(cfg: Config) -> list[Path]:
    dirs = [Path(os.path.expanduser(entry)) for entry in GGUF_SEARCH_DIRS]
    current = cfg.model.gguf
    if current:
        dirs.insert(0, Path(os.path.expanduser(current)).parent)
    seen: list[Path] = []
    for directory in dirs:
        if directory not in seen:
            seen.append(directory)
    return seen


def catalog(cfg: Config) -> list[Entry]:
    """Every .gguf reachable through the configured search directories."""
    active = Path(os.path.expanduser(cfg.model.gguf)).resolve() if cfg.model.gguf else None
    found: dict[Path, Entry] = {}
    for directory in _search_dirs(cfg):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.gguf")):
            try:
                real = path.resolve()
                size = path.stat().st_size
            except OSError:
                continue  # a dangling symlink or a file we may not read
            if real in found:
                continue
            found[real] = Entry(
                path=real,
                name=display_name(path.stem),
                size=size,
                active=real == active,
            )
    # Two files can shorten to the same name; number them so /model 2 is
    # unambiguous.
    entries = sorted(found.values(), key=lambda e: e.name.lower())
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.name] = counts.get(entry.name, 0) + 1
    return [
        Entry(
            path=e.path,
            name=f"{e.name}#{index}" if counts[e.name] > 1 else e.name,
            size=e.size,
            active=e.active,
        )
        for index, e in enumerate(entries, start=1)
    ]


def resolve(cfg: Config, query: str) -> Entry:
    """Find one entry by number, file name, path or unique fragment."""
    text = query.strip()
    if not text:
        raise ModelError("give a model number, name or path")
    entries = catalog(cfg)
    if not entries:
        raise ModelError(
            "no .gguf files found in " + ", ".join(str(d) for d in _search_dirs(cfg))
        )

    for entry in entries:  # exact display name first
        if entry.name.lower() == text.lower():
            return entry
    for entry in entries:  # full path or bare file name
        if str(entry.path) == str(Path(os.path.expanduser(text))) or entry.path.name == text:
            return entry
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(entries):
            return entries[index - 1]
        raise ModelError(f"{index} is out of range; there are {len(entries)} models")

    matches = [e for e in entries if text.lower() in e.path.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        hint = ", ".join(e.name for e in entries)
        raise ModelError(f"no model matches {text!r}; available: {hint}")
    hint = ", ".join(e.name for e in matches)
    raise ModelError(f"{text!r} matches {len(matches)} files: {hint}")


# --- the sidecar ---------------------------------------------------------
def _state_path(cfg: Config) -> Path:
    return cfg.data_dir / "models.json"


def remembered(cfg: Config) -> dict[str, dict]:
    path = _state_path(cfg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def remember(cfg: Config, path: str | os.PathLike[str]) -> None:
    """Record the settings that work for the model currently configured."""
    key = str(Path(os.path.expanduser(str(path))).resolve())
    state = remembered(cfg)
    state[key] = {
        "n_ctx": int(cfg.model.n_ctx),
        "extra_args": list(cfg.model.extra_args),
    }
    target = _state_path(cfg)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return  # a read-only home dir must not break model switching


def recall(cfg: Config, path: str | os.PathLike[str]) -> dict | None:
    key = str(Path(os.path.expanduser(str(path))).resolve())
    return remembered(cfg).get(key)


def switch(cfg: Config, entry: Entry, *, ctx_floor: int | None = None) -> tuple[int, int]:
    """Point the config at ``entry``; returns ``(old_n_ctx, new_n_ctx)``.

    The caller restarts the backend and saves the config - this only mutates the
    in-memory config, so a failed restart leaves the file untouched.
    """
    previous_ctx = int(cfg.model.n_ctx)
    previous = cfg.model.gguf
    if previous and Path(os.path.expanduser(previous)).resolve() != entry.path:
        remember(cfg, previous)
    cfg.model.gguf = str(entry.path)
    saved = recall(cfg, entry.path) or {}
    ctx = saved.get("n_ctx")
    if isinstance(ctx, int) and ctx > 0:
        if ctx_floor is not None and ctx < ctx_floor:
            ctx = ctx_floor
        cfg.model.n_ctx = int(ctx)
    args = saved.get("extra_args")
    if isinstance(args, list) and all(isinstance(a, str) for a in args):
        cfg.model.extra_args = list(args)
    return previous_ctx, int(cfg.model.n_ctx)

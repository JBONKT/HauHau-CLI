"""Agent skills: SKILL.md directories discovered at startup, loaded on demand.

Follows the open Agent Skills convention (agentskills.io): a skill is a folder
whose entrypoint is ``SKILL.md`` - YAML frontmatter with ``name`` and
``description``, then a markdown body. Progressive disclosure keeps the resident
context tiny: only name and description enter the system prompt, the body is
returned by the ``skill`` tool when the model judges the skill relevant, and
scripts bundled in the skill folder run through the ordinary ``shell`` tool so
only their output enters the conversation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class SkillError(ValueError):
    """A SKILL.md that cannot be used, described for the maintainer."""


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path  # the SKILL.md file; its folder holds any bundled scripts


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---``-fenced frontmatter from the body, one scalar per key.

    The spec's required fields are plain scalars; unknown keys and multi-line
    values from other products are ignored rather than rejected, so a skill
    written for Claude Code, Codex or Vibe loads here unchanged.
    """
    if not text.startswith("---"):
        raise SkillError("SKILL.md must start with a '---' frontmatter fence")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError("SKILL.md frontmatter is not closed with a second '---'")
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, parts[2].strip("\n")


def parse_skill(text: str, path: Path) -> Skill:
    meta, body = _frontmatter(text)
    name = meta.get("name", "").strip()
    description = meta.get("description", "").strip()
    if not _SLUG.match(name):
        raise SkillError(f"name {name!r} must be a lowercase slug (letters, digits, hyphens)")
    if not description:
        raise SkillError("frontmatter needs a non-empty 'description'")
    folder = path.parent.name
    if name != folder:
        raise SkillError(f"name {name!r} does not match its folder {folder!r}")
    return Skill(name=name, description=description, body=body, path=path)


def load_skill(skill_dir: Path) -> Skill:
    """Read one skill folder, or raise SkillError naming the problem."""
    entry = skill_dir / "SKILL.md"
    if not entry.is_file():
        raise SkillError(f"{skill_dir} has no SKILL.md")
    try:
        text = entry.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError(f"cannot read {entry}: {exc}") from exc
    return parse_skill(text, entry)


def builtin_dir() -> Path:
    """Skills shipped inside the package, below any user skill of the same name."""
    return Path(__file__).resolve().parent / "builtin_skills"


def skill_dirs(cfg) -> list[Path]:
    """Where to look, highest priority first: extra dirs, then the user dir,
    then the bundled skills. The first folder with a given name wins, so a copy
    in ~/.hauhau/skills overrides a built-in."""
    dirs = [Path(os.path.expanduser(d)) for d in getattr(cfg.skills, "dirs", [])]
    dirs.append(cfg.data_dir / "skills")
    dirs.append(builtin_dir())
    return dirs


def discover(dirs: list[Path]) -> tuple[list[Skill], list[str]]:
    """Scan skill folders. Returns (skills, warnings); a broken SKILL.md is a
    warning, never a startup failure - one bad file must not hide the rest."""
    skills: list[Skill] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for base in dirs:
        if not base.is_dir():
            continue
        try:
            entries = sorted(base.iterdir())
        except OSError as exc:
            warnings.append(f"cannot list {base}: {exc}")
            continue
        for folder in entries:
            if not folder.is_dir() or folder.name.startswith(("_", ".")):
                continue
            if folder.name in seen:
                continue
            try:
                skill = load_skill(folder)
            except SkillError as exc:
                warnings.append(f"skill {folder.name}: {exc}")
                continue
            seen.add(skill.name)
            skills.append(skill)
    return skills, warnings


def prompt_block(skills: list[Skill]) -> str:
    """The resident context: one line per skill, nothing more."""
    if not skills:
        return ""
    lines = [
        "Skills: procedures you can load in full with the skill tool when one "
        "matches the task. Loading one is cheap; guessing the procedure is not."
    ]
    lines.extend(f"- {s.name}: {s.description}" for s in skills)
    return "\n".join(lines)

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

import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

# The resident prompt carries one line per skill, so a description copied from a
# long document must not push the conversation out of a small context window.
DESCRIPTION_CHARS = 180
MAX_LISTED = 40


class SkillError(ValueError):
    """A SKILL.md that cannot be used, described for the maintainer."""


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path  # the SKILL.md file; its folder holds any bundled scripts
    allowed_tools: tuple[str, ...] = field(default=())

    @property
    def folder(self) -> Path:
        return self.path.parent

    def summary(self, width: int = DESCRIPTION_CHARS) -> str:
        """One prompt line's worth of the description."""
        line = " ".join(self.description.split())
        return line if len(line) <= width else line[: width - 1].rstrip() + "…"


def _scalar(value: str) -> str:
    return value.strip().strip("\"'").strip()


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---``-fenced frontmatter from the body.

    The spec's required fields are plain scalars, but real skills written for
    other products use ``>``/``|`` block scalars and quoted lists, and a skill
    that loads elsewhere should load here unchanged. Unknown keys are ignored
    rather than rejected.
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError("SKILL.md must start with a '---' frontmatter fence")
    # A line-based scan, not a split on '---': a value or a body containing
    # '---' used to end the frontmatter early and mangle both halves.
    closing = next((i for i, line in enumerate(lines) if i > 0 and line.strip() == "---"), None)
    if closing is None:
        raise SkillError("SKILL.md frontmatter is not closed with a second '---'")
    fence_lines, body = lines[1:closing], "\n".join(lines[closing + 1 :]).strip("\n")

    meta: dict[str, str] = {}
    key = ""
    block: list[str] = []
    style = ""  # ">" folds, "|" keeps line breaks

    def flush() -> None:
        nonlocal key, block, style
        if key:
            if style == ">":
                meta[key] = " ".join(part.strip() for part in block if part.strip())
            elif style == "|":
                meta[key] = "\n".join(block)
            else:
                meta[key] = _scalar(" ".join(part.strip() for part in block))
        key, block, style = "", [], ""

    for raw in fence_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1] in (" ", "\t") and key:
            # A continuation line belongs to the key above it.
            block.append(stripped)
            continue
        name, sep, value = raw.partition(":")
        if not sep:
            continue
        flush()
        key = name.strip()
        value = value.strip()
        if value in (">", ">+", ">-", "|", "|+", "|-"):
            style = value[0]
        elif value:
            meta[key] = _scalar(value)
            key = ""
    flush()
    return meta, body


def _tool_list(value: str) -> tuple[str, ...]:
    """``allowed-tools`` as names: JSON, YAML or comma separated, all tolerated."""
    text = value.strip().lstrip("[").rstrip("]")
    # A YAML block list puts "- read_file" on its own line; splitting on
    # whitespace left a tool literally named "-" in the header.
    names: list[str] = []
    for part in re.split(r"[,\s]+", text):
        name = part.strip().strip("\"'").lstrip("-").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def parse_skill(text: str, path: Path) -> Skill:
    meta, body = _frontmatter(text)
    name = meta.get("name", "").strip()
    description = " ".join(meta.get("description", "").split())
    if not _SLUG.match(name):
        raise SkillError(f"name {name!r} must be a lowercase slug (letters, digits, hyphens)")
    if not description:
        raise SkillError("frontmatter needs a non-empty 'description'")
    folder = path.parent.name
    if name != folder:
        raise SkillError(f"name {name!r} does not match its folder {folder!r}")
    return Skill(
        name=name,
        description=description,
        body=body,
        path=path,
        allowed_tools=_tool_list(meta.get("allowed-tools", "")),
    )


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


def index(skills: list[Skill]) -> dict[str, Skill]:
    return {skill.name: skill for skill in skills}


def for_tool(skills: list[Skill], tool: str) -> list[Skill]:
    """Skills that declare this tool.

    A failing tool is the moment the model most needs the installed procedure
    for it, and ``allowed-tools`` is exactly the declaration that says which
    skill that is.
    """
    return [skill for skill in skills if tool in skill.allowed_tools]


def nearest(name: str, skills: list[Skill]) -> str:
    """Best guess for a mistyped skill name, or '' when nothing is close."""
    matches = difflib.get_close_matches(name, [s.name for s in skills], n=1, cutoff=0.6)
    return matches[0] if matches else ""


def prompt_block(skills: list[Skill]) -> str:
    """The resident context: one line per skill, nothing more.

    The instruction line matters more than the list: a small model tends to
    improvise a procedure it has never seen and only then look for the skill.
    """
    if not skills:
        return ""
    lines = [
        "Skills: procedures installed on this machine. When a task matches one, load it with "
        "the skill tool *before* acting - the procedure beats improvising."
    ]
    shown = skills[:MAX_LISTED]
    lines.extend(f"- {s.name}: {s.summary()}" for s in shown)
    hidden = len(skills) - len(shown)
    if hidden:
        lines.append(f"- ({hidden} more installed; /skills lists every one)")
    return "\n".join(lines)

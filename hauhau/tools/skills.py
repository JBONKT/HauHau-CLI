"""The ``skill`` tool: progressive disclosure, level two.

The system prompt carries only each skill's name and description; this tool
hands the model the body of one skill, plus any support file bundled with it
(such as a script), when the task calls for it.
"""

from __future__ import annotations

from ..skills import discover, index, nearest, skill_dirs
from ..util import clip_text
from .args import text
from .base import Tool, ToolResult, fail, ok

# A body is capped well under the 5k-token spec ceiling: a 3B model with a 16k
# context loses the middle of long instructions first.
_MAX_BODY_CHARS = 12000
_LIST_LIMIT = 40


def build(cfg, audit, skills) -> list[Tool]:
    if not skills:
        return []
    known = index(skills)
    # A skill dropped into ~/.hauhau/skills while the app is running is invisible
    # to the prompt until the next start. One rescan on a miss is cheap and turns
    # "I have no such skill" into a working call.
    rescan = {"done": False}

    def refresh() -> bool:
        if rescan["done"]:
            return False
        rescan["done"] = True
        try:
            found, _warnings = discover(skill_dirs(cfg))
        except Exception:  # a broken skills folder must not break the tool
            return False
        if not found:
            return False
        known.update(index(found))
        return True

    def _read(skill, rel: str) -> str:
        """One support file inside the skill folder, or an error that says so.

        ``..`` and absolute paths are refused: a skill folder is data, not a
        window onto the whole disk.
        """
        target = (skill.folder / rel).resolve()
        root = skill.folder.resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError(f"'file' must be a path inside the skill folder, not {rel!r}")
        if target.is_dir():
            listing = sorted(p.relative_to(root).as_posix() for p in target.rglob("*") if p.is_file())
            if not listing:
                raise ValueError(f"{rel!r} is an empty folder in skill {skill.name}")
            shown = "\n".join(f"- {name}" for name in listing[:_LIST_LIMIT])
            more = f"\n({len(listing) - _LIST_LIMIT} more)" if len(listing) > _LIST_LIMIT else ""
            return f"files in {rel!r}:\n{shown}{more}"
        if not target.is_file():
            raise ValueError(f"no file {rel!r} in skill {skill.name}")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{rel!r} in skill {skill.name} is not text ({target.stat().st_size} bytes); "
                "run it with the shell tool instead of reading it"
            ) from exc

    def load(args: dict) -> ToolResult:
        name = text(args, "name", required=True).strip().lstrip("/")
        skill = known.get(name)
        if skill is None and refresh():
            skill = known.get(name)
        if skill is None:
            close = nearest(name, list(known.values()))
            hint = f" Did you mean {close!r}?" if close else ""
            # The list is capped: with many skills installed this failure was
            # itself a context flood.
            names = sorted(known)
            shown = ", ".join(names[:20])
            if len(names) > 20:
                shown += f", and {len(names) - 20} more"
            audit.log("skill.miss", name=name, suggestion=close or None)
            return fail(f"skill: no skill named {name!r}.{hint} Known skills: {shown}")
        rel = text(args, "file").strip()
        try:
            content = skill.body if not rel else _read(skill, rel)
        except (OSError, ValueError) as exc:
            audit.log("skill.error", name=skill.name, file=rel or None, error=str(exc))
            return fail(f"skill {skill.name}: {exc}")
        if not content.strip():
            return fail(f"skill {skill.name}: {'that file is' if rel else 'the skill body is'} empty")
        audit.log("skill_load", name=skill.name, file=rel or None, chars=len(content))
        header = f"# skill: {skill.name}" if not rel else f"# skill: {skill.name} - {rel}"
        if skill.allowed_tools and not rel:
            header += f" (needs: {', '.join(skill.allowed_tools)})"
        return ok(clip_text(f"{header}\n\n{content}", _MAX_BODY_CHARS))

    return [
        Tool(
            name="skill",
            description=(
                "Load the full procedure of a skill listed in the system prompt, and follow it "
                "instead of improvising. Optional 'file' reads (or, for a folder, lists) a support "
                "file inside the skill folder."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "skill name from the system prompt"},
                    "file": {
                        "type": "string",
                        "description": "relative path of a file inside the skill folder, optional",
                    },
                },
                "required": ["name"],
            },
            fn=load,
        )
    ]

"""The ``skill`` tool: progressive disclosure, level two.

The system prompt carries only each skill's name and description; this tool
hands the model the body of one skill, plus any support file bundled with it
(such as a script), when the task calls for it.
"""

from __future__ import annotations

from ..util import clip_text
from .args import integer, text
from .base import Tool, ToolResult, fail, ok

# A body is capped well under the 5k-token spec ceiling: a 3B model with a 16k
# context loses the middle of long instructions first.
_MAX_BODY_CHARS = 12000


def build(cfg, audit, skills) -> list[Tool]:
    if not skills:
        return []
    by_name = {s.name: s for s in skills}

    def _read(skill, rel: str) -> str:
        """One support file inside the skill folder, or an error that says so.

        ``..`` and absolute paths are refused: a skill folder is data, not a
        window onto the whole disk.
        """
        target = (skill.path.parent / rel).resolve()
        root = skill.path.parent.resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError(f"'file' must be a path inside the skill folder, not {rel!r}")
        if not target.is_file():
            raise ValueError(f"no file {rel!r} in skill {skill.name}")
        return target.read_text(encoding="utf-8")

    def load(args: dict) -> ToolResult:
        name = text(args, "name", required=True).strip()
        skill = by_name.get(name)
        if skill is None:
            known = ", ".join(sorted(by_name))
            return fail(f"skill: no skill named {name!r}. Known skills: {known}")
        rel = text(args, "file").strip()
        try:
            content = skill.body if not rel else _read(skill, rel)
        except (OSError, ValueError) as exc:
            return fail(f"skill {skill.name}: {exc}")
        k = integer(args, "max_chars", _MAX_BODY_CHARS, minimum=500, maximum=_MAX_BODY_CHARS)
        audit.log("skill_load", name=skill.name, file=rel or None, chars=len(content))
        return ok(clip_text(content, k))

    return [
        Tool(
            name="skill",
            description=(
                "Load a skill's instructions. The system prompt lists the known "
                "skills by name with a one-line description; pass the one that "
                "matches the task to get its full procedure. Optional 'file' "
                "reads a support file bundled in the skill folder."
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

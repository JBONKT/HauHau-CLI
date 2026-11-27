from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hauhau.audit import Audit
from hauhau.config import Config
from hauhau.skills import (
    Skill,
    builtin_dir,
    discover,
    load_skill,
    parse_skill,
    prompt_block,
    skill_dirs,
)
from hauhau.tools import build_registry
from hauhau.tools import ui as ui_module

MD = """---
name: test-skill
description: Does one thing well, when the task calls for it.
allowed-tools: ["shell"]
---

# test-skill

Step 1. Do the thing.
"""


def _write_skill(root: Path, name: str, description: str = "does a thing", body: str = "do it") -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8"
    )
    return folder


class TestParse(unittest.TestCase):
    def test_valid(self) -> None:
        skill = parse_skill(MD, Path("/skills/test-skill/SKILL.md"))
        self.assertEqual(skill.name, "test-skill")
        self.assertEqual(skill.description, "Does one thing well, when the task calls for it.")
        self.assertIn("Step 1.", skill.body)

    def test_unknown_keys_ignored(self) -> None:
        # A skill written for another agent (Claude, Vibe, Codex) loads unchanged.
        text = MD.replace('allowed-tools: ["shell"]', 'when_to_use: always\nmetadata-version: 3')
        skill = parse_skill(text, Path("/x/test-skill/SKILL.md"))
        self.assertEqual(skill.name, "test-skill")

    def test_name_must_match_folder(self) -> None:
        with self.assertRaises(ValueError):
            parse_skill(MD, Path("/skills/other/SKILL.md"))

    def test_bad_slug(self) -> None:
        text = MD.replace("name: test-skill", "name: Test Skill")
        with self.assertRaises(ValueError):
            parse_skill(text, Path("/x/Test Skill/SKILL.md"))

    def test_missing_description(self) -> None:
        text = MD.replace("description: Does one thing well, when the task calls for it.", "")
        with self.assertRaises(ValueError):
            parse_skill(text, Path("/x/test-skill/SKILL.md"))

    def test_no_frontmatter(self) -> None:
        with self.assertRaises(ValueError):
            parse_skill("just a body", Path("/x/test-skill/SKILL.md"))


class TestDiscover(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_discover_order_and_override(self) -> None:
        user = self.root / "user"
        extra = self.root / "extra"
        _write_skill(user, "mine", "user copy")
        _write_skill(extra, "shared", "extra copy wins")
        _write_skill(user, "shared", "user copy loses to extra")
        skills, warnings = discover([extra, user, Path(self.root / "nope")])
        self.assertEqual(warnings, [])
        self.assertEqual([s.name for s in skills], ["shared", "mine"])
        self.assertEqual(skills[0].description, "extra copy wins")

    def test_broken_skill_is_warning_not_failure(self) -> None:
        folder = _write_skill(self.root, "broken", "half a skill")
        (folder / "SKILL.md").write_text("---\nname: broken\n---\nno description", encoding="utf-8")
        skills, warnings = discover([self.root])
        self.assertEqual(skills, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("broken", warnings[0])

    def test_underscore_folders_skipped(self) -> None:
        (self.root / "_drafts").mkdir()
        _write_skill(self.root / "_drafts", "hidden", "not a skill yet")
        skills, warnings = discover([self.root])
        self.assertEqual((skills, warnings), ([], []))

    def test_builtin_skills_load(self) -> None:
        skills, warnings = discover([builtin_dir()])
        self.assertEqual(warnings, [])
        names = [s.name for s in skills]
        self.assertIn("app-actions", names)
        self.assertIn("dialog-handler", names)
        for skill in skills:
            self.assertTrue(skill.body, f"{skill.name} has an empty body")
            self.assertLess(len(skill.body), 12000, f"{skill.name} body is huge for a 3B model")

    def test_dirs_from_config(self) -> None:
        cfg = Config()
        cfg.logging.data_dir = str(self.root / "data")
        cfg.skills.dirs = [str(self.root / "extra")]
        dirs = skill_dirs(cfg)
        self.assertEqual(dirs[0], self.root / "extra")
        self.assertEqual(dirs[1], cfg.data_dir / "skills")
        self.assertEqual(dirs[-1], builtin_dir())


class TestSkillTool(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.audit = Audit(self.cfg.audit_path, "skills-test")
        self.skill = load_skill(_write_skill(self.root, "demo", "a demo", "body text\nmore"))
        helper = self.root / "demo" / "scripts"
        helper.mkdir()
        (helper / "run.sh").write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        self.registry = build_registry(self.cfg, self.audit, skills=[self.skill])

    def tearDown(self) -> None:
        self.audit.close()
        self._tmp.cleanup()

    def run_tool(self, tool_name, **args):
        tool = self.registry.get(tool_name)
        assert tool is not None, tool_name
        return tool.run(args)

    def test_unknown_skill_names_the_known_ones(self) -> None:
        result = self.run_tool("skill", name="nope")
        self.assertFalse(result.ok)
        self.assertIn("demo", result.content)

    def test_load_body(self) -> None:
        result = self.run_tool("skill", name="demo")
        self.assertTrue(result.ok)
        self.assertIn("body text", result.content)

    def test_load_support_file(self) -> None:
        result = self.run_tool("skill", name="demo", file="scripts/run.sh")
        self.assertTrue(result.ok)
        self.assertIn("echo hi", result.content)

    def test_support_file_cannot_escape_the_folder(self) -> None:
        for bad in ("../SKILL.md", "scripts/../../SKILL.md", "nothing.txt"):
            result = self.run_tool("skill", name="demo", file=bad)
            self.assertFalse(result.ok, bad)

    def test_no_skills_no_tool(self) -> None:
        empty = build_registry(self.cfg, self.audit, skills=[])
        self.assertIsNone(empty.get("skill"))

    def test_registry_has_ui_tool(self) -> None:
        self.assertIsNotNone(self.registry.get("ui"))


class TestUiTool(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.audit = Audit(self.cfg.audit_path, "ui-test")
        self.registry = build_registry(self.cfg, self.audit)
        self._real = ui_module._run_osascript

    def tearDown(self) -> None:
        ui_module._run_osascript = self._real
        self.audit.close()
        self._tmp.cleanup()

    def run_tool(self, **args):
        tool = self.registry.get("ui")
        assert tool is not None
        return tool.run(args)

    def test_read_lists_elements(self) -> None:
        def fake(script, timeout):
            self.assertIn(json.dumps("Finder"), script)
            return 0, json.dumps(
                {"head": "window \" Untitled\"", "lines": ["[0] AXButton \"OK\""], "truncated": False}
            )

        ui_module._run_osascript = fake
        result = self.run_tool(action="read", app="Finder")
        self.assertTrue(result.ok)
        self.assertIn("AXButton \"OK\"", result.content)

    def test_press_uses_the_index(self) -> None:
        def fake(script, timeout):
            self.assertIn("e.i === 3", script)
            return 0, '{"pressed": "[3] AXButton \"OK\""}'

        ui_module._run_osascript = fake
        result = self.run_tool(action="press", app="Finder", index=3)
        self.assertTrue(result.ok)

    def test_set_passes_the_value(self) -> None:
        def fake(script, timeout):
            self.assertIn(json.dumps("report.pdf"), script)
            return 0, '{"set": "[1] AXTextField"}'

        ui_module._run_osascript = fake
        result = self.run_tool(action="set", app="Finder", index=1, value="report.pdf")
        self.assertTrue(result.ok)

    def test_permission_hint_names_the_missing_grant(self) -> None:
        self.assertIn("Accessibility", ui_module._permission_hint("not allowed assistive access"))
        self.assertIn("Automation", ui_module._permission_hint("osascript: error -1743"))
        self.assertEqual(ui_module._permission_hint("syntax error near line 1"), "")

    def test_no_osascript_is_an_actionable_failure(self) -> None:
        ui_module._run_osascript = lambda script, timeout: (127, "osascript not found - the ui tool needs macOS")
        result = self.run_tool(action="read", app="Finder")
        self.assertFalse(result.ok)
        self.assertIn("macOS", result.content)

    def test_bad_action_rejected_before_any_subprocess(self) -> None:
        def fake(script, timeout):
            raise AssertionError("must not run")

        ui_module._run_osascript = fake
        result = self.run_tool(action="close", app="Finder")
        self.assertFalse(result.ok)

    def test_press_requires_index(self) -> None:
        result = self.run_tool(action="press", app="Finder")
        self.assertFalse(result.ok)
        self.assertIn("index", result.content)


class TestPromptBlock(unittest.TestCase):
    def test_lists_names_and_descriptions(self) -> None:
        block = prompt_block([Skill("a", "does a", "body", Path("/x/a/SKILL.md"))])
        self.assertIn("- a: does a", block)
        self.assertIn("skill tool", block)

    def test_empty_is_nothing(self) -> None:
        self.assertEqual(prompt_block([]), "")


if __name__ == "__main__":
    unittest.main()

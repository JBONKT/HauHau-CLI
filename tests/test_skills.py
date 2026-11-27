from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hauhau.audit import Audit
from hauhau.config import Config
from hauhau.skills import (
    MAX_LISTED,
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


class TestFrontmatterVariants(unittest.TestCase):
    """A skill written for another agent must load here unchanged."""

    def test_folded_block_scalar_description(self) -> None:
        text = (
            "---\nname: fold\ndescription: >\n  Does one thing\n  when the task calls for it.\n---\n\nbody\n"
        )
        skill = parse_skill(text, Path("/x/fold/SKILL.md"))
        self.assertEqual(skill.description, "Does one thing when the task calls for it.")

    def test_literal_scalar_is_parsed_and_collapsed_for_the_prompt(self) -> None:
        # A `|` block keeps its line breaks in the meta, but the prompt line is a
        # single line by design, so the description arrives with spaces.
        text = "---\nname: lit\ndescription: |\n  first\n  second\n---\nbody\n"
        self.assertEqual(parse_skill(text, Path("/x/lit/SKILL.md")).description, "first second")

    def test_crlf_and_bom_are_tolerated(self) -> None:
        text = "\ufeff---\r\nname: win\r\ndescription: a windows copy\r\n---\r\nbody\r\n"
        skill = parse_skill(text, Path("/x/win/SKILL.md"))
        self.assertEqual(skill.name, "win")
        self.assertIn("body", skill.body)

    def test_allowed_tools_in_three_spellings(self) -> None:
        for value, expected in (
            ('["shell", "ui"]', ("shell", "ui")),
            ("[shell, ui]", ("shell", "ui")),
            ("shell ui", ("shell", "ui")),
        ):
            text = f"---\nname: t\ndescription: d\nallowed-tools: {value}\n---\nbody\n"
            self.assertEqual(parse_skill(text, Path("/x/t/SKILL.md")).allowed_tools, expected, value)

    def test_no_allowed_tools_means_none(self) -> None:
        self.assertEqual(parse_skill(MD, Path("/skills/test-skill/SKILL.md")).allowed_tools, ("shell",))


class TestAgentPrompt(unittest.TestCase):
    """The bot has to know the skills exist before it can master them."""

    def _agent(self, skills):
        from hauhau.agent import Agent
        from hauhau.tools.base import Registry

        cfg = Config()
        cfg.brain.enabled = False
        return Agent(cfg, None, Registry([]), brain=None, skills=skills)

    def test_the_system_prompt_lists_the_skills(self) -> None:
        skill = parse_skill(MD, Path("/skills/test-skill/SKILL.md"))
        prompt = self._agent([skill]).system_prompt("hi")
        self.assertIn("Skills: procedures installed on this machine", prompt)
        self.assertIn("- test-skill:", prompt)

    def test_no_skills_means_no_block(self) -> None:
        self.assertNotIn("Skills:", self._agent([]).system_prompt("hi"))


class TestSkillToolRefinements(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.cfg.skills.dirs = []
        self.audit = Audit(self.cfg.audit_path, "skills-test", True)
        self.skill = load_skill(_write_skill(self.root / "installed", "demo", "a demo", "body text"))
        self.registry = build_registry(self.cfg, self.audit, skills=[self.skill])

    def tearDown(self) -> None:
        self.audit.close()
        self._tmp.cleanup()

    def run_skill(self, **args):
        return self.registry.get("skill").run(args)

    def test_the_body_carries_a_header_with_its_tools(self) -> None:
        result = self.run_skill(name="demo")
        self.assertTrue(result.content.startswith("# skill: demo"))
        self.assertIn("body text", result.content)

    def test_a_mistyped_name_is_guessed(self) -> None:
        result = self.run_skill(name="dem")
        self.assertFalse(result.ok)
        self.assertIn("Did you mean 'demo'", result.content)

    def test_a_folder_is_listed_rather_than_read(self) -> None:
        (self.skill.folder / "scripts").mkdir()
        (self.skill.folder / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
        result = self.run_skill(name="demo", file="scripts")
        self.assertTrue(result.ok, result.content)
        self.assertIn("scripts/run.sh", result.content)

    def test_a_binary_support_file_points_at_the_shell(self) -> None:
        (self.skill.folder / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")
        result = self.run_skill(name="demo", file="logo.png")
        self.assertFalse(result.ok)
        self.assertIn("not text", result.content)
        self.assertIn("shell", result.content)

    def test_a_skill_added_after_startup_is_found(self) -> None:
        # The user drops a skill into ~/.hauhau/skills while the app is running.
        late = _write_skill(self.cfg.data_dir / "skills", "late", "arrived later", "late body")
        self.assertTrue((late / "SKILL.md").is_file())
        result = self.run_skill(name="late")
        self.assertTrue(result.ok, result.content)
        self.assertIn("late body", result.content)

    def test_a_miss_is_audited(self) -> None:
        self.run_skill(name="absent")
        events = [e.get("event") for e in self.audit.tail(20)]
        self.assertIn("skill.miss", events)

    def test_a_leading_slash_is_tolerated(self) -> None:
        self.assertTrue(self.run_skill(name="/demo").ok)


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

    def test_a_failure_names_the_skill_that_documents_it(self) -> None:
        from hauhau.skills import Skill

        skill = Skill("dialog-handler", "answers dialogs", "body", self.root / "d" / "SKILL.md",
                      allowed_tools=("ui",))
        registry = build_registry(self.cfg, self.audit, skills=[skill])
        ui_module._run_osascript = lambda script, timeout: (1, "not allowed assistive access")
        result = registry.get("ui").run({"action": "read", "app": "Finder"})
        self.assertFalse(result.ok)
        self.assertIn("dialog-handler", result.content)
        self.assertIn("load it with the skill tool", result.content)

    def test_without_such_a_skill_there_is_no_hint(self) -> None:
        ui_module._run_osascript = lambda script, timeout: (1, "not allowed assistive access")
        result = self.run_tool(action="read", app="Finder")
        self.assertFalse(result.ok)
        self.assertNotIn("skill tool", result.content)

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

    def test_block_tells_the_model_to_load_before_acting(self) -> None:
        block = prompt_block([Skill("a", "does a", "body", Path("/x/a/SKILL.md"))])
        self.assertIn("before* acting", block)

    def test_a_long_description_is_clipped_to_one_line(self) -> None:
        skill = Skill("a", "word " * 200, "body", Path("/x/a/SKILL.md"))
        line = [ln for ln in prompt_block([skill]).splitlines() if ln.startswith("- a:")]
        self.assertEqual(len(line), 1)
        self.assertLess(len(line[0]), 200)
        self.assertTrue(line[0].endswith("…"))

    def test_the_list_is_capped(self) -> None:
        skills = [
            Skill(f"s-{i:02d}", "d", "b", Path(f"/x/s-{i:02d}/SKILL.md"))
            for i in range(MAX_LISTED + 5)
        ]
        self.assertIn("5 more installed", prompt_block(skills))


class TestReviewFollowups(unittest.TestCase):
    """The gaps the review of f9bc8a3 found: index stability, fence parsing,
    the size of failure messages, and JXA that at least parses."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.audit = Audit(self.cfg.audit_path, "review-followups")
        self.registry = build_registry(self.cfg, self.audit)
        self._real = ui_module._run_osascript
        self.captured: list[str] = []

    def tearDown(self) -> None:
        ui_module._run_osascript = self._real
        self.audit.close()
        self._tmp.cleanup()

    def run_ui(self, **args):
        tool = self.registry.get("ui")
        assert tool is not None
        return tool.run(args)

    def _capture(self, reply: str) -> None:
        def fake(script, timeout):
            self.captured.append(script)
            return 0, reply

        ui_module._run_osascript = fake

    def _scripts_for_every_action(self) -> list[str]:
        self._capture('{"head": "w", "lines": ["[0] AXButton \\"OK\\""], "truncated": false}')
        assert self.run_ui(action="read", app="Finder").ok
        self._capture('{"pressed": "[3] AXButton"}')
        assert self.run_ui(action="press", app="Finder", index=3).ok
        self._capture('{"set": "[1] AXTextField"}')
        assert self.run_ui(action="set", app="Finder", index=1, value="report.pdf").ok
        return self.captured

    def test_walk_is_identical_across_actions(self) -> None:
        """An index from `read` must address the same element in `press`/`set`:
        the walk (depth included) is a fixed constant, not a per-call setting."""
        scripts = self._scripts_for_every_action()
        for script in scripts:
            self.assertIn("walk(win, WALK_DEPTH, out)", script)
        # No per-call depth survives anywhere, and the schema does not offer one.
        for script in scripts:
            self.assertNotIn("%(depth)d", script)
        schema = self.registry.get("ui").schema()["function"]["parameters"]["properties"]
        self.assertNotIn("depth", schema)
        # A depth argument from an old conversation is ignored, not an error.
        self._capture('{"head": "w", "lines": [], "truncated": false}')
        self.assertTrue(self.run_ui(action="read", app="Finder", depth=9).ok)

    def test_no_whose_frontmost_filter(self) -> None:
        """`frontmost` is a process property; the whose() call was dead code
        that silently fell through to windows[0]."""
        for script in self._scripts_for_every_action():
            self.assertNotIn("whose", script)
            self.assertIn("proc.windows[0]", script)

    @unittest.skipUnless(shutil.which("osascript"), "JXA syntax check needs macOS")
    def test_jxa_templates_parse_as_javascript(self) -> None:
        for script in self._scripts_for_every_action():
            wrapped = "(function(){\n" + script + "\n})"
            proc = subprocess.run(
                ["osascript", "-l", "JavaScript", "-e", wrapped],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, f"JXA does not parse: {proc.stderr}")

    def test_frontmatter_survives_dashes_in_a_value(self) -> None:
        text = (
            "---\nname: dash-skill\ndescription: handles a---b and more\n---\n\n"
            "body line\n---\nhorizontal rule\n"
        )
        skill = parse_skill(text, Path("/x/dash-skill/SKILL.md"))
        self.assertEqual(skill.description, "handles a---b and more")
        self.assertIn("horizontal rule", skill.body)

    def test_unclosed_frontmatter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_skill("---\nname: x\n", Path("/x/x/SKILL.md"))

    def test_known_list_is_capped_in_the_error(self) -> None:
        skills = [
            Skill(f"s-{i:02d}", "d", "body", Path(f"/x/s-{i:02d}/SKILL.md")) for i in range(25)
        ]
        registry = build_registry(self.cfg, self.audit, skills=skills)
        result = registry.get("skill").run({"name": "missing"})
        self.assertFalse(result.ok)
        # 25 fakes plus the bundled built-ins: the list shows a sample and a
        # count, not every name.
        self.assertRegex(result.content, r"and \d+ more")
        self.assertLess(len(result.content), 400)


class TestAllowedToolSpellings(unittest.TestCase):
    def test_a_yaml_block_list_yields_names(self) -> None:
        from hauhau.skills import _tool_list

        self.assertEqual(_tool_list("\n  - read_file\n  - shell"), ("read_file", "shell"))

    def test_other_spellings_still_work(self) -> None:
        from hauhau.skills import _tool_list

        self.assertEqual(_tool_list("[read_file, shell]"), ("read_file", "shell"))
        self.assertEqual(_tool_list("read_file, shell"), ("read_file", "shell"))
        self.assertEqual(_tool_list("read_file shell"), ("read_file", "shell"))


if __name__ == "__main__":
    unittest.main()

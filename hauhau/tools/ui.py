"""The ``ui`` tool: read and drive macOS app windows through the accessibility tree.

Design, per what small-model research says works: the model never sees a pixel
and never produces a coordinate. ``read`` lists the elements of an app's
frontmost window as an indexed outline (the accessibility API already knows the
geometry), and ``press``/``set`` address an element by that index - selection,
not regression. Everything runs through ``System Events``, so it needs the
Accessibility permission granted to the interpreter binary, and an Automation
grant for System Events itself; both failures surface as the osascript error,
which names the missing permission.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from ..util import clip_text
from .args import integer, text
from .base import Tool, ToolResult, fail, ok

# One shared walk, so the index a `read` printed is the index a later
# `press` lands on. Depth-first, parent before children, capped so a
# pathological window cannot flood the context.
_WALK = """
const MAXE = %(cap)d;
// One fixed walk depth for every action: an index printed by `read` must
// address the same element in a later `press`/`set`, or a dialog agent can
// click the button next to the one it meant.
const WALK_DEPTH = 3;
let count = 0;
function describe(el) {
  let role = "", name = "", value = "";
  try { role = el.role() || ""; } catch (e) {}
  try { const n = el.name(); if (n && n !== "missing value") name = String(n); } catch (e) {}
  try { const v = el.value(); if (v != null && v !== "missing value") value = String(v); } catch (e) {}
  let line = role;
  if (name) line += " " + JSON.stringify(name);
  if (value) line += " = " + JSON.stringify(value.slice(0, 80));
  return line;
}
function walk(el, depth, out) {
  let kids = [];
  try { kids = el.uiElements(); } catch (e) { kids = []; }
  for (const kid of kids) {
    if (count >= MAXE) return;
    const entry = { i: count, d: describe(kid), el: kid };
    out.push(entry);
    count++;
    if (depth > 0) walk(kid, depth - 1, out);
  }
}
"""

_READ = (
    _WALK
    + """
function run() {
  const se = Application("System Events");
  const proc = se.processes.byName(%(app)s);
  let win;
  try {
    win = proc.windows[0];  // System Events lists a process's windows front first
  } catch (e) {
    throw new Error("no accessible window for process " + %(app)s + ": " + e.message);
  }
  const out = [];
  let title = "";
  try { title = win.name() || ""; } catch (e) {}
  const head = "window " + JSON.stringify(title);
  walk(win, WALK_DEPTH, out);
  const lines = out.map((e) => "[" + e.i + "] " + e.d);
  return JSON.stringify({ head: head, lines: lines, truncated: count >= MAXE });
}
"""
)

_PRESS = (
    _WALK
    + """
function run() {
  const se = Application("System Events");
  const proc = se.processes.byName(%(app)s);
  let win;
  try {
    win = proc.windows[0];  // System Events lists a process's windows front first
  } catch (e) {
    throw new Error("no accessible window for process " + %(app)s + ": " + e.message);
  }
  const out = [];
  walk(win, WALK_DEPTH, out);
  const target = out.filter((e) => e.i === %(index)d)[0];
  if (!target) throw new Error("index %(index)d not found; ui read again, the window changed");
  try {
    target.el.click();
  } catch (e) {
    throw new Error("cannot press [" + target.i + "] " + target.d + ": " + e.message);
  }
  return JSON.stringify({ pressed: "[" + target.i + "] " + target.d });
}
"""
)

_SET = (
    _WALK
    + """
function run() {
  const se = Application("System Events");
  const proc = se.processes.byName(%(app)s);
  let win;
  try {
    win = proc.windows[0];  // System Events lists a process's windows front first
  } catch (e) {
    throw new Error("no accessible window for process " + %(app)s + ": " + e.message);
  }
  const out = [];
  walk(win, WALK_DEPTH, out);
  const target = out.filter((e) => e.i === %(index)d)[0];
  if (!target) throw new Error("index %(index)d not found; ui read again, the window changed");
  try {
    target.el.value = %(value)s;
  } catch (e) {
    throw new Error("cannot set [" + target.i + "] " + target.d + ": " + e.message);
  }
  return JSON.stringify({ set: "[" + target.i + "] " + target.d, to: %(value)s });
}
"""
)


def _permission_hint(err: str) -> str:
    """The TCC grant an osascript error is really asking for, if it is one."""
    low = err.lower()
    if "assistive" in low or "accessibility" in low or "-25211" in err or "-1719" in err:
        return " (grant Accessibility to this python binary, System Settings > Privacy & Security)"
    if "not allowed" in low or "-1743" in err:
        return " (grant Automation control of System Events to this python binary)"
    return ""


def _run_osascript(script: str, timeout: float) -> tuple[int, str]:
    if shutil.which("osascript") is None:
        return 127, "osascript not found - the ui tool needs macOS"
    try:
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"osascript timed out after {timeout:.0f}s"
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        return proc.returncode, f"osascript: {err or 'failed'}{_permission_hint(err)}"
    return 0, out


def build(cfg, audit, skills=None) -> list[Tool]:
    timeout = cfg.tools.shell_timeout
    # A failure here is usually a dialog the model is trying to click blind. The
    # skill that documents the read-decide-act loop is one call away, so say so
    # in the error rather than letting it improvise four more attempts.
    documented = [skill.name for skill in (skills or []) if "ui" in skill.allowed_tools]
    hint = (
        f"\nInstalled skill {documented[0]!r} has the procedure for this - load it with the "
        "skill tool before retrying."
        if documented
        else ""
    )

    def _fail(message: str) -> ToolResult:
        return fail(message + hint)

    def _script(action: str, app: str, cap: int, index: int | None, value: str | None) -> str:
        params = {
            "app": json.dumps(app),
            "index": 0 if index is None else index,
            "value": json.dumps(value or ""),
            "cap": cap,
        }
        template = {"read": _READ, "press": _PRESS, "set": _SET}[action]
        return template % params

    def ui(args: dict) -> ToolResult:
        action = text(args, "action", "read").strip().lower()
        if action not in ("read", "press", "set"):
            return fail("ui: 'action' must be read, press or set")
        app = text(args, "app", required=True).strip()
        if not app:
            return fail("ui: 'app' (the process name, as the menu bar shows) is required")
        cap = integer(args, "max", 150, minimum=10, maximum=300)
        index = None
        value = None
        if action in ("press", "set"):
            if args.get("index") is None:
                return fail("ui: 'index' (the element index from a previous read) is required")
            index = integer(args, "index", 0, minimum=0, maximum=999)
        if action == "set":
            value = text(args, "value", required=True)
        audit.log("ui", action=action, app=app, index=index)
        code, out = _run_osascript(_script(action, app, cap, index, value), timeout)
        if code != 0:
            return _fail(f"ui {action} on {app!r}: {out}")
        if action == "read":
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                return _fail(f"ui read: unreadable reply: {out[:200]}")
            lines = data.get("lines") or []
            if not lines:
                return ok(f"{data.get('head', 'window')} has no readable UI elements")
            body = "\n".join([data.get("head", "window"), *lines])
            if data.get("truncated"):
                body += f"\n(list truncated at {cap} elements; re-read with a higher 'max')"
            return ok(clip_text(body, cfg.tools.max_output_chars))
        return ok(f"{action} ok: {out}")

    return [
        Tool(
            name="ui",
            description=(
                "Read or drive a macOS app's frontmost window via the accessibility tree "
                "(no screenshots, no coordinates). 'read' lists elements as [index] role "
                "\"name\" = value; 'press' clicks element index; 'set' types value into it. "
                "Re-read after each action to verify."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "press", "set"]},
                    "app": {"type": "string", "description": "process name shown in the menu bar"},
                    "index": {"type": "integer", "description": "element index from a previous read"},
                    "value": {"type": "string", "description": "new value, for set"},
                    "max": {"type": "integer", "description": "element cap for read, default 150"},
                },
                "required": ["action", "app"],
            },
            fn=ui,
        )
    ]

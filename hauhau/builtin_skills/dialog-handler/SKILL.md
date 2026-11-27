---
name: dialog-handler
description: Answer or dismiss a dialog, sheet or popup blocking a running macOS app - save dialogs, permission prompts, "are you sure" boxes. Use when an app waits for a click the shell cannot reach.
allowed-tools: ["ui"]
---

# dialog-handler

A dialog is a small window with a few labelled elements. The `ui` tool reads it
through the accessibility tree - no screenshots, no coordinates - and you pick
an element by the index it printed.

## Procedure

1. Read the app's frontmost window. The process name is what the menu bar
   shows next to the Apple menu when the app is front (often the app name
   without spaces, e.g. "Finder").

   ```json
   {"action": "read", "app": "AppName"}
   ```

2. The reply lists elements like `[3] AXButton "Save"` and `[4] AXStaticText
   "Where do you want to save it?"`. Read the texts to understand the dialog,
   then pick the button.
3. Press it by index:

   ```json
   {"action": "press", "app": "AppName", "index": 3}
   ```

4. Verify: read again. If the dialog is gone, the window's elements change
   (often to the app's main UI). If it is still there, the click did not take -
   choose a different index; never repeat the same press.

## Setting a value

For a "save as" name or a text prompt, set the field first, then press:

```json
{"action": "set", "app": "AppName", "index": 1, "value": "report.pdf"}
```

## Rules

- One action at a time: read, decide, act, re-read.
- Destructive buttons ("Delete", "Don't Save", "Erase") need the user's
  explicit go-ahead first. Everything else is safe to click.
- If `read` fails with a permission error, report which grant macOS asks for
  (Accessibility or Automation, for the python binary running the agent).
- The walk is fixed: indices stay valid between `read` and `press`. If the
  list was truncated, re-read with a higher `"max"` before pressing anything.

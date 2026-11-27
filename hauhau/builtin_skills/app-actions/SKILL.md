---
name: app-actions
description: Do things in Apple apps without touching their interface - create reminders and notes, open or draft mail, open Safari tabs, reveal files in Finder, play music. Use when the user wants an action in Mail, Notes, Reminders, Calendar, Safari, Music or Finder, and the task is not already covered by a dedicated tool.
---

# app-actions

macOS apps are scriptable: the `shell` tool runs `osascript` with AppleScript
one-liners that act in the app directly - faster and more reliable than driving
the interface, and no clicking is involved. Each first run per app triggers a
one-time macOS permission prompt (Automation); if a command fails with
"not allowed", tell the user which grant is missing instead of retrying.

## Reminders

```
osascript -e 'tell application "Reminders" to make new reminder with properties {name:"TEXT"}'
osascript -e 'tell application "Reminders" to make new reminder at list "LIST" with properties {name:"TEXT", due date:date "YYYY-MM-DD HH:MM"}'
```

## Notes

```
osascript -e 'tell application "Notes" to make new note with properties {name:"TITLE", body:"BODY"}'
osascript -e 'tell application "Notes" to show note (first note whose name contains "TEXT")'
```

## Mail

```
osascript -e 'tell application "Mail" to make new outgoing message with properties {subject:"SUBJ", content:"BODY", visible:true}'
```

Add a recipient to a visible draft, and send only when the user explicitly
asked for it - sending is irreversible:

```
osascript -e 'tell application "Mail" to tell outgoing message 1 to make new to recipient at end of to recipients with properties {address:"ADDR"}'
osascript -e 'tell application "Mail" to send outgoing message 1'
```

## Calendar

```
osascript -e 'tell application "Calendar" to make new event at end of events of calendar "CALENDAR" with properties {summary:"TITLE", start date:date "YYYY-MM-DD HH:MM"}'
```

## Safari

```
osascript -e 'tell application "Safari" to open location "URL"'
osascript -e 'tell application "Safari" to tell front document to do JavaScript "JS"'   # needs Develop > Allow JavaScript from Apple Events
```

## Finder and Music

```
osascript -e 'tell application "Finder" to reveal (POSIX file "PATH" as alias)'
osascript -e 'tell application "Music" to play playlist "NAME"'
```

## How to work

1. Prefer a journal tool (`agenda`, `journal_search`) for *reading*; use this
   skill for *writing and doing*.
2. Build the one-liner with the user's actual values, run it, and read the
   error if any - AppleScript errors name the app and the problem.
3. Quote text with single quotes outside and double quotes inside; escape
   user text containing quotes by reading the journal or a file instead.
4. Report what was created, with its name, so the user can verify.

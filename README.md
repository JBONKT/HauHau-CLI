# Hauhau

A conversational CLI agent that runs entirely on your machine, with a shell, a
filesystem, and a memory that survives between sessions.

Built on a local GGUF served by `llama.cpp`. No API keys, no cloud, no torch.

```
you › how much disk is left and what's eating it?

⚙ shell {"cmd":"df -h / && du -sh ~/* 2>/dev/null | sort -rh | head -5"}
  ↳ /dev/disk3s3s1  228Gi  13Gi  109Gi  11%  /
The volume is 11% full with 109 GB free. The largest thing in your home
directory is `modeller` at 3.4 GB, almost entirely the model file.
```

## Status

Phases 0–3 of the plan are implemented and tested: scaffold, conversation loop,
native tool use with full terminal access, and the persistent brain with
post-session self-learning, a behavioural eval suite, a bounded parameter
hill-climb and gated prompt self-evolution. Soak testing is not built — see
[Auto-tuning and self-evolution](#auto-tuning-and-self-evolution).

### M1 Air 8 GB profile (2026-11-27) — complete

Hardware-specific runtime work for this machine is **done and on `main`**
([PR #1](https://github.com/JBONKT/HauHau-CLI/pull/1)):

| Gate | Result |
|---|---|
| Unit suite | 810 tests OK |
| Live load integration | 16/16 checks (multi-turn tools, long context, clean shutdown) |
| Final perf benchmark | ~26 tok/s warm gen, burst σ ≈ 0.2, `/eval` ≥ 92%, `n_slots=1` |
| Shipped config | 8k ctx · 4 threads · full Metal · `-lm none -fa on -np 1` |

Gains vs the prior 16k / 8-thread / 4-slot setup are mainly **stability and RAM
layout**, not a large decode-speed jump (still ~26 tok/s on a 3.2 GB GGUF in
8 GB unified memory). Details: [Final configuration (M1 Air, 8 GB)](#final-configuration-m1-air-8-gb).

Release snapshots of this tree (source only, no `.venv` / weights) live under
`~/Documents/HauHau-archives/` when produced with the archive steps in
[Development](#development).

### Verified end to end

Exercised against the live model on this machine, not just unit-tested:
multi-step tool turns; cross-session recall; distillation, merging and
`/prune`/`/compact`; context compression (summarises, then reports
`summarised yes`); Ctrl-C mid-answer (the child `llama-server` survives and the
next turn works); `/perf`; `--resume last` and an explicit session id; `--think`;
`--no-brain`; `--no-server` and a missing model file (clear errors, no
traceback); piped output (answer only on stdout); and the agent starting and
stopping its own server. Load integration and the final performance benchmark
(2026-11-27) reconfirm gen stability and agent tool behaviour under sequential
pressure.

The full-screen interface is verified in a terminal emulator, not by eye:
`scripts/screen_ui_check.py` starts a stub llama-server, runs the CLI in a pty,
renders the byte stream through `pyte` and asserts the layout — header on row 0,
chrome pinned to the bottom, a long transcript scrolled inside a fixed frame,
`PgUp` freezing the view, typing resuming the newest line, and a live
`SIGWINCH` resize to 14×50 re-wrapping the text. It needs `pyte`
(`.venv/bin/pip install pyte`) and exits non-zero on a failure.

## Installation

Works on macOS and Linux. You need a local GGUF on disk; nothing is downloaded at
install or runtime.

Do **not** run macOS/BSD `install https://github.com/…` — that is a file-copy
utility, not a package installer. Clone the repo, then run setup as below.

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python3` on `PATH`. On macOS: `brew install python`. |
| `llama-server` | From [llama.cpp](https://github.com/ggerganov/llama.cpp). On macOS: `brew install llama.cpp`. Or set `[model] llama_server` in `config.toml` to an absolute path. |
| A `.gguf` model | Path in `config.toml` under `[model] gguf`, or override with `HAUHAU_GGUF` / `--model`. |

```bash
# macOS example
brew install python llama.cpp
```

### Clone and one-command setup

```bash
git clone https://github.com/JBONKT/HauHau-CLI.git
cd HauHau-CLI

# Point config.toml [model] gguf at your local .gguf first if the default path
# does not exist on this machine (setup lists nearby candidates on failure).

./scripts/setup.sh   # safe to re-run after a pull or a Python upgrade
```

What `./scripts/setup.sh` does:

1. Checks Python ≥ 3.11 and that `llama-server` is on `PATH`
2. Confirms the GGUF path in `config.toml` exists (lists nearby candidates if not)
3. Creates `.venv` if missing and upgrades `pip`
4. Installs the package editable with extras: `pip install -e '.[dev,tui]'`
5. Creates `~/.hauhau/{logs,brain,sessions}`
6. Writes `~/.local/bin/hauhau` → `.venv/bin/python -m hauhau --config <repo>/config.toml`
7. Smoke-tests that the launcher runs from another directory (`hauhau --profile`)

Successful setup ends with something like `ready. run: hauhau "…"` and
`launcher: ~/.local/bin/hauhau`.

Optional environment overrides for the script:

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` | Interpreter used to build the venv |
| `LAUNCHER_DIR` | `~/.local/bin` | Where the `hauhau` shim is written |

If setup prints that `~/.local/bin` is not on `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Manual install (without the script)

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e '.[dev,tui]'   # drop ,tui if you do not want Textual
.venv/bin/python -m hauhau --help
```

Core dependencies are `rich` and `prompt_toolkit`. The dashboard needs the `tui`
extra (`textual`). Dev linting uses the `dev` extra (`ruff`).

### Verify the install

Run these in order after setup. Steps 1–2 need no model load; 3–4 start
`llama-server` (first load can take a minute on cold Metal).

```bash
# 1) Launcher on PATH + CLI parses (exit 0)
which hauhau
hauhau --help

# 2) Same smoke check as setup.sh (exit 0 = launcher + import + config load OK).
# --profile may print little or nothing on success; only the exit code matters here.
cd /tmp && hauhau --profile
echo "profile exit=$?"

# Optional: print hardware probe and recommended server flags
hauhau --plain --profile

# 3) One-shot turn against the live model (starts llama-server, stops on exit)
hauhau --plain "Reply with exactly: hauhau-ok"
# Expect a short answer containing hauhau-ok and a timing line (tok / tok/s).

# 4) Interactive multi-turn (conversational flow + clean shutdown)
hauhau --plain
# At the prompt, try:
#   What is 2+2? Answer with just the number.
#   What did I just ask you?
#   /help
#   /quit
# Expect: numeric answer; a recall of the prior question; distill on quit; "bye".
```

| Check | Pass criteria |
|---|---|
| `which hauhau` | Resolves to `~/.local/bin/hauhau` (or your `LAUNCHER_DIR`) |
| `hauhau --profile` | Exit code 0 from any working directory |
| One-shot `--plain` | Banner shows GGUF + ctx/threads; model reply; process exits |
| Interactive `--plain` | Multi-turn context holds; `/quit` distills session and exits cleanly |

If step 3 fails, confirm `llama-server` works, the GGUF path exists, and nothing
else is bound to the configured port (default `http://127.0.0.1:8080`). Inspect
`~/.hauhau/logs/server.log` after a failed start.

**The model.** `config.toml` points `[model] gguf` at a local GGUF (currently
`~/modeller/gguf/Gemma-4-E2B-…-Q4_K_P.gguf`, ~3.2 GB). Any compatible GGUF works;
put it anywhere, update that line, or set `HAUHAU_GGUF` / pass `--model`.

Every value in `config.toml` is type-checked on load: a typo such as
`extra_args = "not-a-list"` (which used to be spliced into the server command
line one character at a time) or `max_steps = "many"` (which used to raise
`TypeError` mid-turn) is reported at startup and the default is used.

**What "it manages the server" means.** The first turn starts `llama-server` with
the configured context, threads and Metal offload, and stops it on exit —
verified: no process survives a normal quit. If a run is killed hard (`kill -9`,
closing the terminal mid-turn) a stray server can survive; `pkill llama-server`
clears it. To use your own long-lived server instead, pass
`--server-url http://127.0.0.1:8080` (or `--no-server`), and it will be reused
untouched.

If a startup fails (bad model path, `llama-server` exits, health timeout) the
child it spawned is stopped before the error surfaces, and the `server.log`
handle is closed on every `stop()`; otherwise each attempt leaked a process or a
descriptor.

Runtime state — memory, audit log, sessions, snapshots, server log — lives in
`~/.hauhau` by default (`--data-dir` to override), never in the repo.

## Usage

### Everyday commands

```bash
hauhau                              # interactive (full-screen UI on a real tty)
hauhau --plain                      # streaming UI (also used when piped / TERM=dumb)
hauhau --tui                        # Textual dashboard (needs hauhau[tui])
hauhau "what is in my Downloads?"   # one-shot: one turn, then exit
hauhau --resume last "continue"     # pick up the previous session
hauhau --resume SESSION_ID          # named session under ~/.hauhau/sessions
hauhau --plain --profile            # hardware profile + recommended flags
hauhau --think "…"                  # show model reasoning / thinking stream
hauhau --confirm "…"                # ask before every shell tool call
hauhau --no-brain "…"               # no persistent memory for this run
hauhau --model /path/to/model.gguf  # override GGUF for this run
hauhau --config /path/config.toml   # alternate config
hauhau --server-url http://127.0.0.1:8080   # attach to an existing llama-server
hauhau --no-server                  # never spawn llama-server ourselves
hauhau --data-dir ~/alt-hauhau      # alternate runtime directory
hauhau --max-steps 12               # cap tool steps per turn
```

Inside an interactive session, type `/help` for slash commands (`/model`,
`/status`, `/mem`, `/eval`, `/tune`, `/sync`, …). See [Slash commands](#slash-commands).

### CLI reference

| Flag | Purpose |
|---|---|
| `prompt …` | Positional: run a single turn and exit |
| `--config` | Path to `config.toml` |
| `--model` | Path to a `.gguf` (overrides config) |
| `--server-url` | Reuse an existing `llama-server` |
| `--no-server` | Never start `llama-server` |
| `--plain` | Stream interface instead of full-screen |
| `--tui` | Widget dashboard instead of full-screen frame |
| `--no-brain` | Disable persistent memory |
| `--confirm` | Confirm before every shell command |
| `--think` | Enable the model’s reasoning output |
| `--data-dir` | Memory, logs, and sessions directory |
| `--max-steps` | Max tool steps per turn |
| `--profile` | Print hardware profile and exit |
| `--resume [id]` | Resume last or named session |
| `--yolo` | Compatibility no-op; auto-run is already default |

### Interfaces

| Mode | When | Built with |
|---|---|---|
| Full screen (default) | Real tty, not `--plain`/`--tui` | `prompt_toolkit` fixed frame |
| Stream (`--plain`) | Flag, pipe, or `TERM=dumb` | `rich` line-oriented |
| Dashboard (`--tui`) | Explicit flag | `textual` (optional extra) |

For scripts and CI, prefer `--plain` so output is ordinary stdout. Bare
`--profile` can look empty when a full-screen console is selected without a live
app loop; `hauhau --plain --profile` always prints the table.

### Quick functional checks

```bash
# Core path: load model, answer, exit
hauhau --plain "Reply with exactly: hauhau-ok"

# Reasoning path
hauhau --plain --think "A bat and a ball cost \$1.10. The bat costs \$1.00 more than the ball. How much is the ball?"

# Tool path (needs shell tool allowed by config)
hauhau --plain "how much free disk is on / ? Use the shell if needed."
```

The first real turn starts `llama-server`; it stops when the process exits.

## Choosing a model

Any `.gguf` in `~/modeller/gguf` (or the directories in `config.py`'s
`GGUF_SEARCH_DIRS`) is a candidate. Drop a file there and it appears:

```
you › /model
│ # │ name                   │ size   │ in use   │
│ 1 │ Gemma-4-E2B-Uncensored │ 3.2 GB │ ✓ active │
│ 2 │ LFM3-1.7B              │ 1.0 GB │          │
you › /model lfm3          (also: /model 2, or a path)
· loading LFM3-1.7B (1.0 GB, ctx 16384)…
· LFM3-1.7B is live · ctx 16384 · 41.2 tok/s · saved to config.toml
```

Switching restarts `llama-server` and only then writes `config.toml`: a model
that fails to load leaves both the running session and the file untouched, and
the old server is started again. Each switch is in `audit.jsonl` as
`model.switch`, and the config is snapshotted before the write like every other
automated edit.

Two things travel with a model, remembered in `<data_dir>/models.json`:

| Setting | Why it is per model |
|---|---|
| `n_ctx` | A 1.7B with a small KV cache can afford 32k where a 3.2B cannot; carrying one context length across models either wastes RAM or makes llama-server reject the prompt. Whatever the tuner last adopted for a file is restored with it, clamped to the agent's context floor. |
| `extra_args` | Server flags that helped one model (`-ctk q8_0`, a rope setting) are usually wrong for another. |

The tuner is model-aware too: `_ctx_options` subtracts the weight file from the
RAM budget before choosing which context sizes are worth trying, so a 14 GB
model on a 16 GB machine is no longer offered 32k.

Measured on this machine (M1 Air, 8 GB) with the [final 8 GB profile](#final-configuration-m1-air-8-gb) below (`n_ctx=8192`, full Metal, `-np 1`):

| Model | Q4 size | gen tok/s | `/eval` | Notes |
|---|---|---|---|---|
| Gemma-4-E2B (Q4_K_P) | 3.2 GB | ~26 | 13/14 (93%) | **default agent brain** — multi-step tools hold under load |
| LFM3-1.7B (Q4_K_M) | 1.0 GB | ~36 | 8/12 (67%) | faster chat; weaker tool routing. Liquid's card says it is *not* recommended for agentic multi-step reasoning |

Judge a swap with `/eval`, not by feel: it scores tool choice, and a 2-point drop
there is a regression no matter how pleasant the prose reads.

## Final configuration (M1 Air, 8 GB)

`config.toml` is tuned for this box. The profiler (`hauhau --plain --profile`)
now recommends the same shape of settings for any ≤8 GB Apple Silicon machine:
8k context, half the cores for threads, full Metal offload, one server slot,
flash attention on, and SSD weight streaming when the GGUF is a large share of
RAM.

| Key | Value | Why |
|---|---|---|
| `model.n_ctx` | `8192` | 16k + 3.2 GB weights thrashed swap; 8k keeps KV small |
| `model.n_threads` | `4` | 8 logical cores; leave headroom for the client/OS |
| `model.n_gpu_layers` | `99` | full Metal is faster than partial (`-8`) here |
| `model.extra_args` | `["-lm", "none", "-fa", "on", "-np", "1"]` | stream weights; FA prefill; **one slot** (auto parallel opened 4× KV) |
| `generation.max_tokens` | `768` | shorter answers on a tight budget |
| `agent.max_steps` | `10` | fewer thrashing tool loops |
| `agent.context_budget_tokens` | `4000` | must stay ≤ `n_ctx/2` so `/tune`'s `ctx_floor` does not force 10k+ |
| `agent.keep_recent_turns` | `4` | smaller rolling history |
| `tools.max_output_chars` | `4000` | tool dumps must fit the 8k window |
| `brain.inject_max_chars` | `1200` | faster prompt; `top_k` stays 5 |

**Do not** enable KV-cache q8 (`-ctk q8_0 -ctv q8_0`) on this hardware: gen
speed roughly halved in `/tune` trials. `--cache-reuse` is refused by the
current `llama-server` build and is discarded by the tuner.

Load integration (unit suite + live multi-turn, 2026-11-27) held **~26 tok/s**
through a 6-turn agent session (max ~2.6 s/turn), `/eval` 13/14, long-context
tool turn OK, and a clean server shutdown. Final performance benchmark the same
day: warm gen **26.2 tok/s**, burst median **26.1** (σ ≈ 0.2), `/eval` 11/12
(92%), boot ~2.6 s, `n_slots=1`. Expect swap growth under the 3.2 GB model on
8 GB unified memory; that is normal if free RAM dips while wired stays around
3 GB.

After editing `extra_args` or context size, restart so a leftover server does
not keep old flags:

```bash
pkill -x llama-server
hauhau --plain --profile   # confirm recommendations
hauhau --plain             # then /status, /eval, /tune
```

## The second-brain journal

The memory brain holds lessons; the journal holds the record. Every mail,
message, call, calendar event and reminder is stored once in
`~/.hauhau/journal/journal.db`, keyed by `(source, external_id)` so a repeated
sync costs nothing, with a cursor per source so a run only asks for the delta.

```bash
.venv/bin/python scripts/journal_sync.py --once      # pull everything now
.venv/bin/python scripts/journal_sync.py --brief     # today's digest
.venv/bin/python scripts/journal_sync.py --status    # what it holds, what it can read
.venv/bin/python scripts/journal_sync.py --install   # 15-min sync + 07:30 brief
```

Inside the REPL that is `/sync`, `/brief` and `/journal [query]`, and the model
gets four more tools: `journal_search`, `agenda`, `person` and `brief`.

| source | mechanism | needs |
|---|---|---|
| mail | AppleScript, INBOX subjects from the last 7 days | Automation permission for Mail |
| calendar | AppleScript, ±14 days across all calendars | Automation permission for Calendar |
| reminders | AppleScript, open reminders with their list | Automation permission for Reminders |
| messages | `chat.db`, read-only, iMessage and SMS | **Full Disk Access** |
| calls | `CallHistoryDB`, read-only | **Full Disk Access** |

macOS grants those to the *interpreter*, not to the terminal:
`.venv/bin/python` must be added under System Settings → Privacy & Security →
Full Disk Access, because the launchd agent has no terminal session. `--status`
reports which sources actually work, so a silent "permission denied" every 15
minutes does not go unnoticed.

Two decisions worth recording. **The journal does not decay**: the brain ranks
memories by use and lets stale ones fade, which is right for lessons and wrong
for a record you may want to search in March. **Nothing leaves the machine**:
there is no account, no API key and no network call in the sync path, which is
also why Gmail/Google Calendar would need a different fetcher rather than a
switch.

## The interface

Three front ends share one agent, and `--plain` / `--tui` choose between them:

| | Stream (`--plain`, pipes, `TERM=dumb`) | Full screen (default on a tty) | Dashboard (`--tui`) |
|---|---|---|---|
| Built with | `rich` | `prompt_toolkit` | `textual` (optional: `pip install 'hauhau[tui]'`) |
| Layout | prints, terminal scrolls | fixed frame, scrolls inside it | two columns, real scrollback |
| Live stats | toolbar | header bar + stats row | side panel: tokens/s sparkline, memory, journal, tuner |
| For | scripts, `jq`, editors | normal use | watching a long task work |

```
 ⭘                                  Hauhau                                  02:05:51
                                    hur mycket disk ar det kvar egentligen? │ Hauhau
 ⚙ shell  df -h / && du -sh ~/* | sort -rh | head -3                        │ LFM3-1.7B
   ✓ Filesystem  Size  Used  Avail Capacity iused ifree %iused  M…          │ context   16k
   Disken är 11% full.                                                      │ threads   7t
                                                                            │ state     ready
   • 109 GB ledigt                                                          │ last turn 3.3s · 38 tok/s
   • störst: modeller 3.4 GB                                                │
                                                                            │ speed · tok/s history
                                                                            │ ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
                                                                            │ memories  41 mem
                                                                            │ journal   47 events
▊▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▎
▊  write a message…  (/help lists commands)                                                 ▎
 ^q quit  f2 prompt                                                        ^p palette
```

### What the dashboard costs

A terminal UI is drawn by the terminal, so "graphics acceleration" is not a
switch inside the app - but the app decides how much it asks the terminal to
draw. Measured on this machine (M1 Air, 8 GB) with `top -l 2` against a stub
server streaming 120 chunks/s:

| | idle | while an answer streams |
|---|---|---|
| before | 6.0% | 5.8% |
| after | 0.4% | 2.5% |

Three things changed:

* **The panel only repaints when a number changes.** `_panel_signature` is a tuple
  of every value on screen; the 8 Hz tick compares it and returns. Before, six
  widget updates plus a screen layout ran eight times a second for numbers that
  change between turns.
* **One batched, layout-free update.** `batch_update()` plus
  `Static.update(..., layout=False)`: the column has a fixed width, so only the
  characters change, and six separate updates each triggered a full layout.
* **Streaming is rate-limited** to one Markdown re-parse every 0.2 s (0.4 s on a
  CPU-rendered terminal). `Markdown` re-parses the *whole* answer, so a parse per
  token is quadratic.

The rest is the terminal. Textual asks for DEC 2026 (synchronized output) and
skips Apple Terminal, which answers neither the query nor the protocol - every
repaint lands as separate diffs that Terminal.app renders on the CPU. That is
what still feels sticky, and the app now says so on startup:

> Apple Terminal cannot synchronise repaints - Ghostty, kitty or WezTerm will
> scroll and stream noticeably smoother.

Warp (already installed here), Ghostty, kitty and WezTerm render on the GPU and
honour the sync protocol; `scripts/tui_check.py` prints whether sync output was
offered, which is the quickest way to tell what a given terminal is doing.

### Keys in the dashboard

| Key | What it does |
|---|---|
| `Enter` | send the line |
| `Tab` or `→` | accept the ghost suggestion |
| `Ctrl-Y` | copy the selection, or the last answer if nothing is selected |
| `Ctrl-V` | paste the system clipboard at the caret, flattened to one line |
| `Ctrl-C` | interrupt the turn (restarts the backend) / clear the line |
| `Ctrl-Q` | quit |
| `F2` | focus the prompt |
| `PgUp`/`PgDn` | scroll the conversation |

Copy is both: `App.copy_to_clipboard` speaks OSC 52 (iTerm2, WezTerm, Ghostty,
kitty) and the text also goes through `pbcopy`, because Terminal.app ignores OSC
52 and `pbcopy` is the only thing that works there. Paste reads `pbpaste` rather
than Textual's internal clipboard for the same reason; the binding is
`priority=True` so the focused `Input` does not swallow it with its own paste.

### Task suggestions while you type

The prompt line completes what you are typing, in two layers, cheapest first:

1. **Deterministic** (`hauhau/suggest.py`) - a table of the tasks this agent is
   actually good at, keyed by the word that introduces them, plus the slash
   commands. Typing `hur mycket disk` ghosts `är det kvar och vad tar mest
   plats?` with no model call at all.
2. **The local model** - for anything the table does not recognise, the model is
   asked to finish the sentence in at most twelve words (`max_tokens=48`,
   `temperature=0.3`), on a worker thread so the prompt keeps drawing. It is
   gated and throttled: never under 10 characters, never for a `/command`, never
   for a sentence that already ends in punctuation, at most one call every
   0.9 s and never two at once. A 2B model answers in about a second; a
   completion that arrives after you have typed past it is worse than none.

Only the *completion* is returned, never a rewritten prompt: ghost text renders
after the caret, so a suggestion that does not literally extend the line could
not be shown - and would fight the user's own wording. The side panel says which
layer produced the current ghost (`suggest  keyword|command|ai`), and
`suggest.propose` is in `audit.jsonl`, so you can tell what the model was asked.

The dashboard runs every turn on a worker thread and marshals widget updates
back to Textual's event loop; the answer streams into one `Markdown` widget that
is re-parsed at ~8 fps, the tool rows are clipped with an ellipsis instead of
reflowing, and the side panel repaints on a 0.12 s timer with the two aggregate
queries cached for two seconds. `scripts/tui_check.py` runs it in a pty against
a stub server and asserts the layout (18 checks: both columns, the ghost text and
Tab accepting it, the right-aligned user message, a one-line tool result, the
Markdown answer, no traceback).


On a terminal on both ends this is a **fixed, full-screen layout**: nothing
ever scrolls the terminal itself, the frame is redrawn in place.

```
 Hauhau · model Gemma-4-E2B · ctx 16k · threads 7t · ram 8 GB        <- header bar
╭ hauhau ─────────────────────────────────────────────────────────────╮
│                      how much disk is left and what is eating it?    │  <- your message,
│                                                                      │     a right-aligned
│   ⚙ shell  df -h / && du -sh ~/* | sort -rh | head -5                │     bubble
│     ✓ /dev/disk3s3s1  228Gi  13Gi  109Gi  11%  /  +4 lines · /show  │
│                                                                      │
│ The volume is 11% full with 109 GB free. The largest thing in your    │
│ home directory is `modeller` at 3.4 GB.                              │
╰ 1-9 of 9 ───────────────────────────────────────────────────────────╯
 11.0s · 53 tok · 27 tok/s                   /help · PgUp scrolls · Ctrl-D quits
❯ _                                                                     <- input
```

- **A chat window, not a log**: the transcript is drawn inside a rounded frame
  whose bottom edge carries the scroll position (`12-34 of 140`), the header is
  a full-width bar with model, context, threads and RAM, and your own messages
  are right-aligned bubbles with their own background so "said" and "answered"
  are separable at a glance. The frame is dropped below 56 columns, where two
  border columns cost more than they explain.
- **Three fixed rows of chrome**: header, stats, input. The transcript gets
  whatever is left, is wrapped to the terminal width, and is sliced to fit — the
  newest lines stay visible while a turn streams. `PgUp`/`PgDn` (also
  `Ctrl-B`/`Ctrl-F`, arrows) scroll *inside that frame*; the stats row says so
  and typing jumps back to the newest line.
- **A spinner only costs CPU when it spins**: the earlier build repainted 8
  times a second whether or not anything was happening, which measured 3.9% CPU
  on an idle M1 Air. Now a repaint clock runs only while a turn is in flight
  (0.0% idle), and the memory count in the header is refreshed at most every two
  seconds instead of being queried from sqlite on every frame.
- **Everything rich lands in the chat window**: `/status` panels, `/mem` tables
  and help boxes are captured through a `rich` console routed into the
  transcript, so no command can paint over the layout.
- **`--plain` (automatic when piped, or `TERM=dumb`)** keeps the streaming
  one-line-at-a-time interface for scripts, `jq`, and editors.

- **Answers render as Markdown**, not raw tokens. The model emits `**bold**`,
  bullets and fences constantly, so tokens accumulate into a Live region that
  re-renders the block (throttled to 8 fps — a re-parse per token is wasted work).
- **Tool calls show the interesting argument**, not raw JSON: `⚙ shell  df -h /`
  rather than `{"cmd":"df -h /"}`. Results are one line with `✓`/`✗`; a failed
  result wraps instead of ellipsizing, because the reason is the one line you
  must not lose.
- **`/show` replays the full output** of the last five tool calls. The one-line
  preview is a summary, not the evidence: `/show` (latest), `/show 2` (one back).
- **`/status` is one screen**: model, backend, context, memory, which prompt
  version is live, the workspace, the session file and where the config came from.
- **Slow commands show progress.** `/eval`, `/tune`, `/evolve`, `/learn` and a
  large `/compact` run under a spinner that names the task (`task 7/14:
  disk-space`), print each finished step (`n_threads=7 → 29.5 tok/s`), and fall
  back to plain one-line steps when output is piped.
- **`/help` is grouped** (this session, working, memory, model, self-tuning) and
  `/help tune` prints usage for one command. A mistyped command gets a
  suggestion: `/modle → did you mean /model?`.
- **Prompt:** `Tab` completes `/commands` *and* their arguments (`/prompt v` →
  `v1 v2`, `/help ev` → `eval evolve`), `Ctrl-C` cancels generation without
  leaving (`Ctrl-C` twice, or `Ctrl-D`, exits) — in the full-screen interface it
  restarts the backend to abort a stuck request. The stats row carries the token
  count and rate of the last turn.
- **Piped output stays clean**: `hauhau "what is in /tmp" | jq` gets only the
  answer on stdout — no banner, no colour, no session chatter. Markdown is left
  raw so it stays greppable.

Colour and the bottom status bar are plain `prompt_toolkit`/`rich`, so they
follow `TERM`/`NO_COLOR`. In a `TERM=dumb` environment both degrade silently to
plain text.

## Working it out instead of asking

A 2B model hands the turn back: *"vill du att jag listar diskarna?"* or *"det kan
jag inte"*. Both are answerable inside the turn, so the agent does not return
them. `AUTONOMY_RULES` in `hauhau/agent.py` states the three rules plainly -
look it up yourself, pick the most likely reading and say the assumption, never
stop at "I can't" - and the loop enforces them mechanically:

- a reply that says it cannot (`det kan jag inte`, matched by shape so Swedish
  word order works) is nudged to work out what is missing, try it, and record the
  result with `note_lesson`. Only the agent's own voice counts: a quoted error
  (`the build said unable to find crate`) or advice (`du kan köra make om den inte
  är installerad`) is not a refusal;
- a reply whose every sentence is a question is nudged, and the model continues in
  the *same* turn with a tool call. A finished answer with a closing offer
  (`Klart. Vill du att jag rensar?`) is an answer and is returned;
- a reply that pastes a command - in a fence with or without a language tag, or
  inline in backticks - is nudged to run it itself, and told to stop and ask only
  if the command would install, delete or overwrite;
- at most two nudges per turn, so a model that keeps asking cannot loop, and none
  at all after the step budget is spent or once a tool has *failed*: then the
  question is about a real obstacle and it is yours to see;
- the nudge text lives for exactly one request: it never enters the history, the
  session file or the next turn's prompt.

Each nudge is a `nudge` event in `audit.jsonl` with its kind (`stuck`, `act`,
`ambiguous`) and the sentence that triggered it, so "is it thinking for itself?"
has an answer.

`note_lesson(lesson, tags)` writes the distilled procedure to the procedural
memory, deduplicated, so the next time the step is instant instead of learned
again. It is offered only when a brain is enabled.

Turn the whole thing off with `autonomy = false` under `[agent]` in
`~/.hauhau/config.toml` and the bot answers exactly what it was given, question
and all.

## Development

```bash
.venv/bin/ruff check hauhau tests scripts              # clean
.venv/bin/python -m unittest discover -s tests -t .    # 760 tests
.venv/bin/python scripts/screen_ui_check.py            # 10 layout checks, real pty
.venv/bin/python scripts/tui_check.py                  # 18 dashboard checks, real pty
```

### Keeping a session going

Hauhau answers, then waits. `scripts/nudge.py` types one line into the terminal
tab the app owns when the session has produced nothing for a while, so a
run left alone keeps working instead of sitting at the prompt:

```bash
.venv/bin/python scripts/nudge.py --dry-run            # what it would do
.venv/bin/python scripts/nudge.py --idle-seconds 300   # push when quiet 5 min
```

Idleness is the newest mtime of `logs/server.log` and `logs/audit.jsonl`, so a
turn in progress is never interrupted. Each push is logged to
`~/.hauhau/logs/nudge.jsonl` and rate-limited: `--min-gap` between pushes and
`--max-per-hour` overall, because a stuck model that is fed lines forever burns
the context window without making progress. Delivery uses Terminal's
`do script … in tab`, which asks for Automation permission once.

## Assumptions this build corrects

The original plan made three assumptions that the machine disproved. Each one
changed the design, so they are recorded here rather than quietly dropped.

| Plan assumed | Reality | Consequence |
| --- | --- | --- |
| "gemma e2b" ≈ Gemma 2/3 9B–12B, plus an E2B *sandbox* | `Gemma-4-E2B-...-HauhauCS-Aggressive-Q4_K_P.gguf`, 3.2 GB. **E2B is a size class**; HauhauCS is the model author | No `e2b` SDK. Sandboxing is a separate decision, not implied by the model name |
| Gemma has no native function calling → hand-rolled ReAct JSON | The template implements `tools`/`tool_call`/`tool_response`; native calls work | `llm.py` uses the OpenAI tools API. No ReAct parser, no prompt-side schema |
| ChromaDB + sentence-transformers for RAG; Ollama as backend | 8 GB Apple M1; no `ollama`; torch/chromadb unavailable on Python 3.15 | `llama-server` backend, sqlite+FTS5 brain, stdlib everywhere but `rich`/`prompt_toolkit` |

Three model quirks are handled in code, each verified against the live server:

- **Thinking eats the budget.** With `max_tokens: 160`, a trivial request returned
  empty `content` and `finish_reason: length` — every token went to
  `reasoning_content`. `enable_thinking` is now sent explicitly on every request
  and `think` defaults to off.
- **EOS is unreliable.** The loader drops `</s>` from the EOG list for this merge
  and warns that `<|tool_response>` is not control-type. Termination is driven by
  `finish_reason` plus explicit `max_tokens`, never by a stop token.
- **The model cannot embed.** `/v1/embeddings` returns `Pooling type 'none' is not
  OAI compatible`. Vector recall therefore needs a *second* server
  (`llama-server -m <embed.gguf> --embeddings`) pointed at by
  `[brain] embeddings_url`. Empty by default: the brain then runs keyword-only,
  which is fully functional rather than degraded.

## How it works

```
hauhau/
  cli.py        REPL, slash commands, picks the screen or stream UI
  tui.py        full-screen UI: fixed frame, wrapped transcript, own scrolling
  agent.py      tool loop, system prompt, context compression
  llm.py        llama.cpp client: SSE streaming, native tools, think toggle
  server.py     owns the llama-server child process
  profiler.py   sysctl/proc hardware probe + parameter recommendation
  audit.py      append-only JSONL audit trail
  tools/        shell, file, search, memory tools behind one JSON-schema registry
  brain/        sqlite+FTS5 store, hybrid retrieval, distillation pass
```

**Tool loop.** The model is offered JSON-schema tools and calls them natively.
After `agent.max_steps` the tools are withheld for one final call, so a turn
always ends in prose instead of a dangling tool call. A crashing tool becomes a
failed tool result, never a failed turn.

**Memory.** Every write is deduplicated twice: by normalised hash, and by
token overlap — both Jaccard (same length, reworded) and containment (same fact
plus an extra clause, which Jaccard scores too low to catch). A merge keeps the
richer phrasing and sums reuse counts. Retrieval is FTS5 BM25 with recency decay
(`half_life_days`); the decay clock starts at the last *reinforcement*, so a fact
you re-confirm today is fresh again instead of sinking for being old.
`/compact` retro-merges duplicates that predate a threshold change.

Only a deliberate recall (`/mem <query>`, `rag_query`) or a merge counts as a
use. The memory block injected into every prompt deliberately does not: counting
it made `uses` meaningless within one session and, because `/prune` spares used
memories, it immunised the whole brain against forgetting.

Deduplication ignores the kind label — the distiller writes one fact as both
`semantic` and `procedural` in a single pass — and a shared file or path lowers
the bar, since `docs/release.md` in both texts means the same fact twice. A
number present on one side only is treated as detail, not contradiction; two
different numbers (`port 5432` vs `5433`) still stay apart.

**Self-learning.** On exit (or `/learn`) the transcript is distilled into
`preference` / `semantic` / `procedural` / `episodic` memories. The prompt is
primed with what is *already* remembered so the model stops restating known
facts. This runs once per session, not per turn — at 27 tok/s a per-turn pass
would cost more latency than it saves, so the exit pass prints
`distilling this session into memory…` and a session with no real work in it
(one short turn, no tool call) skips the pass entirely.

A 2B model will happily report "ran `echo OK` to check the shell" as a durable
procedure. Facts and procedures therefore have to name something concrete — a
path, filename, version, port, URL or flag — or mention this project, machine or
user, or they are dropped before they are written (`learn.rejected` in the audit
log counts them). Preferences are exempt: "wants short answers" is legitimately
a plain sentence.

**Context.** Above `context_budget_tokens` the oldest turns are summarised by a
cheap non-streaming call, and the summary itself is capped at a third of the
budget: it is re-injected every turn, and an uncapped one grew until no amount
of trimming could bring the prompt back under budget. Trimming only ever cuts at a `user` boundary, so an
assistant `tool_calls` message is never separated from its `tool` result.

**Token budget.** `[generation] max_tokens` applies to *every* call, tool turns
included. That is a correction, not a default: tool turns were originally capped
separately at 512, and a measured 600-word answer needs ~750 tokens, so long
answers were silently cut mid-sentence and returned as if complete. Now a
truncated answer is announced (`answer hit the N-token cap and was cut off`) and
a tool call cut off at the cap is **not executed** — its arguments are
incomplete, so the model is told to retry in smaller pieces rather than the
agent guessing the rest.

## The journal

Separate from the memory brain on purpose: the brain holds a few dozen distilled
lessons that decay; the journal holds *every* mail, message, call, event and
reminder, keyed by `(source, external_id)` so a repeated `/sync` is idempotent,
and it never decays — "what did Dana say in March" is a record question, not a
lesson. It lives in `~/.hauhau/journal/journal.db` (FTS5, keyword search) and
the model reaches it through the `journal_search`, `agenda`, `person` and
`brief` tools, so it can answer "what was said" without confusing records with
memories.

`/sync` reads the Mac's own records — nothing leaves the machine, no account
passwords are involved:

- **Mail, Calendar, Reminders** via AppleScript (the apps must be installed;
  macOS asks for Automation permission once per app pair).
- **iMessage/SMS** (`~/Library/Messages/chat.db`) and **call history**
  (`CallHistoryDB`) via read-only sqlite handles; both need **Full Disk Access**
  for the interpreter that runs Hauhau (System Settings > Privacy & Security).
  Without it the source reports a permission error instead of pretending there
  was nothing to read, and the other sources still sync.

Windows and limits are set in `[journal]` in `config.toml` (`mail_days`,
`calendar_days_back/ahead`, `messages_limit`, `calls_limit`,
`brief_lookahead_days`); `enabled = false` turns the journal and its tools off.
`scripts/journal_sync.py --status` reports which sources currently work.

## Skills

A skill is a folder with a `SKILL.md` (open [Agent Skills][agent-skills]
convention: YAML frontmatter with `name` and `description`, then a markdown
procedure), and optional support files. HauHau ships two, and scans in order:
`skills.dirs` from `config.toml`, then `~/.hauhau/skills/`, then the built-ins —
first name wins, so dropping a folder into `~/.hauhau/skills/` overrides a
built-in.

Only each skill's name and description live in the system prompt (progressive
disclosure, ~30 tokens per skill; a description is clipped to one 180-character
line and the list is capped, so fifty installed skills cannot crowd out the
conversation). The model pulls the full procedure on demand with the `skill`
tool, and scripts bundled in the folder run through the ordinary `shell` tool so
only their output enters the conversation.

Getting the model to actually *use* one is the hard part, so three things point
at it:

- the prompt block says so in the imperative — load the skill *before* acting,
  because the written procedure beats improvising;
- a tool that fails names the skill that documents it: `allowed-tools` in the
  frontmatter is how a skill says which tool it is for, and `ui` failing on a
  dialog now ends with "installed skill 'dialog-handler' has the procedure for
  this — load it with the skill tool before retrying" (one live run went from
  three blind `ui` attempts to loading the skill after the first failure);
- a mistyped name gets the closest match, and a skill dropped into
  `~/.hauhau/skills/` while the app is running is found on the next call: the
  tool re-scans once on a miss, so no restart is needed.

`/skills` lists what is installed (with each skill's tools and folder),
`/skills <name>` shows one procedure, `/skills reload` re-scans the folders and
reports what appeared or disappeared. A broken `SKILL.md` is a warning at
startup, never a failure, and a frontmatter written for another agent still
loads: `>`/`|` block scalars, CRLF, a BOM and `allowed-tools` in three spellings
are all accepted.

| skill | does |
|---|---|
| `app-actions` | acts in Apple apps without their interface: reminders, notes, mail drafts, calendar events, Safari, Finder — `osascript` one-liners the model fills in |
| `dialog-handler` | answers a dialog, sheet or popup blocking a running app |

`dialog-handler` and the window work underneath it use the `ui` tool: it reads
an app's frontmost window through the **accessibility tree** — no screenshots,
no coordinates — as an indexed outline, and `press`/`set` address an element by
that index. The model picks; macOS supplies the geometry. Both need the
interpreter binary (`.venv/bin/python`) granted **Accessibility** plus
Automation of System Events under System Settings → Privacy & Security; when a
grant is missing the tool fails with the osascript error naming it.

[agent-skills]: https://agentskills.io

## Safety

The default policy is **auto-run everything** — `shell_confirm = false` and an
empty denylist — as chosen for this build. The machinery to change that is in
place and tested:

```toml
[tools]
shell_confirm = true          # or: hauhau --confirm
denylist = ["rm\\s+-rf\\s+/", "mkfs"]
```

A command that times out is killed **with its children**: the shell runs in its
own session and the whole process group is signalled, because killing only
`bash` left backgrounded work running. The tool then says the partial output was
discarded, so the model does not present a partial run as a success.

Known limitation: the model is 2B and can misdescribe a refusal in its own prose
(it has claimed a declined command "ran successfully" while the tool result said
NOT RUN). The tool result, the red `✗` line on screen and the audit log are
authoritative — the command never executed.

Every shell command, tool call, file write, snapshot and config change is
appended to `~/.hauhau/logs/audit.jsonl`. Overwrites snapshot the previous file
into `~/.hauhau/.snapshots`. The denylist is checked *before* execution.

`/tune` rewrites `config.toml` from the in-memory values, so hand-written
comments in that file do not survive an adoption (the previous file is
snapshotted into `~/.hauhau/.snapshots` first, and every adopted setting is
logged in `evolve/tuning.jsonl`).

Treat the denylist as a speed bump, not a boundary. Quoting, case and line
continuations are normalised before matching, but shell indirection
(`X=rm; $X -rf /`) still gets through — only `shell_confirm` (the prompt) is a
real gate. `read_file` pages at 2000 lines by default; ask for more with
`limit`/`offset` instead of pulling a whole file into the prompt.

## Slash commands

`/help [command]` `/status` `/show [n]` `/tools` `/mem [query]` `/compact`
`/prune` `/remember` `/forget` `/learn` `/eval` `/tune` `/evolve` `/prompt`
`/model` `/perf` `/config` `/audit` `/context` `/clear` `/save` `/quit`
`/sync [source…]` `/journal [query]` `/brief` `/skills`

`/sync` pulls mail, iMessage/SMS, call history, calendar and reminders into the
journal (see below); `/journal` with no argument shows what it holds, with a
query it searches it; `/brief` prints today's digest — what arrived since
midnight, what is due, what is next. `scripts/journal_sync.py --install` sets
up launchd agents that sync every 15 minutes and push the brief at 07:30.

`/compact` merges memories that say the same thing. `/prune` is the forgetting
half of the decay model: it drops memories that are stale *and* were never
deliberately recalled again (an optional weight argument makes it stricter).

The injected block is assembled memory by memory until `inject_max_chars` is
spent, then says how many were withheld. Clipping the joined string instead cut
sentences mid-word — `/srv/… che lives at` — and presented the fragments as
facts.

## Auto-tuning and self-evolution

The agent can measure itself and change its own runtime knobs and system prompt.
Nothing is changed without a measurement that beats the current state.

**`/eval` — the yardstick.** 14 tasks, one short completion each (~12 s here).
A task passes if the model's *first action* is the right tool **and** the
arguments name the right thing (`df` for free space, `context_budget_tokens` for
a code search); text tasks pass only if the answer contains the required content
and no tool was called. It measures behaviour — "does it inspect the file or
invent a number" — not facts, because a 2B model's facts are not a stable
signal. Tasks whose tools are missing (brain off → `rag_*`) are dropped.

**`/tune [trials]` — the hill-climb.** Each key is climbed separately, with its
own baseline and its own yardstick: `model.*` keys need a backend restart and
are scored on generation tok/s, `brain.*` keys only change the prompt and are
scored on prompt tok/s. Bare `/tune` (no argument) uses a **hardware-aware
default trial budget**: 12 per key on ≤8 GB, 8 on ≤16 GB, otherwise 6 (the old
fixed 4 never reached flag groups on this machine). Pass an integer to override.

| key | candidates |
|---|---|
| `model.n_threads` | half the cores, all-but-one, capped at 8, every core |
| `model.n_ctx` | scaled by RAM: 2k on a 4 GB box up to 128k on a 64 GB box |
| `model.n_gpu_layers` | all layers, CPU only, all-but-8 (GPU machines only) |
| `model.extra_args` | registered flag bundles (below), never stacked |
| `brain.top_k` | 3 / 8 / 12 memories in the prompt |

The flag bundles are the llama.cpp knobs that change speed or memory without
changing output, each verified against `llama-server --help`: `-fa on|off`
(flash attention), `-np 1` (single server slot — auto parallel multiplies KV;
`-np 2` only offered on ≥16 GB), `-lm none` / `-lm dio` (stream the model from
SSD instead of mmap — offered when RAM is tight), `-ctk q8_0 -ctv q8_0`
(quantized KV cache), `--cache-reuse 256` (reuse KV chunks across turns),
`--context-shift` (auto-compact a full context instead of failing), `-cram` (KV
cache cap, scaled by RAM) and `-ub 1024` (physical batch, big machines only).
`set_value()` only accepts tokens from that registry plus whatever you put in
`extra_args` yourself, so the tuner cannot smuggle in `--host` or a different
model path.

A value is adopted only if it beats the best by 2 %. On this machine, after the
8 GB profile above was applied by hand, further `/tune` trials were mostly
noise around ~26 tok/s, so nothing else was adopted. Adopted values are
snapshotted and written to `config.toml`, and every trial is appended to
`evolve/tuning.jsonl`. Saving edits the file's values in place, so the comments
that explain why each one is set the way it is survive an adoption —
regenerating the file from the dataclasses silently deleted them once.

**`/evolve` — the prompt rewrite, gated.** Seeds `v1` with the current prompt,
scores it, then asks the model for up to three rewrites at rising temperature
using the failing tasks as feedback, scoring each, and **adopts the first that
scores strictly higher**. Measured here: a deliberately weak poetic prompt
scored 10/14 (71 %), the rewrite reached 14/14 and was adopted as `v2`.

Against the current 13/14 prompt the gate has held nine times, and the spread is
the interesting part — three attempts at 0.4/0.7/0.9 scored 12/14, 13/14 and
11/14; a rewrite with thinking on named the exact routing rule the failure asks
for and still scored 13/14. Hand-written candidates that added one routing
sentence reached 13/14 and 10/14. One rewrite is a coin flip, so several are
asked for; nothing wins, because the last failure is the model's tool choice on
one phrasing, not the prompt's clarity. That is what a gate is for: the file
says `kept the current prompt`, and the reason is now in the audit trail.

**Two guards came from a real failure.** A sweep adopted `n_ctx=4096` while the
session was sending 5.6k-token prompts, because `model.*` trials are scored on
tok/s alone and a smaller KV cache always wins; every turn after that was
rejected by the server. `model.n_ctx` candidates are now filtered by
`ctx_floor()` — twice `agent.context_budget_tokens`, never below 8192 — and
`set_value` refuses to adopt below it (a value the user already had can still be
restored). The same sweep also adopted `--cache-reuse 256`, which llama-server
logged as `cache_reuse is not supported by this context, it will be disabled`:
a trial that produces such a line is now scored 0 and discarded, matched by the
feature name llama.cpp prints rather than the flag that was passed.

**`/prompt [vN]` — history and rollback.** Versions live in
`<data_dir>/evolve/prompts/versions/v<N>.md` with a JSONL index and an `active`
pointer; the store wins over `config.toml` at startup. `/prompt v1` is the undo.
Every adoption snapshots `config.toml` first and is logged as `evolve.prompt` /
`prompt.apply` in the audit trail.

What stays deliberately unbuilt, because the plan's version would not survive
this hardware:

- **Quantization and GPU-offload hill-climbing.** Both need multi-gigabyte
  re-downloads, so they remain one-shot profiler output.
- **Soak testing** for memory growth and config thrash.

Safety rails for the tuner itself: `TUNABLE` is an allowlist of five keys,
`set_value()` enforces a numeric range per key and refuses anything else, so the
tuner cannot edit `shell_confirm`, the denylist, the workspace or the audit
switch. `audit.jsonl` stays append-only and outside every write tool's path.

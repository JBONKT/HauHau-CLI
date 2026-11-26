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

### Verified end to end

Exercised against the live model on this machine, not just unit-tested:
multi-step tool turns; cross-session recall; distillation, merging and
`/prune`/`/compact`; context compression (summarises, then reports
`summarised yes`); Ctrl-C mid-answer (the child `llama-server` survives and the
next turn works); `/perf`; `--resume last` and an explicit session id; `--think`;
`--no-brain`; `--no-server` and a missing model file (clear errors, no
traceback); piped output (answer only on stdout); and the agent starting and
stopping its own server.

The full-screen interface is verified in a terminal emulator, not by eye:
`scripts/screen_ui_check.py` starts a stub llama-server, runs the CLI in a pty,
renders the byte stream through `pyte` and asserts the layout — header on row 0,
chrome pinned to the bottom, a long transcript scrolled inside a fixed frame,
`PgUp` freezing the view, typing resuming the newest line, and a live
`SIGWINCH` resize to 14×50 re-wrapping the text. It needs `pyte`
(`.venv/bin/pip install pyte`) and exits non-zero on a failure.

## Setup (macOS)

Prerequisites, both from Homebrew:

```bash
brew install python llama.cpp      # python 3.11+, llama-server on PATH
```

Then one command, which is safe to re-run after a pull or a Python upgrade:

```bash
./scripts/setup.sh
```

It verifies Python and `llama-server`, checks the model file exists, builds
`.venv`, installs the package, creates `~/.hauhau`, and drops a launcher at
`~/.local/bin/hauhau` that works from any directory. It prints what it found and
fails loudly rather than half-installing.

```bash
hauhau                              # interactive
hauhau "what is in my Downloads?"   # one-shot
hauhau --resume last "continue"     # pick up the previous session
hauhau --profile                    # what hardware the tuner sees
```

**The model.** `config.toml` points `[model] gguf` at a local GGUF (currently
`~/modeller/gguf/Gemma-4-E2B-…-Q4_K_P.gguf`, 3.2 GB). Any Gemma-4-E2B GGUF works;
put it anywhere, update that line, or set `HAUHAU_GGUF`. Nothing is downloaded at
runtime.

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
`~/.hauhau`, never in the repo.

## The interface

On a terminal on both ends this is a **fixed, full-screen layout**: nothing
ever scrolls the terminal itself, the frame is redrawn in place.

```
 Hauhau · model Gemma-4-E2B…Q4_K_P.gguf · ctx 8192 · threads 4 · ram 8 GB   <- header
 11.0s · 53 tok · 27 tok/s · /help for commands · Ctrl-D exits · PgUp/PgDn  <- stats
────────────────────────────────── chat (scrolls inside this frame) ───────
❯ how much disk is left?
  ⚙ shell  df -h /
    ✓ /dev/disk3s3s1  228Gi  13Gi  109Gi  11%  /  +1 lines · /show

The volume is 11% full (109 GB free).
────────────────────────────────────────────────────────────────────────────
 ready · write a message and press Enter                                   <- hint
❯ _                                                                        <- input
```

- **One screen, four fixed rows of chrome**: header, stats, hint, input. The
  transcript gets whatever is left, is wrapped to the terminal width, and is
  sliced to fit — the newest lines stay visible while a turn streams. `PgUp`/
  `PgDn` (also `Ctrl-B`/`Ctrl-F`, arrows) scroll *inside that frame*; the hint
  row says so and typing jumps back to the newest line.
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

## Development

```bash
.venv/bin/ruff check hauhau tests                      # clean
.venv/bin/python -m unittest discover -s tests -t .    # 496 tests
```

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
scored on prompt tok/s. What it can try, all hardware-aware:

| key | candidates |
|---|---|
| `model.n_threads` | half the cores, all-but-one, capped at 8, every core |
| `model.n_ctx` | scaled by RAM: 2k on a 4 GB box up to 128k on a 64 GB box |
| `model.n_gpu_layers` | all layers, CPU only, all-but-8 (GPU machines only) |
| `model.extra_args` | registered flag bundles (below), never stacked |
| `brain.top_k` | 3 / 8 / 12 memories in the prompt |

The flag bundles are the llama.cpp knobs that change speed or memory without
changing output, each verified against `llama-server --help`: `-fa on|off`
(flash attention), `-lm none` / `-lm dio` (stream the model from SSD instead of
mmap — offered when RAM is tight), `-ctk q8_0 -ctv q8_0` (quantized KV cache),
`--cache-reuse 256` (reuse KV chunks across turns), `--context-shift`
(auto-compact a full context instead of failing), `-cram` (KV cache cap, scaled
by RAM) and `-ub 1024` (physical batch, big machines only). `set_value()` only
accepts tokens from that registry plus whatever you put in `extra_args`
yourself, so the tuner cannot smuggle in `--host` or a different model path.

A value is adopted only if it beats the best by 2 %. On this machine the honest
answer was *nothing*: 29.5 and 29.9 tok/s against a 29.7 baseline is noise, so
the config was left alone. Adopted values are snapshotted and written to
`config.toml`, and every trial is appended to `evolve/tuning.jsonl`.

**`/evolve` — the prompt rewrite, gated.** Seeds `v1` with the current prompt,
scores it, asks the model for an improved version using the failing tasks as
feedback, scores that, and **adopts only on a strictly higher score**. Measured
here: a deliberately weak poetic prompt scored 10/14 (71 %), the rewrite reached
14/14 and was adopted as `v2`; a rewrite of the already-good prompt also scored
13/14 and was *rejected*, so the agent did not churn.

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

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
post-session self-learning. Phases 4–6 (auto-tuning hill-climb, prompt
self-evolution, soak testing) are **not** built — see [Not built yet](#not-built-yet).

Requires Python 3.11+ and a `llama-server` on `PATH`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/hauhau                       # interactive
.venv/bin/hauhau "what is in /tmp?"    # one-shot

.venv/bin/pip install -e ".[dev]"      # lint tooling
.venv/bin/ruff check hauhau tests      # clean
.venv/bin/python -m unittest discover -s tests -t .   # 133 tests
```

Configuration lives in `config.toml` (see `hauhau --profile` for what the
profiler recommends on this hardware). Runtime state — memory, audit log,
sessions — lives in `~/.hauhau`, never in the repo.

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
  cli.py        REPL, slash commands, Rich streaming UI
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
(`half_life_days`) and a reuse boost; hits bump their own counter.
`/compact` retro-merges duplicates that predate a threshold change.

**Self-learning.** On exit (or `/learn`) the transcript is distilled into
`preference` / `semantic` / `procedural` / `episodic` memories. The prompt is
primed with what is *already* remembered so the model stops restating known
facts. This runs once per session, not per turn — at 27 tok/s a per-turn pass
would cost more latency than it saves.

**Context.** Above `context_budget_tokens` the oldest turns are summarised by a
cheap non-streaming call. Trimming only ever cuts at a `user` boundary, so an
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

Every shell command, tool call, file write, snapshot and config change is
appended to `~/.hauhau/logs/audit.jsonl`. Overwrites snapshot the previous file
into `~/.hauhau/.snapshots`. The denylist is checked *before* execution.

## Slash commands

`/help` `/tools` `/mem [query]` `/compact` `/remember` `/forget` `/learn`
`/model` `/perf` `/config` `/audit` `/context` `/clear` `/save` `/quit`

## Not built yet

Left for Phases 4–6, deliberately, because the plan's version of each would not
survive this hardware:

- **Auto-tuning hill-climb.** On Apple Silicon, quantization and GPU-offload
  choices require re-downloading multi-gigabyte weights, so they are one-shot
  profiler output, not session knobs. The honest tunables are `n_ctx`,
  `n_threads`, think on/off, retrieval `k`, `temperature` and the prompt text.
  `profiler.recommend()` already derives the first three; a hill-climb would
  need an eval set first.
- **Prompt self-evolution.** Needs a ~20-task pass/fail suite before it can mean
  anything, otherwise the "evolutionary log" just records regressions. Build the
  eval harness first.
- **Soak testing** for memory growth and config thrash.

A tuner must also not be able to edit its own safety policy or the audit log;
`audit.jsonl` is append-only and outside every write tool's path for that reason.

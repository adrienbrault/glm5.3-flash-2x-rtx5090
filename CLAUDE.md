# CLAUDE.md — GLM-5.3-Flash on 2× RTX 5090 (ExLlamaV3 + TabbyAPI, host-RAM expert offload)

This file is the working agreement for this repository, for people and coding agents alike (`AGENTS.md` is a link to it). Read it before changing anything here.

## What this repo is

Launcher, instruments, overlays and measurements for GLM-5.3-Flash (zai-org, 321 B total parameters, 288 routed experts per MoE layer, 8 per token) served by **TabbyAPI on top of ExLlamaV3** on the two RTX 5090 cards (32 GB each) and 60 GB of DDR5 of the `flan` box, port **8029**. The EXL3 checkpoint does not fit the 64 GB of VRAM: part of every MoE layer's routed experts lives in host RAM. The work here is making that fast at one stream (c1) first.

The LLM stack on that box has a second home in a private infrastructure repository: its `flan/r858/` directory is the source of `scripts/` here. The two must never disagree about the launcher or the drivers.

## Visibility — read this before pushing anywhere

This repository is **public** on GitHub as `adrienbrault/glm5.3-flash-2x-rtx5090` (created 2026-10-07, first pushed 2026-10-08). Write every commit as if it were public already: once pushed, GitHub keeps serving a commit by its SHA after a history rewrite.

- Commits use the GitHub noreply address (`git config user.email adrienbrault@users.noreply.github.com` in this checkout).
- Run `scripts/check-public-hygiene.sh` before every commit and `scripts/check-public-hygiene.sh --tree` before the first push.
- Stage explicit paths; `git add -A` and `git add .` are forbidden.
- Commit messages carry no session links or tool attribution lines.
- Nothing from the Qwen3.8-Flash-Next or 27B work is mirrored here; those have their own repositories. Results directories on the box are named, the box's addresses and home paths are not.
- Boot logs are never published (the launcher prints the LAN address). Token ids of private generations are text: publish only synthetic prompts and their outputs.

## README

The README is read top-down by someone who wants to know what runs and how fast. **Top half = the current state only**: what is served, with the numbers measured on it. Comparisons with earlier configurations, the story of how a setting was found, instrument caveats and review findings go to `docs/HISTORY.md`, the round's write-up in `bench/results/`, or `docs/GOTCHAS.md`; the top half may link to them in one clause.

**Decode metrics**: per-stream decode rate is the headline; time to the first token is separate; a round-wall figure appears only as a labelled secondary. Never per-stream × N as an aggregate.

## Prose (README, THIRD_PARTY.md, docs/)

Enforced by `scripts/check-prose.sh`, which `scripts/check-public-hygiene.sh` runs on every staged Markdown file.

- Declarative sentences. Each states what was measured, when, on which configuration, where the raw output is, and what it means.
- No evaluative or promotional words, no rhetorical devices, no exclamation marks, no questions in running text. The banned list is in the script; extend it when a new one gets through.
- Numbers carry their conditions: content kind (code, prose, chat) for every decode rate, prompt tokens and tokens per second for every prefill figure, concurrency, forced length, sampler, results directory.
- One paragraph is one line in the source; no manual wrapping.
- Exempt: `bench/results/`, and the delivered package docs under `docker/<overlay>/` (kept verbatim; the package SHA256SUMS covers them). Each overlay's own `README.md` follows these rules.
- A line that must keep a flagged word (a quoted error string, a proper name) carries `prose-ok: <reason>`.

## Credit

Every technique taken from someone else's work is credited in `THIRD_PARTY.md` in the same commit that uses it: code, ideas, measurements that set a design choice. The expert-offload design follows published work by others (`docs/PLAN.md` names each source per item).

## Measurement rules

1. **Force the length** with `min_tokens` (TabbyAPI's `ban_eos_token` is ignored by the exllamav3 backend). A sample whose `finish_reason` is not `length` is not a throughput sample.
2. **Report the conditions with the number**: content kind, concurrency, forced length, prompt depth, sampler, offload split, results directory. With the MTP draft on, decode rate moves with the text.
3. **Warm the batch shape** before measuring it.
4. **Compare in one session**, alternating arms; a comparison across days or images is not evidence of a change.
5. **Prefill is timed by the engine** (`usage.prompt_time`) on a cold prompt (`cached_tokens == 0`).
6. **No summary may hide the work done.** Every request writes one JSONL line; the summary is a convenience, never the record.

## Layout

| path | what it is |
| --- | --- |
| `scripts/launch-glm53.sh` | the launcher: writes the TabbyAPI config from `glm53_plan.py`, sanitises the image's environment, starts the container on :8029 |
| `scripts/glm53_plan.py` | config generation and the offload fit ladder |
| `scripts/glm53_probe.py` | the instrument: short-answer checks, forced-length decode, cold prefill, decode at depth |
| `scripts/r*.sh` | one driver per round, named after its R number |
| `docker/<overlay>/` | image overlays: patch, pristine base, landing assert, SHA256SUMS, Dockerfile |
| `bench/results/<rNNN-slug>.md` | one write-up per round; first line names the results directory on the box and the driver |
| `bench/results/<date>-<rNNN-slug>/` | raw records copied from the box; no boot logs, no files over 2 MB |
| `bench/RESULTS.md` | the index, newest first |
| `docs/PLAN.md` | the expert-offload work plan and the source of each idea |
| `docs/HISTORY.md` | how the served configuration changed |
| `docs/GOTCHAS.md` | the traps, as "what it looks like / what it is" |

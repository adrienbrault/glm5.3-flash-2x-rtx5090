# R968 text clock implementation

The executing source agrees with the brief; no contradiction was found. `src/exllamav3` and the previous `out/ANALYSIS.md` remain unchanged. The implementation and all supporting files are in `out/`. No network, SSH, GPU operation, Docker daemon call, or commit was performed.

## Patch and overlay

`textclock.patch` applies from the `src/exllamav3` directory with `patch -p1`. It changes only `modules/block_sparse_mlp_cpu.py` and `model/moe_exchange.py`. Their complete patched copies are under `textclock/` and are the Docker build inputs.

- `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1` enables a process-initialized flag, default `0`. Import raises a descriptive `ValueError` unless effective MODE/POLICY/CADENCE are `exchange`/`histogram`/`exact`. Existing defaults still apply: absent MODE means checkpoint and is rejected; absent POLICY/CADENCE mean histogram/exact.
- `swap_clock_owner(reg)` resolves the first current `cpu_component == "text"` module, falling back to `reg[0]`. With the flag off it returns `reg[0]`. Both tick gating and exchange sweep reset use this same helper. No cached owner survives registry changes.
- The counter still starts at zero, checks before increment, first sweeps on invocation 65 for interval 64, and subsequently sweeps every 64 owner invocations. Queue-drained sweeps reset the same owner. Capture still skips ticking/sweeping.
- Flag enabled: each dynamic registration emits `[SWAP-OWNER]`, showing the fallback during draft-only loading and then the text owner. The last startup line has the final registered module count. DEBUG additionally emits `[SWAP-CLOCK]` owner tick, cumulative owner invocations, pending state, and completed sweep count. Counters/logs add no CUDA tensor reads. DEBUG off emits no clock counters.
- Flag disabled: legacy owner/reset, clock schedule, and existing diagnostics are retained. Histogram collection, registry/budget order, thresholds, expert transactions, reader fences, and CUDA/native source are unchanged.

`Dockerfile.textclock` uses `ARG BASE` / `FROM ${BASE}`. It copies only the two patched Python files, resolves the installed package location, verifies both original daily source SHA-256 hashes, replaces the files, verifies patched hashes, and **imports both installed modules** to assert `TEXT_CLOCK_PATCH_VERSION == 1`. The RUN step explicitly enables the supported clock configuration for the assertion; this does not persist an enabled flag into the image. A mismatched daily source aborts the build. No dependency installation is required.

Offline build, from the workspace root (run on the GPU server where the daily image already exists):

```bash
docker build --network=none --pull=false \
  --build-arg BASE=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1 \
  -f out/Dockerfile.textclock \
  -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1_textclock1 out
```

## CPU verification

`CPU-TESTS.txt` records **11 module tests passing, 10/10 mutation variants caught, and 4 orchestration/summary tests passing**. Reproduce with:

```bash
python3 out/test_textclock.py
python3 out/test_gpu_unit.py
```

The module harness executes the complete patched modules with Torch/native dependencies stubbed. It covers MTP-first/text-first registries, first-text selection, flag off/default, fallback and registry replacement, invocation 64/65 and repeated reset behavior, queue-drained reset, capture, validation rejection, startup/debug counters, and legacy off-mode trace/log parity against the supplied daily modules. Mutations individually break gating, reset, first-text selection, fallback, off/default behavior, validation, startup logging, sweep counters, and capture exclusion; each is detected.

The orchestration harness executes the real shell unit and summary in temporary directories with synthetic launcher/probe/Docker commands. It checks debug isolation, fresh ABBA order, c4-first measurement sequencing, DEBUG off, FLAN_POWER=0, fixed fingerprint salt, daily restoration after a failed probe, each independent gate failure, and rejection of missing/tampered evidence. It never calls a Docker daemon or network endpoint.

Packaging checks also passed: patch application to temporary exact-source copies; byte comparison with overlay inputs; Python compilation; shell syntax; Docker RUN assertion syntax and original/patched hash guards. These checks do not execute expert arithmetic, transfers, or the real image import.

## GPU unit

Install these files together at `/srv/qwen5090/r968/`: `r968-glm53-textclock.sh`, `glm_arms.sh`, `textclock_summary.py`, `Dockerfile.textclock`, and the `textclock/` directory. `glm_arms.sh` is an unchanged copy of the supplied reference. The unit uses the server's installed GPU queue/drain libraries and daily tools, including `glm53_probe.py`.

The script's header contains the systemd-run command. It registers before waiting, acquires/validates the GPU lock, drains the gateway and waits idle, checks the daily image tag, and builds the overlay with `--network=none --pull=false`. It fails if the baseline daily already overrides the clock/debug flags. Original daily environment words are preserved; B appends its clock/debug overrides through `EXL3_EXTRA`. DEBUG off is an **empty value**, because the existing DEBUG implementation treats the string `0` as enabled.

A separate fresh Bdebug boot runs one distinct c4 round (`--runs 1`, 1024 tokens per stream). It records the final startup owner and counts completed exchange sweeps and matching `[SWAP-CLOCK] event=sweep` lines in the c4 log window; it requires a text owner and positive matching counts. This boot's state is discarded.

Four measured fresh boots follow in **A1, B1, B2, A2** order. A uses the daily image as supplied; B uses the overlay with TEXT_CLOCK=1 and DEBUG off. Each measured boot runs, in order:

1. Cold c4: `--phase decode --concurrency 4 --runs 2 --distinct`.
2. c1: `--phase c1 --runs 2 --kinds code,prose,chat,html,edit`.
3. c4 again with the same decode arguments.
4. The supplied `glm_arms.sh` `fp` function: greedy, five kinds, one run, fixed `r968fp` salt.

There is no extra client warmup before cold c4; the launcher's normal load-time behavior remains. `FLAN_POWER=0` is exported for probes and passed to boots. Container image and effective clock/debug/mode/policy/cadence are checked per boot. The EXIT trap restores the **current installed daily**, re-reading its environment, even after failure or interruption and even if another unit is queued. Restoration happens while holding the GPU/drain locks; a restore failure produces FATAL and a failing exit status.

`textclock_summary.py` reports each arm's cold c4 aggregate median, median over the ten c1 rates, post-c1 c4 aggregate median, and five SHA-256 fingerprints. Fingerprint hashes are checked against their archived reasoning/content streams. The summary requires complete expected sample/run sets and valid finite rates. It aggregates A1/A2 and B1/B2 using the median of each arm's median, then gates:

- B cold c4 / A cold c4 >= 1.10.
- 0.98 <= B c1 / A c1 <= 1.02.
- All five fingerprint hashes identical across all four arms.
- No FATAL lines in available run logs or summary; no DEBUG sweep/clock lines in measured engine logs.

Results include `summary.txt`, per-arm logs/JSONL/container environment, the debug c4 sweep window, `gate.json`, restore logs, and final `last.txt`. Incomplete evidence or failed gates exits nonzero and still restores the daily.

Expected wall time is **50–70 minutes**, excluding queue wait and gateway drain: six boots including restoration at about 2.3 minutes each (~13.8 minutes), approximately 35–50 minutes of workloads at c4 75–95 and c1 30–50 tok/s, and roughly 1–3 minutes to build.

The real overlay build/import, throughput gates, and greedy output equality remain GPU-server work. Local CPU checks establish the clock/control flow and unit behavior; they do not establish GPU speed or cross-placement numerical equality.

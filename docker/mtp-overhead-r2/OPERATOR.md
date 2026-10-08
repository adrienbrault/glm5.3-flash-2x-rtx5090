# Build and GPU slot instructions

Build before acquiring the GPU slot. From the workspace root:

```sh
docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2 -f out/Dockerfile out
python3 out/cpu_tests.py
```

The Dockerfile inherits `tabbyapi:cheapswap-r3-agent-r2_mtpfast1`. `SHA256.json` checks all 10 touched originals, asserts the new profiler path is absent, stages the full patch, applies `patch -p1 --fuzz=0`, rejects offsets/fuzz, checks all 11 landing SHA256 values, syntax-checks them, and only then copies them into the installed ExLlama package and `/app`. The final build step imports torch **before** importing ExLlama/the precompiled extension. It neither loads the model nor rebuilds CUDA.

Expected build landing lines:

```text
[MTP-OVERHEAD] landing SHA256 verified: 11 files; patch fuzz=0, offsets=0
[MTP-OVERHEAD] CPU import landing PASS (all new optimization flags default off)
```

For a pinned image identity, record its image ID after build. A checksum mismatch means the base is a different snapshot; do not loosen the patch/checksum checks.

## Launch settings and landing

Retain N104 (`OFFLOAD_N=104`), depth1 (`DRAFT=1 DRAFT_N=1`), `EXL3_MTP_FAST=1`, eight physical CPU threads, vision on, cache262144, original Q8 cache modes and four serving slots. Use NVMe checkpoint loading. Fix the same static expert statistics file in every arm. For the operator's launcher, use `PLACEMENT=static`; verify the **container** has:

```text
EXL3_MOE_CPU_SWAP=0
EXL3_MOE_CPU_SPLIT_STATS=/absolute/container/path/to/the/same/static-statistics.json
EXL3_MTP_FAST=1
```

Record the statistics-file SHA256 and actual model/module placements. Setting only an environment variable named PLACEMENT inside Docker does not configure ExLlama; that name belongs to the operator launcher. The provided launcher is not in this workspace, and previous rounds sanitized EXL3_*: add the following names to its explicit pass-through whitelist, or pass them as actual `docker run -e NAME=value` arguments. **Read `docker inspect` container Env**, rather than trusting host exports.

```text
EXL3_MTP_PHASE_PROF EXL3_MTP_PHASE_STEPS EXL3_MTP_PHASE_TRACE
EXL3_MTP_GPU_DRAFT EXL3_MTP_GPU_EMBED EXL3_MTP_GREEDY_ACCEPT
EXL3_MTP_CACHED_REWIND EXL3_MTP_MAX_BATCH EXL3_MTP_MIRROR_RESERVE_MB
EXL3_DRAFT_PINNED_STAGING EXL3_MOE_HANDOFF_PROF
EXL3_MTP_TRACE_SKIP EXL3_MTP_TRACE_STEPS EXL3_MTP_TRACE_FILE
EXL3_MTP_REPAIR_MERGE_PROBE
```

Unset/zero all new optimization flags for controls. Also unset the existing unrelated experimental flags `EXL3_MTP_DEVICE_DRAFT`, `EXL3_EMBED_GPU`, `EXL3_EMBED_GPU_PRUNED`, `EXL3_BATCH_VERIFY`, `EXL3_DECODE_OVERLAP`, draft-pruning/CPU-skip flags and prompt lookup. Existing MLA/KDA graph defaults stay on. Use the same settings for every arm except the named variable.

Expected server landing:

```text
[MTP-OVERHEAD] landed phase_prof=False gpu_draft=False greedy_accept=False max_batch=0
```

Booleans/numeric cap change with the selected arm. Keep the base MTP-fast confirmation: one `CPU MoE worker started: 43 layers`, a text/MTP join message, and no second one-layer worker. Additional flags print:

```text
[MTP-OVERHEAD] gpu_embed=1 (identity embedding, full mirror, guarded VRAM)
[MTP-OVERHEAD] cached_rewind=1 (live geometry keys, existing batched kernels)
[MTP-OVERHEAD] gpu_embedding mirror device=cuda:N bytes=...
[MTP-OVERHEAD] gpu_embedding declined: insufficient VRAM for full exact mirror
[MTP-OVERHEAD] gpu_draft eligible=True (GLM text c1 d1, full mirror with free VRAM required)
```

Mirror/eligibility messages occur at first use, not necessarily during boot. A True flag with `eligible=False` or mirror decline is a fallback arm, **not evidence that GPU drafting ran**. A full mirror is not budgeted by the existing autosplit loader; the guard checks free memory and keeps a reserve, but the real verify workspace still must fit. If the arm OOMs, stop it and report it; do not change N104 or cache/vision inside this comparison.

## Per-arm flags

`EXL3_MTP_FAST=1` is common in every MTP arm. A flag absent from a row must be unset or zero.

| Arm | Env additions |
|---|---|
| A0 / A0-repeat | New flags off, `EXL3_MTP_PHASE_PROF=0` |
| P-MTP | `EXL3_MTP_PHASE_PROF=1 EXL3_MTP_PHASE_STEPS=64` |
| P-plain | Same profile flags, MTP mode disabled at **N104** |
| S | `EXL3_DRAFT_PINNED_STAGING=1`, profiler off |
| E | `EXL3_MTP_GPU_EMBED=1`, profiler off |
| D | `EXL3_MTP_GPU_DRAFT=1`, profiler off |
| B | `EXL3_MTP_GREEDY_ACCEPT=1`, profiler off |
| R | `EXL3_MTP_CACHED_REWIND=1`, profiler off |
| ALL | `EXL3_DRAFT_PINNED_STAGING=1 EXL3_MTP_GPU_EMBED=1 EXL3_MTP_GPU_DRAFT=1 EXL3_MTP_GREEDY_ACCEPT=1 EXL3_MTP_CACHED_REWIND=1`, profiler off |
| C-cap | `EXL3_MTP_MAX_BATCH=1`, other optimizations off; c4 first |
| C-plain | MTP mode disabled, same N104/static placement/four slots |

There is no new production repair-fold/whole-forward graph flag. Those are explicitly diagnostic probes. Do not enable the broader existing EXL3_BATCH_VERIFY flag to claim sampled-seed identity: it supports sampled trajectories with different RNG ordering. The new greedy flag opts into only its stateless greedy subset.

## Exact token gate and fixed workloads

For raw sampled token IDs, run the original Tabby main through the included startup wrapper, preserving its existing arguments and mounts:

```text
python /opt/mtp-overhead/gpu_probe.py /app/main.py [the original Tabby arguments]
```

Override the container command/entrypoint as appropriate for the existing launcher; the image's default entrypoint is inherited. The wrapper files are copied into the image. This is a **server startup command**, not a second process to run with docker exec against an already running model. Set `EXL3_MTP_PHASE_PROF=0 EXL3_MTP_TRACE_STEPS=0 EXL3_MTP_REPAIR_MERGE_PROBE=0` for token-gate/speed arms. The wrapper respects explicitly set phase_prof=0. Otherwise it defaults profiling/trace ranges on for diagnostics.

Every finished physical request emits `[MTP-EXACT]` with actual sampled token IDs, initial prompt-ID SHA256, serial and requeue marker. Requeue segments preserve the initial prompt key. `check_exact.py` concatenates segments per logical serial and compares the complete trajectories grouped by prompt; c4 request ordering/UUIDs need not match. It compares sampled IDs directly, including sampled stop tokens, rather than re-tokenizing displayed text.

Use fixed prompts, same number/order of warmups, temperature0/top_k1, neutral repetition/frequency/presence penalties, no healing/filters, 1024 tokens and the same request sequence. Save all server logs and probe outputs. The included stdlib-only client is runnable on the operator host:

```sh
TABBY_API_KEY=... python3 out/fixed_probe.py --url http://127.0.0.1:8029 --c 1 --rounds 3 --out results/A0
python3 out/check_exact.py results/A0-engine.log results/D-engine.log
python3 out/check_exact.py results/A0-engine.log results/ALL-engine.log
```

The client includes five fixed raw-completion prompts (code/prose/chat/html/edit). `--prompts fixed-prompts.json` replaces them with the original R892 prompt map; use the **same saved file** for all arms if using that option. Existing operator R828_COUNTER probes remain the authority for server step metrics and comparison with R892; the new client prints API usage and wall time only. It sets min_tokens=max_tokens so short EOS completions do not truncate the workload. A request timeout is 120 seconds. Abort an arm whose first request falls below 10% of baseline or fails, rather than spending the slot waiting.

Compare each optimization c1 arm to same-image MTP-enabled flags-off A0 and to A0-repeat. Run lengths must match for the checker: use one round in every arm's gate set, and a separate three-round speed set for controls/ALL, or truncate **whole identical requests** into separate gate logs. R892 runtime exchange made even B0 versus B0b diverge at token0–870: `PLACEMENT=static` is mandatory, and A0-repeat must pass before judging candidates.

For batch cap, compare sustained c4 to C-plain, not to A0 c1 or always-MTP c4. The cap deliberately changes target sequence length; q1/q2 floating-point kernels can differ. The same-image c1 optimizations preserve the MTP verification shape. Report any always-MTP versus cap divergence separately; lossless ordinary greedy decoding is not automatically bit-identical to two-row speculative arithmetic.

## Reading phase lines

A schema-1 line has:

```json
{"schema":1,"mode":"mtp","batch":1,"depth":1,"steps":64,"host":{"draft.tables":{"ms_per_step":0.0,"calls":64}},"cuda":{"verify.forward_device@cuda:0":{"ms_per_step":0.0,"calls":64}}}
```

The zeros above illustrate **format only**, not measurements. All ms_per_step values divide totals by the bucket's number of steps, so repair costs are acceptance-weighted and sparse sweeps are amortized. `calls` explains how often a child phase ran. Missing phases did not run in that bucket. `mode=plain` is ordinary MTP-off; `plain_suspended` is cap/policy-suppressed MTP; `depth` is the selected ceiling, and first-token fallback/confidence truncation can produce fewer actual draft columns.

```sh
python3 out/summarize_phase.py results/P-MTP-engine.log results/P-plain-engine.log > results/phases.jsonl
```

The summarizer weights averages by steps and skips the first aggregate per mode/shape per file. Do not add parent/child or device/stream ranges. Forward spans already include worker waits, sampling includes receive_sample/rewind, cache_commit includes repair, and head_argmax is inside draft.sample. Host synchronization duration often waits for forward work already counted by CUDA events. Use CUDA traces to distinguish actual kernels from exposed idle waits. Record profiler-on native R828 time separately to quantify observer cost.

## Graph and repair probes in the real served process

P-MTP can run via gpu_probe.py with:

```text
EXL3_MTP_PHASE_PROF=1 EXL3_MTP_PHASE_TRACE=1
EXL3_MTP_TRACE_SKIP=128 EXL3_MTP_TRACE_STEPS=32
EXL3_MTP_TRACE_FILE=/results/mtp-trace.json
EXL3_MTP_REPAIR_MERGE_PROBE=1
```

The repair probe requires `EXL3_MOE_CPU_SWAP=0`, c1 d1, non-windowed draft cache, no TP. It snapshots only the affected physical draft-cache pages across **every cache tensor**, including pooled indexer keys, runs separate repair/draft versus a merged causal q2 forward, compares last draft hidden and cache bytes, prints `[MTP-REPAIR-PROBE]`, and restores the real post-repair state before normal drafting continues. It adds real work and synchronizations and is not a speed arm. Any `hidden_equal=false` or `cache_pages_equal=false` rejects that merge. Eight accepted rounds might not cover every mod4 or a page crossing; extend the diagnostic if necessary. A PASS for a few samples alone is not permission to deploy the merge.

Run P-plain through the wrapper with repair probe off and a distinct trace filename. Optionally use a fixed longer prompt to cross index_topk and repeat the trace capture in a later separate boot; compare below/above sparse regime, mod4 and page boundaries. The output trace can be opened in Perfetto:

```sh
python3 out/trace_summary.py results/mtp-trace.json results/plain-trace.json
```

Inspect actual cudaGraphLaunch and capture/synchronization calls, associated with `MTP/verify.layers`, `MTP/draft.layers`, and their per-device envelopes. The summary counts runtime events; follow correlation IDs in the trace for attribution. If no CUDA graph runtime events were exported, the trace is inconclusive (some CUDA/Kineto builds expose graph nodes without runtime launch entries); repeat using the operator's Nsight Systems tooling. Component slot configuration or a graph-enabled env flag is not proof of replay. Do not whole-capture CPU-worker host submissions.

At static placement no exchange sweep should occur. To price exchange, run a separate profiling-only dynamic-placement boot with the exact daily exchange env and same content, then inspect exchange_sweep/copy/fence ranges. Never use that boot to judge token identity; the provided R892 data already proves its repeatability floor is inadequate.

## One approximately 60-minute GPU slot

Build/import-check on CPU first. Budgets assume NVMe boots near one minute; abort early on health/import/VRAM failure. Prioritize the actual phase breakdown and exactness over extra throughput repeats if boot time overruns.

| Elapsed minutes | Work |
|---|---|
| 0–5 | A0 static N104 d1 flags off. Five kinds ×3 speed requests (1024 each); save a separate one-round exact gate log. |
| 5–11 | P-MTP and P-plain N104. Five kinds ×1 each, warmed 32-step traces; enable eight-round repair merge probe only in P-MTP and discard its first phase bucket. |
| 11–36 | S, E, D, B, R, one fresh boot each. Five kinds ×1 exact gate, same prompts/settings; repeat the speed set when arm time permits. Record whether mirror/eligibility actually activated. |
| 36–43 | ALL flags, profiler off. One-round exact gate versus A0, then five kinds ×3 speed set; health/vision/basic tool-call checks. |
| 43–48 | C-cap N104 d1 cap1, profiler off. Five kinds at sustained c4, save token gate + R828; c2 and a c1→c2→c1 transition if time allows. |
| 48–54 | C-plain N104 MTP-off, same c4 workload/order/slots; token gate versus C-cap and compare server step cost. |
| 54–60 | A0-repeat exact/speed control. Gate flags-off repeatability; restore the operator's prior serving configuration. |

For explicit transition coverage: begin a 4096-token request, add a second 1024-token request while the first remains active, then let the second finish. The first stays plain after concurrency falls to one. Send a follow-up using its complete output as a prefix (dirty page reuse), and a new unrelated c1 request. With a short profiling-only transition pass, expect plain_suspended/no draft/no repair for the former and normal MTP for the fresh job. Test a physical requeue using the operator's small per-request requeue budget if available. These are additional correctness checks; do not mix transition rounds into steady-state timing aggregates.

## Tabby configuration alternative to the cap flag

The updated Tabby config accepts:

```yaml
draft_model:
  draft_mode: mtp
  draft_num_tokens: 1
  draft_num_tokens_by_batch:
    - [1, 1]
    - [4, 0]
```

This covers serving max_batch4; for a larger configured capacity, replace 4 with that capacity (above the last rule, the positive default depth is used). Config depth0 uses decode-ready count; the env cap uses all active jobs. Both conservatively suspend an affected job through completion/requeue. Persistent EXL3_NVME_TIER must be unset: this patch rejects suspension with persistent KV because draft-validity metadata cannot survive process restart; in-memory CPU-tier eviction is supported. Policy changes do not deallocate MTP cache/history or automatically rebuild a suspended cache. The model backend already passes the policy into Generator and reserves the maximum recurrent history; those files need no new patch.

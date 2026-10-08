# MTP overhead overlay, revision 2

`tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2` is `tabbyapi:cheapswap-r3-agent-r2_mtpfast1` (`../mtp-fast-r1/` on top of `../glm-agent-r2-on-cheapswap-r3/`) with this package applied. It is the served image since 2026-10-08.

The patch changes ten Python files of ExLlamaV3 and one of TabbyAPI (`common/config_models.py`) and adds `exllamav3/util/mtp_phase.py`. No native extension is rebuilt. `apply_overlay.py` checks the SHA256 of every original file against `SHA256.json`, applies `mtp-overhead.patch` with `--fuzz=0` in a scratch tree, rejects any offset, checks the SHA256 of every patched file, parses each one, and only then installs them. The Dockerfile then imports the generator and asserts that the phase profiler is off.

Every flag below defaults to off; with none set, the image runs the code paths of its base.

| flag | what it does | served |
|---|---|---|
| `EXL3_MTP_MAX_BATCH=1` | drafts only while one request is active; with two or more active requests every job decodes one token per step without a draft, and a job suspended this way stays plain until it completes | yes |
| `EXL3_DRAFT_PINNED_STAGING=1` | the existing pinned staging flag; this revision also makes the MTP input layer's upload non-blocking | yes |
| `EXL3_MTP_GPU_DRAFT=1` | keeps the draft token ids on the GPU for the verify step and reads them back on a copy stream; text requests at c1, depth 1 only | yes |
| `EXL3_MTP_GREEDY_ACCEPT=1` | one batched acceptance readback instead of one per position, for greedy requests only; sampled requests take the original path | yes |
| `EXL3_MTP_CACHED_REWIND=1` | reuses the recurrent-state rewind descriptors per slot and geometry; the native rewind kernels are unchanged | yes |
| `EXL3_MTP_GPU_EMBED=1` | a GPU copy of the embedding table for the MTP input layer | no |
| `EXL3_MTP_PHASE_PROF=1` | per-phase CUDA-event and host timers of the MTP step, logged every `EXL3_MTP_PHASE_STEPS` steps | no (diagnostic) |

TabbyAPI's `draft_model.draft_num_tokens_by_batch` accepts a depth of 0 after this patch, as a configuration alternative to `EXL3_MTP_MAX_BATCH` (`OPERATOR.md`).

## Measurements

- R902b, 2026-10-08, static placement, 104 routed experts per layer on the CPU, MTP depth 1 with `EXL3_MTP_FAST=1`, c1 over five content kinds: the four host flags together 29.82 ms per MTP step against 29.96 and 30.60 ms for the two arms without them (results directory `2026-10-08-r902b-glm53-mtp-overhead-noembed-092828`).
- R902, same settings: the arm with all five flags, `EXL3_MTP_GPU_EMBED=1` included, failed with a CUDA out-of-memory error on GPU0 at its first request (`2026-10-08-r902-glm53-mtp-overhead-071847`).
- R914, 2026-10-08, the served configuration with and without the four host flags: 15.69 ms per token with them, 15.46 and 15.73 ms per token in the two arms without them (`../../bench/results/r914-glm53-promote-combo.md`).

## Files

- `mtp-overhead.patch`, `SHA256.json`, `apply_overlay.py`, `package.py`: the patch, the landing manifest, the build-time landing and the script that produced both from `base/` and `src/` trees.
- `cpu_tests.py`, `CPU-TESTS.txt`: 23 CPU tests with mocked CUDA and loader parts, and their output.
- `gpu_probe.py`, `repair_merge_probe.py`, `fixed_probe.py`, `check_exact.py`, `summarize_phase.py`, `trace_summary.py`: the GPU-side probes and log summarisers of the round.
- `DIAGNOSIS.md`, `ESTIMATE.md`, `OPERATOR.md`, `log-analysis.json`: the round's diagnosis, cost estimates and run plan, kept as delivered. Their estimates are not measurements.

## Build

```sh
sha256sum -c SHA256SUMS
docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2 .
```

The launcher passes the `EXL3_*` flags through `EXL3_EXTRA` for overlay images only (`scripts/launch-glm53.sh`); `scripts/glm-daily.env` lists the served set.

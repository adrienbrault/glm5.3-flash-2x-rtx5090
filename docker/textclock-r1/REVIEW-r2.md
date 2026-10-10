# REVIEW r2: EXL3_MOE_CPU_SWAP_TEXT_CLOCK patch + R968 unit (Opus, 2026-10-10)

**Verdict: FIX-FIRST.** The patch, Dockerfile and GPU flow are fine to run. Two summary bugs make OVERALL fail whatever the GPU does, and one unit-header value can leave the daily down. All three fixes are small and none of them touches the patch.

Checked locally: `textclock.patch` applies to `src/exllamav3` with `patch -p1` and the result is byte-identical to `out/textclock/*`. The sha256 of the two source files and the two patched files match the hashes in `Dockerfile.textclock`. `test_textclock.py` passes 11/11 with 10/10 mutations caught; `test_gpu_unit.py` passes 4/4. `out/glm_arms.sh` is identical to `ref/glm_arms.sh`.

## Blockers (minimal fix)

1. **The summary rejects the real probe's html rows, so every arm reports "invalid evidence".** In the live probe (`flan/r858/glm53_probe.py:222`) `KIND_TOKENS = {'html': 2048}`, and `:330` passes that value to `c1_decode`. So both the html c1 rows and the html fingerprint row carry `tokens: 2048`. `textclock_summary.py:48` (c1) and `:59` (fp) require `tokens == 1024`, raise `ValueError`, and the unit exits 2 after about 60 min of GPU. The CPU test's fake probe does not model this. Fix: drop the fixed 1024 check (the probe already raises when `completion_tokens != tokens`, so every emitted row is complete), or compare against `{'html': 2048}.get(kind, 1024)`. Before queueing, check that `/srv/qwen5090/glm-daily-tools/glm53_probe.py` has the same KIND_TOKENS.

2. **`fingerprints_identical` cannot pass on this daily, so it must not be an OVERALL gate.** Greedy output is not reproducible across boots under dynamic exchange plus the CPU worker's reduction order:
   - LEARNINGS.md:563: B0b diverged from B0 on every kind.
   - :652: two identical static boots diverged at char 0.
   - :818: the daily config differs on 5/5 kinds unless `EXL3_FREEZE_EXCHANGE=1`.
   - :1029: "Fingerprint DIFFERS carries no information".

   The determinism flags (`EXL3_ORDERED_MOE`, `EXL3_FIXED_STREAM_T`, `EXL3_FREEZE_EXCHANGE`) are not in this image's source. B also sweeps during cold c4 and A does not, so placements differ by the time fp runs. Even A1 vs A2 will differ. Fix: keep the hashes and print per-kind identity for A1 vs A2 (the control), B1 vs B2, and A vs B as information only. Remove `fingerprints_identical` from `gates`/`passed` (`textclock_summary.py:104-112`). Output equality under exchange needs the in-process teacher-forced method (LEARNINGS:626 swap KL floor 0.00048; :785). That is out of scope for R968.

3. **The unit header's `RuntimeMaxSec=10800` (r968 line 5) can leave the daily down.** The queue wait counts toward this limit. If the cap fires mid-run, systemd sends SIGTERM, but bash defers the TERM trap until the foreground probe exits, and the probe has a timeout of up to 2400/3600 s. `TimeoutStopSec=600` then SIGKILLs the cgroup before the `arms_cleanup` restore runs. Use `RuntimeMaxSec=43200` (house rule for queued units). Optionally raise `TimeoutStopSec` to at least 1800 as well.

## Design checks (pass)

- **Owner selection.** `swap_clock_owner` (cpu.py:19-23) picks the first `cpu_component=="text"` module, falling back to `reg[0]`. With the flag off it returns `reg[0]`. The same helper is used by the tick gate (cpu.py:762) and by the exchange reset (moe_exchange.py:267-268), so after interval 64 the counter cannot re-fire on every call. The checkpoint-path reset at cpu.py:121 still uses `reg[0]`, but import validation (cpu.py:8-15) rejects the flag unless MODE=exchange, so that path is unreachable while the flag is on.
- **Component labels.** Components are set by `model.py:533` (`mtp` for the head). Registration (cpu.py:263-273) runs after `load_cpu_split` has set `cpu_component`. Bdebug asserts `component=text` on the last `[SWAP-OWNER]` line.
- **Cost and thread safety.** The registry is mutated only on load and unload. The forward path is single-threaded, and the lookup scans at most two entries when MTP registers first. The per-call cost is negligible.
- **Flag off.** The only change on the off path is an in-function import in `run_sweep`. Byte-identical behaviour with the flag off is CPU-tested only, because A runs the daily image rather than the overlay with flag=0. That is acceptable per the brief.
- **Exactness under c4 sweeps (correctness, not bit-equality).**
  - The tick runs at the start of the owner's `cpu_split_submit` (cpu.py:345-350), before this layer's map translate and GPU expert dispatch (block_sparse_mlp.py:1109). The owner is the first text split layer, and earlier layers' collects are enqueued before their forward returns. All four rows of the batch therefore see one placement per layer.
  - `run_sweep` is unchanged: per-layer `fence_layer`, `committed.synchronize()` and `require_healthy` (moe_exchange.py:270-281). It is the same path R967 already fired 113 times at c1. The patch changes when sweeps happen, not how.
  - The existing assumption is not bit-exactness: a placement change is a numeric step (cpu.py:794-798 comment; LEARNINGS:626). See blocker 2.
- **Dockerfile.** `FROM ${BASE}`, copies only the two files, guards the original and patched hashes, and asserts `TEXT_CLOCK_PATCH_VERSION==1` and `_text_clock_enabled` on import. The build-time exllamav3 import has precedent in earlier overlay Dockerfiles (`patches/exllamav3/*/Dockerfile.box`). The unit builds with `--network=none --pull=false` (r968:75).
- **Unit.**
  - Locking and drain: `gpu_lock` then `flock -n 9` (r968:58-59). Drain and idle wait (r968:71-72). Checks the base tag and that the daily does not already set the clock or DEBUG flags (r968:67-69).
  - Arms: Bdebug runs first and is discarded, then fresh boots A1 B1 B2 A2. Each arm runs cold c4, then c1, then c4 after, then fp (r968:132-150). Only `GLM_IMG` and `EXL3_EXTRA` differ, and `container.json` asserts both (r968:96-105).
  - DEBUG is passed as an empty value, which is correct because `os.environ.get` treats `"0"` as true. The launcher regex accepts an empty value (launch:128). The image tag matches `SWAP_FAMILY_RE` and the AGENT regex.
  - Restore: the trap restores the re-read daily on every exit path after the first boot (r968:26-53). Before the first boot (`TOUCHED=0`) the daily was never touched.
  - Timeouts are bounded: build 900 s, boot 1500 s, c4 2400 s, c1 3600 s, fp 1800 s.
  - The unit needs only `/srv/qwen5090/lib/*`, `glm-daily-tools` and `$P/{glm_arms.sh,textclock_summary.py,Dockerfile.textclock,textclock/}`.
- **Gate math.** Each side's value is the median of two arm medians, and the bars are 1.10 (cold c4) and 0.98-1.02 (c1), as the brief states.

## Nits

- **c1 within 2 % is inside GLM boot-to-boot drift** (R961b: about 2.7 % per arm, LEARNINGS:1031). ABBA cancels linear drift but not noise, so a c1 FAIL is inconclusive rather than evidence against the patch. Also print the per-pair ratios (B1/A1, B2/A2).
- **Raw capture is lost.** The probe's `stream()` writes `Path(out).parent/tag` with mode `'w'`. So `c4-after` overwrites cold c4's `decode-c4-r*-s*.events/request` files, and fp overwrites c1's `c1-*-r0.*` files. The summary still reads fp correctly because fp runs last. Fix: give each phase its own `--out` subdirectory.
- **The FATAL scan is too wide.** `rglob('*.log')` (summary:97) also scans `$R/tools/`, a copy of `glm-daily-tools`. One stray "FATAL" there gives a spurious FAIL. Scope the scan to the arm directories, Bdebug and the top level.
- **The unit always restores, even when another unit is queued.** The brief asks for this, but CLAUDE.md's chain rule prefers glm_arms' queue-aware skip.

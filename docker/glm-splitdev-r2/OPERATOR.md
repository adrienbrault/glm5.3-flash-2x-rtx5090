# splitdev2 operator slot (maximum 30 minutes)

Stage this entire `out/` directory at `/srv/qwen5090/r915c-splitdev2`. The source patch targets the **overhead-r2 base image**, not splitdev1. Keep `/srv/qwen5090/r915b/glm_arms.sh` and `/srv/qwen5090/r915b-tools` from R915b: the unit uses their `boot`, `c1spd`, warmup, distinct and vision probes. `c1spd` is the existing code/prose/chat/html/edit ×2 matrix. No promotion or daily-env edit is part of this slot.

## Build and landing (before acquiring the GPUs, about 1 minute)

```bash
set -euo pipefail
cd /srv/qwen5090/r915c-splitdev2
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
sudo -n docker build --progress=plain -t "$IMG" . 2>&1 | tee build.log
sudo -n docker run --rm --entrypoint bash "$IMG" /opt/splitdev/image-checks.sh 2>&1 | tee landing.log
```

Require the SHA checks, both zero-fuzz patch applications, `SPLITDEV2 LANDED`, and all 25 CPU tests (including the real-Torch scatter test) to pass. A hash mismatch means the starting image is different; do not bypass it. The Dockerfile checks all five modified existing files and all six resulting files. Native sources/ABI are unchanged, so there is **no rebuild**. Only a future native change should rebuild with `TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4`.

## GPU slot

```bash
sudo systemd-run --unit=r915c-glm53-splitdev2 \
  --property=RuntimeMaxSec=1770 --property=TimeoutStopSec=15 \
  bash /srv/qwen5090/r915c-splitdev2/r915c-splitdev2.sh
journalctl -fu r915c-glm53-splitdev2
```

The script follows R915b's queue/lock and cleanup conventions. `arms_init` reads the current `/srv/qwen5090/glm-daily.env`; its EXIT cleanup restores the daily by reading that file again. The provided snapshot is reference material, not a replacement for the live file. The script adds flags to the daily's `EXL3_EXTRA` with **semicolons**:

```text
EXL3_EXTRA=<current daily extras>;EXL3_MOE_CPU_SPLIT_BY_DEVICE=98,104;EXL3_MOE_CPU_SPLIT_CHECK=1
```

Testing stops 25 minutes after the script starts; each boot/probe also has a timeout. The final minutes are reserved for daily restoration; the systemd runtime plus its stop timeout fit within 30 minutes overall. Build before the unit so build time does not consume the GPU slot. Waiting for the GPU queue counts toward the budget; if it expires while queued, resubmit after checking queue state. Long queue waits can leave alternatives incomplete.

1. **C-n98, CHECK=1**, `GPU_SPLIT=31.8,31`, daily N=104 and all daily MTP/vision/exchange settings preserved. Boot and run the existing warmup phase, which must include the saved sparse long-prompt case, not just a short request. Look for `[SPLIT-CHECK]` contracts for both device counts and layer 45. Layer 23 may log `replace_worker_idx=... old_cpu=98 new_cpu=104`. The worker can report 44 registered layers for 43 live layers because the discarded native registration keeps its immutable index. CHECK validates every split forward while enabled; it does not automatically stop after a number of calls.
2. **B-n98, CHECK=0**, fresh container/worker. Repeat warmup, then c1 five kinds ×2, distinct c1/c2/c4 ×2, and vision. Report c1 medians, distinct aggregate T/s, first-token/prefill results, MTP acceptance and server-step figures. Prompts must be distinct where requested; do not compare cached-prefix speed with uncached speed. At least one long prefill and decode beyond the exchange cadence must complete. Existing R915b probes exercise these shapes and /64 sweeps; do not shorten them below that cadence.
3. Capture **per-GPU free VRAM after warmup and after the full matrix**. Use the existing gates: GPU0 ≥450 MiB and GPU1 ≥300 MiB. The supplied logs place MTP on GPU0 at N=98; do not assume it is on GPU1. CHECK reports its actual device and worker count.
4. Only after the entire 98 matrix passes, try **96,104**, then **100,104**, at the same `GPU_SPLIT=31.8,31` with CHECK=0. Each arm that boots gets the same matrix and headroom gate. N=96 previously failed load; classify that as a load failure, not an indexing failure. Timeouts or arms skipped to reserve restoration time are **incomplete/untested**, never successes. If the full three-arm matrix cannot fit, retain the completed 98 results and use another slot for incomplete alternatives; do not exceed 30 minutes.

A CHECK exception includes `layer=... tensor=...` plus actual/expected limits. Stop that arm and return its engine log; do not retry inference in a CUDA-asserted container. A split-map/native-reader error can be localized further with `CUDA_LAUNCH_BLOCKING=1` appended to `EXL3_EXTRA` on the CHECK boot, using the saved request below. Do not time that boot.

## Return these files

Return `build.log`, the landing/CPU-test outputs, `transcript.log`, each arm's `boot*` logs, `engine.log`, `warmup.log/jsonl`, the c1 probe logs, `dec.log/jsonl`, `vision.log/jsonl`, `server-steps.txt`, `vram-warmup.csv`, `vram-matrix.csv`, and `vram.csv`. Verify that the unit's cleanup logged restoration of the **current** daily and that its warmup succeeds. Do not change `glm-daily.env` based on this diagnostic slot alone.

## Optional probe of the original r1 failure

The supplied CUDA log cannot formally name the failing Python call. `Dockerfile.probe-r1` builds a **diagnostic-only** splitdev1 derivative. It retains the defective key-only cache, logs requested vs cached N at registration, and raises immediately before `counts1.scatter_add_` when `shifted` is outside the stale histogram. This provides direct live evidence without letting the scatter assert. Use it only if attribution remains disputed; it is not the fixed image.

```bash
sudo -n docker build -f Dockerfile.probe-r1 \
  -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev1_probe .
```

Boot that image with the same R915b arm parameters (`98,104`, `31.8,31`, daily extras), within a separate bounded diagnostic slot if necessary. Replay the exact supplied `logs/boot/warmup-sparse-c1-s0.request.json` after boot, using port 8029 and the unchanged request JSON included in this output directory:

```bash
curl --fail-with-body --max-time 120 -N -H 'Content-Type: application/json' \
  --data-binary @warmup-sparse-c1-s0.request.json \
  http://127.0.0.1:8029/v1/chat/completions
```

Expected probe: layer 23 cached CPU=98/requested=104, then `index_tensor=shifted`, `size=99`, a maximum in 99..104. If another layer/tensor fails first, return that evidence rather than claiming the live call-site attribution is settled.

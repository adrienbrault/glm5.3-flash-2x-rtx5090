# R3 operator runbook

The workspace ran CPU simulation/tests and a fresh zero-fuzz patch landing only. No GPU,
Docker, SSH or git was used. `POLICY.md` documents the conditional model and results.
The score winner does not beat static plus a small histogram budget. Begin with the
initialization-only r2 arm after R872 validates exchange; score is an optional comparison.

## Verify and build

```sh
cd out/r3
sha256sum -c SHA256SUMS
python -m unittest discover -s tests -v   # needs NumPy; no Torch/GPU import
# These scripts are operator commands; the writing workspace did not run them.
docker build -t tabbyapi:cheapswap-r3 .
```

Dockerfile: `FROM tabbyapi:r828-prompt-lookup-r3`. Source hashes must match that pristine
overlay; apply uses `patch --batch --forward --fuzz=0 -p1` and exact landing hashes/asserts.
Native code is unchanged, so no native rebuild is needed. If source hashes differ, use:

```sh
docker run --rm --entrypoint python tabbyapi:r828-prompt-lookup-r3 -c 'import hashlib,importlib.util,json,pathlib; r=pathlib.Path(next(iter(importlib.util.find_spec("exllamav3").submodule_search_locations))); print(json.dumps({p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in ("model/moe_cpu_host.py","modules/block_sparse_mlp_cpu.py","modules/block_sparse_mlp.py")},indent=2))'
```

Compare `source_hashes.json`; reproduce the stated source overlay before building rather
than relaxing fuzz. `r2-results.json` is retained r2 calibration data, not measured r3 output.

## Profile provenance

The bundled `r869-split-stats.json` is reconstructed from complete R869 actual decode rows,
including warmup. It is not asserted byte-identical to the file used to measure 57.5 tok/s.
Use the **same** profile in all static/initialized/score arms. To locate and compare the
operator's measured profile (its exact location is absent from the sources):

```sh
find /srv/qwen5090 -name 'split-stats-broad-r869.json' -print
python inspect_profile.py /actual/path/from/the/find/output/split-stats-broad-r869.json
```

If using that measured profile, mount it at a known path through the existing launcher and
replace `EXL3_MOE_CPU_SPLIT_STATS` in **every** new arm's env file. `ab_launch.py` preserves
source mounts. It also clears inherited SWAP/SCORE knobs and profiling settings before
applying the chosen env file. The bundled profile needs no extra mount: COPY puts it at
`/opt/cheapswap/r869-split-stats.json`. All counts are in original checkpoint expert IDs.

## Inherited correctness probes, before full-model A/B

Run sequentially with GPUs free:

```sh
docker run --rm --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/cpu_native_selftest.py
docker run --rm --gpus all --ipc=host --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_selftest.py --device cuda:0 --swizzle 1
docker run --rm --gpus all --ipc=host --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_selftest.py --device cuda:1 --swizzle 1
docker run --rm --gpus all --ipc=host --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_selftest.py --bits 8 --swizzle 1
docker run --rm --gpus all --ipc=host --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_ring_selftest.py
docker run --rm --gpus all --ipc=host -e EXL3_MOE_SPLIT_FUSED=0 --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_ring_selftest.py
docker run --rm --gpus all --ipc=host -e EXL3_MOE_RECON_FOLDED=0 --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_ring_selftest.py
docker run --rm --gpus all --ipc=host -e EXL3_MOE_CPU_SWIZZLE=0 --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_ring_selftest.py
docker run --rm --gpus all --ipc=host -e EXL3_MOE_CPU_MAX_ISA=avx2 --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_ring_selftest.py
docker run --rm --gpus all --ipc=host --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/probe_dma.py --mb 6.33 --iters 256
docker run --rm --gpus all --ipc=host --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/policy_gpu_probe.py
```

These inherited numerical/ring fixtures exercise histogram exchange. Their transaction and
layout coverage is unchanged. The extra probe measures isolated vector-decay launches;
full-model no-swap score controls below measure their live exposed cost. Neither probe
establishes full-model score graph/queue/c4 behavior. After an exchange failure reload the
model; if CUDA cannot drain the failed copy stream, restart the process as required by r2.

## Matched full-model arms

Use the same saved serving configuration as R872: N=96, eight threads, MTP off, vision on,
existing chunk/batch/context limits. Confirm 192 GPU + 96 CPU experts on text layers 3..44.
Set SOURCE to that existing container's actual name:

```sh
export SOURCE=your_existing_N96_served_container
python ab_launch.py --source "$SOURCE" --name r3-static --image tabbyapi:cheapswap-r3 --env-file A-static.env --stop-source
# Run existing health/warmup/vision + c1/c4 benchmark harness; save results.
docker stop r3-static
python ab_launch.py --source "$SOURCE" --name r3-initialized --image tabbyapi:cheapswap-r3 --env-file B-initialized.env
# Same checks/harness; force swaps before timing and measure peak allocated/reserved VRAM.
docker stop r3-initialized
python ab_launch.py --source "$SOURCE" --name r3-small --image tabbyapi:cheapswap-r3 --env-file B-small.env
# Same harness; this is the smaller global histogram budget comparator.
docker stop r3-small
python ab_launch.py --source "$SOURCE" --name r3-score --image tabbyapi:cheapswap-r3 --env-file C-score.env
# Same checks/harness, especially prefill -> decode, cancellation -> new request, c4 arrivals.
docker stop r3-score
python ab_launch.py --source "$SOURCE" --name r3-score-noswap --image tabbyapi:cheapswap-r3 --env-file C-score-noswap.env
# Compare with static for decay/signal overhead without exchange; require zero logged swaps.
docker stop r3-score-noswap
python ab_launch.py --source "$SOURCE" --name r3-static-a2 --image tabbyapi:cheapswap-r3 --env-file A-static.env
# Repeat the original static harness, then restore the source.
docker stop r3-static-a2
docker start "$SOURCE"
```

N=96 has about 12.7 MB/device minimum exchange scratch beyond planning; allocator reserve
can be larger. Force swaps and verify bytes before throughput timing. Use the same red/blue
vision probe, small/large prefill, cached-prefix reuse, long prompts, decode, cancellations,
queue drains, module-local graphs and c4 on each arm. Capture and replay score hist addresses
through actual BC calls, verify original router IDs and profile-derived map inverses at
load, and confirm only decode affects the decode histogram. MTP must use a separate fitting
configuration (e.g. the already measured N=100), with draft reject/accept and call/accepted
output-token counts logged separately. Preserve actual source streams; external whole-model
capture bypassing Python cadence is outside r2/r3's supported contract.

For diagnostics copy the relevant env and append:

```sh
cp C-score.env C-score-probe.env
cat >> C-score-probe.env <<'ENV'
EXL3_MOE_CPU_SWAP_DEBUG=1
EXL3_MOE_CPU_SWAP_VERIFY=1
ENV
```

Run it through the same `ab_launch.py` after stopping the previous arm. Save logs: exact
payload/trellis shapes, zero-swap and active sweep wall/fence costs, swaps/call, per-layer
CPU picks, TTFT, inter-token p95, RSS and VRAM. Disable flags for timing. Test both fused and
copy paths. `C-score-no-prefill.env` isolates the prompt signal. If actual projection shapes
reject pair exchanges, compare measured accepted swap count with replay rather than assuming
every proposal was executable.

## Knobs and replay

`EXL3_MOE_CPU_SWAP_POLICY=score` only activates in exchange mode with SWAP=1. Defaults are
histogram selection when the policy is absent. Score knobs and defaults:

| Variable | Default | Meaning |
|---|---:|---|
| EXL3_MOE_CPU_SWAP_INTERVAL | 32 | k decode model calls |
| EXL3_MOE_CPU_SCORE_HALF_LIFE | 512 | H model calls |
| EXL3_MOE_CPU_SCORE_PREFILL | 32 | equivalent prompt observations |
| EXL3_MOE_CPU_SCORE_PRIOR | 256 | equivalent broad-profile observations |
| EXL3_MOE_CPU_SCORE_HYST | 2 | newcomer / evicted score ratio |
| EXL3_MOE_CPU_SWAP_MAX | 32 | M global swaps per round; 0 disables transfers |

Score requires CADENCE=exact and BUDGET_SCOPE=global; invalid values reject load. Configure
before import/model creation. `EXL3_MOE_CPU_SPLIT_STATS` initializes any exchange policy and
provides the score prior. All knob meanings and score normalization appear in DESIGN.md.

To reproduce extraction/grid/shift from the operator's complete trace directories:

```sh
python sim_policy.py --extract --run-r869 /actual/trace-r869/run --run-r860b /actual/trace-sample/run
python sim_policy.py --grid
```

Use a fresh packet copy/output directory to regenerate caches when input traces change;
existing caches are reused. Grid progress flushes `grid.jsonl`. The package includes a real
512-call slice for CPU selector parity; full caches/results are in the parent `out/` delivery.
Repricing sensitivity is already in `sensitivity.json`. Replace calibration assumptions
with measured R872/r3 values in `cost_calibration.json`, then recompute; do not refit an
optimistic coefficient to evaluation outcomes and call it an independent prediction.

R870 routing is absent. Capture actual text/checkpoint routes for its fresh-boot word-list
filler + probe-kind generations, including prefill/requests/positions, and replay those
before claiming recovery of the ~21.7 tok/s case. The supplied shift recovery is only a
nonstationary stream proxy; both directions and censored recovery are documented in POLICY.md.

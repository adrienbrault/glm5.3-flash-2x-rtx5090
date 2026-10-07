# cheapswap overlay, revision 3 (served since R882c, 2026-10-07)

Image `tabbyapi:cheapswap-r3`, built `FROM tabbyapi:r828-prompt-lookup-r3` (ExLlamaV3 1.5.2 with a private patch stack; `source_hashes.json` fingerprints the three files this overlay expects). Python only, no native rebuild.

## What it changes

ExLlamaV3's dynamic placement moves frequently routed experts into the GPU slots of the CPU split by rereading both experts from the checkpoint during an all-device sync, between generations. This overlay adds an exchange mode (`EXL3_MOE_CPU_SWAP_MODE=exchange`): a swap copies the hot expert's bytes from the CPU worker's pinned arena to the GPU slot and the cold expert's bytes back, with no checkpoint read. R872 measured 0.22 ms per expert host-to-device and 0.46 ms per exchange pair on one PCIe 5.0 x8 link. Revision 3 adds two things on top of revision 2: initialising the placement from a per-layer counts file (`EXL3_MOE_CPU_SPLIT_STATS`, the same format as static placement) and a second selection policy (`score`). The served policy is `histogram`, the revision 2 selection.

Launcher knobs (`scripts/launch-glm53.sh`): `SWAP_MODE=exchange`, `SWAP_POLICY`, `SWAP_INIT_STATS`, `SWAP_CADENCE`, `SWAP_INTERVAL`, `SWAP_MAX`, `SWAP_SCOPE`, `SWAP_FLOOR`, `SWAP_HYST`; exchange needs `PINNED_ARENA=1`.

## Files

- `cheapswap.patch`: the patch against the base image's ExLlamaV3 package; `implementation/` holds the patched files; `apply_image.py` applies it with `patch --fuzz=0` and checks `source_hashes.json` before and `patched_hashes.json` after.
- `landing_assert.py`, `cpu_native_selftest.py`, `gpu_selftest.py`, `gpu_ring_selftest.py`, `probe_dma.py`, `probe_static_initialization.py`: the gates. R872 and R875b ran them on both cards (results in `bench/results/r872-glm53-cheapswap.md` and `bench/results/r875c-glm53-cheapswap-r3.md`).
- `sim_policy.py`, `sim_cost.py`, `cost_calibration.json`, `sensitivity.json`, `selected.json`: the replay simulator and the cost model that chose the arms. `DESIGN.md`, `POLICY.md`, `VALIDATION.md`, `REVIEW-r3.md`, `HOW-TO-RUN.md`: the design, the policy results, the review and the operator runbook, as delivered.
- `*.env`: the arm environments of the A/B driver `ab_launch.py`.
- `r860b-split-stats.json`, `r869-split-stats.json`: counts files reconstructed from the R860b and R869 traces (`HOW-TO-RUN.md` explains how they relate to `scripts/split-stats-*.json`).

## Differences from the package on the box

`replay-slice.npz` (a 258 kB slice of 512 decode steps of the R860b routing trace, written by `sim_policy.py`) is left out of this repository, and its line is removed from `SHA256SUMS`. Without it, `RealTraceReplayTests` in `tests/test_score.py` fails with a missing-file error; the other tests and the image build do not read it. Every other file is byte-identical to the package the image was built from.

Written by an OpenAI Codex agent from the ExLlamaV3 sources; ExLlamaV3 is MIT-licensed by turboderp (`THIRD_PARTY.md`).

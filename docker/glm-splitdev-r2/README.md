# Per-device CPU expert count, revision 2

`tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2` is `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2` (`../mtp-overhead-r2/`) with `splitdev2.patch` applied to five Python files of ExLlamaV3 and one new file, `exllamav3/model/moe_split_check.py`. It is the served image since 2026-10-08 about 13:37 UTC. No native extension is rebuilt and no TabbyAPI file changes.

`EXL3_MOE_CPU_SPLIT_BY_DEVICE=N0,N1` sets the number of routed experts per MoE layer that run on the CPU by the CUDA device the layer loads on: N0 for the layers on GPU0, N1 for the layers on GPU1. Without it every layer takes the one count of `moe_cpu_split`. With the layer split, GPU0 kept 1,347 MiB unused after warmup at 104 experts per layer on the CPU while GPU1 kept 365 MiB (R914), so GPU0's layers can hold more experts. The served value is `100,104`: layers 3 to 22 and the MTP layer 45 run 100 of 288 experts on the CPU, layers 23 to 44 run 104.

Revision 1 was the override alone, in `modules/block_sparse_mlp_cpu.py`. It loaded at `98,104` and the first long warmup request (5,590 prompt tokens) failed with a device-side `index out of bounds` assert (R915b). `DIAGNOSIS.md` gives the mechanism: the autosplit first placed layer 23 on GPU0 with 98 CPU experts, then moved it to GPU1 with 104, and the CPU worker kept the first registration of that layer, so `counts1.scatter_add_` in `model/moe_cpu_host.py` received indices up to 104 for a 99-entry histogram. Revision 2 includes revision 1 and gives a layer whose CPU slice changes a new worker registration; the old one stays in the pinned arena unused until shutdown. It releases the GPU tables and the split-layer count of the rejected attempt. Without `EXL3_MOE_CPU_SPLIT_BY_DEVICE` the original code paths run.

`EXL3_MOE_CPU_SPLIT_CHECK=1` checks the per-layer index ranges before the scatters and gathers of the CPU path and raises with the layer and tensor name. It reads values back from the GPU on every forward, so measurements run with it off; the served configuration leaves it off.

## Build

The Dockerfile checks the SHA256 of the five original files against `base-sha256.txt`, applies the patch with `--fuzz=0` (dry run first), checks the six resulting files against `src-sha256.txt`, and runs `image-checks.sh`: the landing assert `landing.py` (it imports torch before the extension) and the 25 tests of `test_splitdev.py` with real CPU torch.

```sh
sha256sum -c SHA256SUMS
docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2 .
```

The launcher passes `EXL3_MOE_CPU_SPLIT_BY_DEVICE` through `EXL3_EXTRA` (`../../scripts/glm-daily.env`), and `../../scripts/glm53_verify.py` accepts the two counts when it checks the registrations at boot.

## Measurements

- R915c, 2026-10-08, results directory `2026-10-08-r915c-splitdev2-131018`, `GPU_SPLIT=31.8,31`, the served settings otherwise: `96,104` does not boot; `98,104` boots with GPU0 587 MiB free after warmup, `100,104` with 847 MiB free; GPU1 365 MiB in both. The arm with the check on (`98,104`) raised nothing during warmup. ([`../../bench/results/r915c-glm53-splitdev2.md`](../../bench/results/r915c-glm53-splitdev2.md))
- R925, same day, results directory `2026-10-08-r925-glm53-splitdev-abab-144311`: `100,104` and `98,104` alternated twice in one session; the sum of four streams with distinct prompts was 92.0 and 90.9 tok/s at `100,104` against 90.5 and 89.6 at `98,104`. ([`../../bench/results/r925-glm53-splitdev-abab.md`](../../bench/results/r925-glm53-splitdev-abab.md))

## Files

- `splitdev2.patch`, `base-sha256.txt`, `src-sha256.txt`: the patch (`diff -ruN base src`) and the digests of the files before and after it.
- `landing.py`, `image-checks.sh`, `test_splitdev.py`, `CPU-TESTS.txt`: the landing assert, the build-time checks, the CPU tests and their output before the image build, delivered as `tests-after.log` (24 passed; the real-torch test skipped where torch was not installed, and the image build runs it).
- `DIAGNOSIS.md`, `OPERATOR.md`: the round's diagnosis and run plan, kept as delivered. They cite boot logs, a probe image and a saved request that are not published.

# DSA indexer ring, revision 3

`tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3` is `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2` (`../glm-splitdev-r2/`) with `ring3.patch` applied to 17 files of ExLlamaV3 and two new files, `exllamav3/cache/mla_index_ring.py` and `exllamav3/cache/mla_index_shadow.py`. Three of the changed files are native (`exllamav3_ext/bindings.cpp`, `dsa_topk.cu`, `dsa_topk.cuh`), so the image build recompiles the extension. It is the served image since 2026-10-09 about 07:15 UTC, with `INDEX_RING=1`. No TabbyAPI file changes, and the six files of `../glm-splitdev-r2/` are unchanged (`SPLITDEV-PRESERVED.txt`).

GLM-5.3's DSA attention layers keep, beside the KV cache, a per-token indexer plane (fp16, 512 bytes per token per layer) from which the 4-token pooled indexer keys are built; decode scores the pooled keys only. With `EXL3_DSA_INDEX_RING=1` (the launcher sets it from `INDEX_RING=1`) the per-token rows live in a fixed ring per layer and request slot, sized for one prefill chunk plus the rows a speculative rewind needs: 2,304 rows per slot, 4 slots, 4.5 MiB per layer. The pooled keys stay in the paged cache in full. At the 262,144-token pool the engine log at boot (not published) reports, per DSA layer, `paged_bytes_removed=134217728 ring_bytes=4718592 net_bytes_saved=129499136`, about 124 MiB per layer; each GPU holds 6 of the 12 DSA layers (the trunk's 11 and the MTP layer). The served configuration spends that VRAM on 4 more routed experts per MoE layer on each GPU (`EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,100` instead of `100,104`).

`EXL3_DSA_INDEX_RING_SHADOW=1` (requires the ring) runs the paged indexer beside the ring and compares their keys, gates, pools, scores and top-k selections byte by byte; a mismatch prints a `RING-SHADOW-MISMATCH` line with the layer, slot, position and both values. It keeps both indexers in VRAM and is for verification boots only. Both flags default to 0 in the image; with both at 0 the original code paths run.

Revision 1, on an older image, passed its kernel tests in R891b (`2026-10-08-r891b-glm53-index-ring-051628`) and was not measured further: that round's exactness gate compared two boots, and two boots did not produce identical output at the time (R916). Revision 2 with its fix FIX1 is the code R919 measured on the image served before the per-device count (`../../bench/results/r919-glm53-index-ring.md`). Revision 3 is revision 2 and FIX1 ported onto the splitdev2 image: its 19 changed or new files are byte-identical to revision 2's (`PORT.txt`). Revisions 1 and 2 are not in this repository.

## Build

The Dockerfile copies the package to `/opt/index-ring/` and runs `install.sh` from `/app`: it checks the 17 original files against `SOURCE-SHA256SUMS`, applies the patch with `--fuzz=0 -p2` to the resolved package directory (GNU patch rejects a symlinked `-d` root), checks the 19 results against `PATCHED-SHA256SUMS`, parses every Python file, recompiles the extension when `NATIVE-CHANGED.txt` is non-empty (`rebuild_extension.py`, `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=4`), runs the landing assert (torch before the extension, both flags off, `BC_MLAttention` and `dsa_topk_shadow` present, splitdev2 still importable) and the seven CPU test programs in `tests/`.

```sh
sha256sum -c SHA256SUMS
docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3 .
```

The GPU tests `tests/test_kernels.py`, `tests/test_graph_shadow.py` and `tests/test_dispatch_shadow.py` run in the built image with `--device cuda:0` or `cuda:1`. The graph-shadow test corrupts 252 bytes on purpose after its clean probe to check that the gate detects them; its last summary line therefore reports 252 mismatches.

## Measurements

- R929c, 2026-10-09, results directory `2026-10-09-r929-glm53-ring-splitdev-061402`: kernel tests passed on both GPUs. Served traffic for 15 minutes with the ring and the shadow at `108,116` (the shadow did not fit at `100,104`, and `108,112` left GPU1 61 MiB free): 52,628 indexer calls, 428,445,793 rows compared, 0 mismatched bytes, all 12 DSA layers. In one session against the configuration served before it (`100,104`, ring off, one boot before and one after): c1 score 66.3 against 64.4 and 64.6 tok/s, 15.25 against 15.68 and 15.63 ms per token on the server, sum of four streams with distinct prompts 91.4 against 90.2 and 85.8 tok/s; free VRAM after warmup 1,073 and 549 MiB at `96,100` against 847 and 365 MiB. ([`../../bench/results/r929-glm53-ring-splitdev.md`](../../bench/results/r929-glm53-ring-splitdev.md))
- R919, 2026-10-08, revision 2 on the earlier image: 446,959,710 rows compared under served traffic, 0 mismatches. ([`../../bench/results/r919-glm53-index-ring.md`](../../bench/results/r919-glm53-index-ring.md))

## Files

- `ring3.patch`, `SOURCE-SHA256SUMS`, `PATCHED-SHA256SUMS`, `NATIVE-CHANGED.txt`: the patch (`diff -ruN base src`), the digests of the files before and after it, and the native files it changes.
- `Dockerfile`, `.dockerignore`, `install.sh`, `rebuild_extension.py`: the build.
- `tests/`: the CPU tests (`test_cpu`, `test_state`, `test_shadow`, `test_graph_fixture`, `test_checker`, `test_r929`, `test_unit_offline`), the GPU tests, the shadow workload `serve_workload.py` and its checker `check_shadow.py`, a fixture of the daily configuration before R929 (`daily.env`). `tests/baseline/modules/attention_fn/dsa_triton.py` and `mla_triton.py` are the unpatched ExLlamaV3 files of the base image (MIT), which `test_cpu.py` compares against.
- `CPU-TESTS.txt`: the output of the seven CPU test programs before the image build, 40 tests, delivered as seven `test_*.log` files and joined here in that order.
- `PORT.txt`, `SPLITDEV-PRESERVED.txt`: the port check against revision 2 and the list of splitdev2 files left unchanged.
- `r929_helpers.py`, `r929-glm53-ring-splitdev.sh`: the round's helper (arm environments, effective-environment check, probe checks, the ABA summary) and the packet's revision of the driver; the Dockerfile copies both into the image for `test_r929` and `test_unit_offline`. The helper here is the revision R929c ran with (shadow pairs `108,116` and `112,120` added after R929b). The driver that ran R929c is [`../../scripts/r929-glm53-ring-splitdev.sh`](../../scripts/r929-glm53-ring-splitdev.sh): it adds the shadow arm's walk-up over `108,112`, `108,116` and `112,120` to this revision. The served image was built from the first revision of the packet; the helper and driver changed afterwards, the patch and the tests did not.
- Left out of the delivered package: the operator guide `OPERATOR.md`, the validation records `VALIDATION.txt`, `PATCH-CHECK.txt` and `artifacts.log`, and `make_artifacts.py`, which name the authoring machine's tools and workspace; `final-message.txt`, which contains local paths; `last.txt`, a summary of the above.

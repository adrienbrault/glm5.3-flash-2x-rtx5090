# Hostile review of cheapswap-r3

**Verdict: no Critical or High finding established. r3 is fine as-is for isolated operator GPU validation; no r3b is required under BRIEF.md's severity gate. Use r3 with static initialization and the r2 B-safe histogram before comparing score. GPU correctness and throughput have not been demonstrated in this workspace.**

No GPU, SSH, Docker or git was used. All review outputs are in `out/`; input packages and the pristine tree remain unchanged. This review concerns the delivered patch, not the earlier simulation's claimed optimality.

## Findings ranked by severity

### F1 — Medium: score incurs work even when no expert moves

`r3/implementation/modules/block_sparse_mlp_cpu.py:273–277` multiplies the full decode histogram on every score decode submission, before knowing whether any future proposal will be admitted. For 42 layers this adds 42 CUDA vector kernels per Python decode call, including with `SWAP_MAX=0`. Prefill also clears a separate histogram at the start of a new prefill run (`model/moe_score.py:81–88`), even with PREFILL weight zero. These are asynchronous launches, not new per-call host waits, but launch/stream latency can erase a small CPU-share gain.

With positive budget and no admitted swaps, the unchanged r2 sweep still fences each visited layer and waits its commit event; score additionally copies prompt counts to the host (`model/moe_score.py:94–96`) alongside map and decode counts, even with zero prompt weight/expired prompt. Concrete case: a static-initialized placement is already good, k=32 fires, and hysteresis rejects every pair. Each layer still pays reader fences, blocking D2H copies, sorting and commit synchronization. **With MAX=0, r2 breaks before fencing/selecting any layer; it still performs health/cadence bookkeeping and the per-call decay remains.** Do not conflate these two no-swap cases.

Disposition: disclosed performance limitation, not a tensor correctness failure. The delivered policy probe measures isolated decay launches; use full-model `C-score-noswap.env` versus the same-profile static arm for exposed overhead, and a positive-budget zero-admission workload for sweep cost. Histogram does not acquire score's per-call decay/prompt work.

### F2 — Medium: supplied GPU tests do not test the new initialization

The loader test in `tests/test_score.py:51–79` executes only the profile-reordering statement slice and `cpu_post_load` with NumPy doubles. It does not call real worker registration, load GPU Linears, build native GPU pointer tables, attach the pinned arena, or execute streamed prefill. The replay likewise substitutes byte exchange. `gpu_selftest.py` and `gpu_ring_selftest.py` explicitly construct identity maps and identity initial expert lists; neither calls `load_cpu_split`/`cpu_post_load` with SPLIT_STATS. `policy_gpu_probe.py` only times vector multiplication.

Concrete missed failure: an incorrect loaded tail key/aux order or GPU pointer-list order could leave a perfectly valid inverse permutation pointing to the wrong expert. All delivered CPU selector tests and inherited identity-initialized GPU tests could pass. This is a validation gap, not evidence that r3 actually misloads bytes.

Disposition: independent full-loader CPU doubles below verify the key/order wiring; `out/probe_static_initialization.py` provides an operator-only full-model load/byte/native-map/streamed-prefill probe. It is syntax-checked, not GPU-executed. Run it on both native/swizzled layouts and folded/unfolded reconstruction before serving A/B, in addition to the inherited transaction/ring probes and full-model lifecycle checks.

### F3 — Medium: an incomplete profile silently produces mixed initialization

`modules/block_sparse_mlp_cpu.py:581–597` warns and uses identity-tail placement plus a zero prior when a layer is missing or its vector has the wrong length. There is no aggregate all-layer coverage assertion. Concrete case: a host file uses `model.layers.*` rather than `model.language_model.layers.*`, or omits layer 44 because it was collected only during prefill. The model loads and the experiment labelled “static-initialized exchange” actually has wholly/partly identity placement. Its bytes/maps remain consistent, but the baseline and recovery comparison are invalid.

Both bundled profiles independently have **exactly 42 keys, text layers 3..44, and 288 finite nonnegative counts each**, including the last layer. There is no delivered-profile coverage defect. Validate the *host* file against those exact keys; the operator probe fails on incomplete coverage. The fallback is documented, so this does not justify a High fix.

### F4 — Low: static equivalence has a valid-input/configuration boundary

With mode unset/checkpoint, the complete disabled executable AST matches the served tree, including SWAP=0 + SPLIT_STATS. With exchange explicitly set and SWAP=0, valid finite numeric counts also give the same expert order and router/aux permutations as served static placement; initialization-only metadata is added and no dynamic map is created. However exchange additionally validates counts/computes a prior, whereas served static code simply sorts them. A length-correct negative/NaN profile accepted by the served sorter can reject exchange-mode load. This is fail-closed input validation, not exact equivalence for arbitrary malformed files. The inherited exchange host also requires PINNED_ARENA=1, even for SWAP=0; use checkpoint/unset mode for the served static control.

Disposition: no fix required. Numerical/inference operation equivalence for valid profiles is supported; byte-identical modules or identical startup timing is not claimed.

## r2 transaction, fence, poison and scratch audit

`model/moe_exchange.py` and `model/moe_cpu_host.py` are **byte-identical** between r2 and r3. Thus worker descriptor/layout acknowledgment, registered arena handling, reader/copy-stream fences, worker-health checks before/after fences and commits, all-host poisoning, scratch preallocation, partial-enqueue drain/retention, and resident/streamed unfolded-scale refresh are unchanged. No native file changes in either patch.

Integration differences are initialization, score state/phase accounting, selector dispatch, and score-only cadence/prompt-expiry handling. `_split_swap_experts` retains the same exchange branch and legacy checkpoint body. Histogram `_split_sweep_layer` retains its r2 floor/hysteresis/tie order, map publication and 0.5 decay. Score returns through the same r2 `run_sweep`: actual byte writes precede map publication; its publication or later-layer exception poisons all affected hosts. The score publication-failure test exercises that boundary. No new exception path permits a failed transfer to resume inference.

The r3 change around inline sweeps sets/restores `_score_inline_sweep` only when score mode is selected, to distinguish a sweep from generator queue drainage. It does not change r2's serialized-producer or stream-tail proof. Existing r2 limitations remain: externally replayed whole-model captures bypassing Python cadence, independently concurrent producers, allocator reserve, global-budget starvation and full lifecycle validation.

## Static-initialization source trace

1. Counts are sorted descending, with ascending router ID on ties. `load_cpu_split` saves original lists, reorders gate/up/down lists by the same permutation, and retains original router order for dynamic exchange by clearing `_split_perm` and saving `_split_initial_order`.
2. Worker registration receives the reordered **tail keys** and correspondingly reordered GPU auxiliary mirrors. Child `fetch` preserves key order, completes deferred loading before copying, and `rehome_experts` writes each expert's gate/up/down trellis block plus separately allocated aux in that order. Descriptor and block arrays retain local-slot order. Swizzle only changes bytes *within* a projection; it does not reorder expert slots.
3. The GPU module tree removes that reordered tail by object identity. Normal `Module.load` loads exactly the remaining head objects, even though module traversal remains in original key order. `load_local` builds MultiLinear/BC pointer tables from the **reordered head lists**, so pointer-table slot matches the permutation. It does not use module traversal order as slot order.
4. `cpu_post_load` keeps router weight/bias, correction bias and per-expert routing scales in original ID space, and sets `map[order[slot]]=slot`. This is a permutation and its inverse is the expert order. Native `moe_split_issue_kernel` and `moe_split_map_kernel` increment counts in original router space, then overwrite selected IDs with physical slots. GPU dispatch subsequently consumes physical slots; CPU receives local tail slots.
5. Streamed prefill counts/grouping use those local CPU slots. The pinned path indexes `layer_blocks[local]`, DMAs the matching trellis block, unswizzles it on the copy stream into native tile order, and pairs it with `aux[projection][local]`. Thus initial permutation composes with pinned DMA and GPU unswizzle. K8 remains exempt from swizzle. No initialization-specific exchange or scratch is needed.

All 42 registered target layers follow these same hooks, including layer 44 even though normal prefill stops before it. This establishes source-level wiring and CPU-double behavior, not observed CUDA/worker byte equality. The operator probe resolves the latter.

## Independent validation

- Delivered CPU suite: **38 discovered, 35 passed, 3 skipped** in this relocated workspace (`out/r3-cpu-tests.log`). The three workspace-relative source comparisons look two parents above the packet and therefore miss this layout; they are not GPU tests or failed implementation tests.
- `out/audit_cpu.py`: complete default-off executable AST comparisons, exact r2 helper/host equality, whole loader with worker-key/aux/GPU-object-order doubles, valid-profile SWAP=0 served-static comparison, both profiles through the full loader for all 42 layers, and all 50 manifest entries. All checks pass (`out/audit-cpu-tests.log`). No Torch/native/GPU operations are executed.
- Fresh zero-fuzz patch landing and exact hashes/untouched files: see `out/landing-tests.log`. Original patch packaging and SHA256SUMS are valid; no native rebuild is introduced.
- Operator probe: syntax-checked only. Full-model numerical/vision, cancellation, queue, c4, actual BC graphs, cross-device autosplit rollback, forced swaps/peak VRAM and separately fitted MTP remain hardware validation, as in r2.

No r3b directory is produced: the brief permits a replacement only for Critical/High, and none was established. The recommended initial GPU arm is r3 static-initialized **histogram**, using the exact B-safe environment in `out/last.txt`. Score remains an optional measured comparison.

## Exact operator probe for the untested GPU initialization

`probe_static_initialization.py` loads the actual model through `Config.from_directory`, `Model.from_config` and `Model.load`. It checks every GPU and worker expert's trellis/scales/bias against original checkpoint keys, the auxiliary mirrors, original router columns, the inverse maps and the actual CUDA translation kernel for all 42 layers. It then submits CPU-tail selections through production `cpu_split_submit` and pinned streamed prefill on one layer per loaded GPU, compares against an independent parent native CPU layer from checkpoint tensors, and compares the streamed GPU native trellis bytes exactly. It forces a huge sweep interval to inspect initialization before any exchange, and defaults STREAM_FUSED_T=0 so the reconstruction cache actually runs. It does not run full-model logits or replace BC/vision/lifecycle tests.

Run on free GPUs in the built **r3** image. Supply the real checkpoint directory, the intended host profile, and the matching loader JSON from the existing serving configuration. No checkpoint location or VRAM reserve is available here; the required JSON argument intentionally avoids inventing those values. The loader JSON must use layer split, include the operator's actual reserve/use limits, chunk and batch sizes, and budget at least 128 rows.

```sh
# Run from this review workspace on the operator host.
# Export MODEL_DIR, SPLIT_STATS, and LOAD_KWARGS to the actual values first.
: "${MODEL_DIR:?actual host checkpoint directory required}"
: "${SPLIT_STATS:?actual host profile file required}"
: "${LOAD_KWARGS:?matching Model.load keyword arguments as JSON required}"
docker build -t tabbyapi:cheapswap-r3 r3
for swizzle in 0 1; do
  for folded in 0 1; do
    docker run --rm --gpus all --ipc=host --entrypoint python \
      -v "$MODEL_DIR:/model:ro" \
      -v "$SPLIT_STATS:/app/split-stats.json:ro" \
      -v "$PWD/out:/review:ro" \
      -e EXL3_MOE_CPU_SWIZZLE="$swizzle" \
      -e EXL3_MOE_RECON_FOLDED="$folded" \
      tabbyapi:cheapswap-r3 /review/probe_static_initialization.py \
      --model /model --stats /app/split-stats.json \
      --load-kwargs "$LOAD_KWARGS" --rows 128
  done
done
```

Expect 42 `bytes_and_map: PASS` records, one streamed numeric/native-trellis PASS per loaded GPU, and the terminal PASS line. A missing layer, wrong map or byte/aux order is an assertion failure, not a fallback. Requested swizzle may be unavailable on an ISA-capped machine; inspect each record's actual `swizzled` value. Repeat with `EXL3_MOE_CPU_MAX_ISA=avx2` for native child layout, and with `EXL3_MOE_STREAM_FUSED_T=256` to cover the normal streamed fused tier. Keep the inherited r3 ring and exchange probes, then the runbook's full-model lifecycle/A-B checks. This script was only syntax-checked here; its numerical tolerance is inherited from the r2 fixtures and does not certify full-model logits.

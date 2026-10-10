# GLM-5.3-Flash CPU/GPU expert exchange: c4 placement investigation

Scope: local source inspection only, using the active `.py`/CUDA files under `src/exllamav3`; `.orig` files are not the executing source. No network or SSH, no live-server access, and no source changes. The measurements and environment are those supplied in `BRIEF.md`.

## Findings

- Every dynamic split layer allocates its own float32 router-ID histogram and router-to-physical-slot map (`src/exllamav3/modules/block_sparse_mlp_cpu.py:223-247`). Histograms are not pooled globally; the module registry and swap budget are shared.
- Both the copy-path map kernel and fused issue kernel add one hit per selected router ID before translating it to a physical slot (`src/exllamav3/exllamav3_ext/routing.cu:824-831`, `:895-915`). Batch rows are not divided or normalized. The fused path counts even GPU-only selections before skipping CPU payload.
- The histogram policy calls `_split_swap_tick()` before submission, without a batch-size or prefill exclusion (`src/exllamav3/modules/block_sparse_mlp_cpu.py:306-345`). The tick is owned by registry entry zero, and the exchange tick returns during CUDA graph capture (`:733-755`). Kernel collection and Python decision timing are separate mechanisms.
- With the supplied settings, a candidate CPU expert must reach `max(2 * max(cold_count, 1), 4 * layer_histogram_mass / num_experts)`. At most 64 successful pairs are exchanged globally per sweep, in registry order. Visited layer histograms halve even if that layer makes zero swaps (`src/exllamav3/modules/block_sparse_mlp_cpu.py:787-815`; `src/exllamav3/model/moe_exchange.py:263-279`).

The graph trace rules out an internal-graph counter bypass for the supplied GLM implementation: `forward_ls()` calls the Python module every pass (`src/exllamav3/model/model_ls.py:418-442`), `TransformerBlock` calls its Python MLP (`src/exllamav3/modules/transformer.py:173-187`), and the split hook precedes the native expert call (`src/exllamav3/modules/block_sparse_mlp.py:1105-1109`, `:1497-1513`). The native fused routed path explicitly has no graph (`src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:134-140`); shared-expert and attention graphs are internal to those components.

**Leading explanation:** c4 histogram collection is neither excluded nor divided by four, but its **decision clock can be excluded**. The shared registry's first module alone owns the counter (`src/exllamav3/modules/block_sparse_mlp_cpu.py:243-247`, `:733-736`). The included initializer uses the same config for MTP and text (`src/exllamav3/model_init.py:225-228`), creates component `mtp` (`:263-267`), and loads draft before text (`:345-360`). If its MTP MoE layer is eligible for dynamic split, that layer registers first. At c4, `MTP_MAX_BATCH=1` suppresses draft and accept-prefill forwards (`src/exllamav3/generator/generator.py:792-827`, `:2185`), so the counter stops while text histograms grow. c1 resumes the owning module and causes sweeps of the whole registry, including text. **The TabbyAPI startup caller is not included**, so live load order must be confirmed before calling this the proven server cause. The comment in `src/exllamav3/model/model.py:531-533` says main-first; the executable initializer above says draft-first. Executable registration order, not that comment, decides.

**Second explanation:** even with a text-owned clock, pooling four different content distributions can keep every remaining CPU candidate below the relative FLOOR=4 threshold indefinitely. Sequential c1 can cross that threshold and leave a placement that subsequent c4 cannot undo. Both mechanisms fit the persistence; only live owner/tick and candidate-threshold diagnostics distinguish them.

**Configuration discrepancy:** the brief calls 96/102 GPU-resident counts. Here `EXL3_MOE_CPU_SPLIT_BY_DEVICE` overrides `split_k`, the CPU-tail count (`src/exllamav3/modules/block_sparse_mlp_cpu.py:160-180`), and `first = E - split_k` (`:568`). Thus GPU slots are E−96 and E−102. Resolve with the existing startup line ` -- CPU split experts (worker, dynamic): <key> [<first>..<E>) of <E>` (`:721-723`). No live model config or startup log was supplied, so E and the effective base split must be read from that log/config.

## 1. Histogram collection and paths

### What is counted

`BlockSparseMLP.forward()` flattens all token positions to `bsz = y.shape[0]`, so **bsz here means token rows, B×S**, not necessarily concurrent streams (`src/exllamav3/modules/block_sparse_mlp.py:1028-1033`). Routing produces selected IDs and weights (`:1061-1072`); the dynamic CPU split hook runs before the resident expert computation (`:1105-1109`). Each selected routed expert contributes **1.0**, independent of its routing weight or eventual acceptance. This counts resident GPU picks and CPU picks alike, in original router-ID space. Shared experts have no selection entry in this vector. With K picks per token, the histogram mass added by one layer invocation is B×S×K, assuming valid routing.

There are two accumulation implementations:

1. Small-row fused issue: `cpu_split_submit()` passes `_split_map` and `_split_hist` into the host (`src/exllamav3/modules/block_sparse_mlp_cpu.py:326-340`), which launches `ext.moe_split_issue()` (`src/exllamav3/model/moe_cpu_host.py:1077-1082`). CUDA loops over **all** `sel.numel()` entries, performs `atomicAdd(&hist[p], 1.0f)` on the raw router ID, then translates selections in place (`src/exllamav3/exllamav3_ext/routing.cu:895-907`, `:976-990`). The no-CPU-selected early return comes **after** these increments (`:909-915`), so GPU-only layers still collect statistics.
2. Copy fallback and large-row/prefill path: `_split_translate()` launches `ext.moe_split_map()` (`src/exllamav3/modules/block_sparse_mlp_cpu.py:341-369`). CUDA adds one hit for each raw selection and then writes physical/CPU-local IDs (`src/exllamav3/exllamav3_ext/routing.cu:824-831`, `:848-858`). No normalization by batch size, token width, content kind, or K occurs.

The fused path applies below `cpu_host.stream_min_rows`; its default is 32, and slot capacity defaults to 64 (`src/exllamav3/model/moe_cpu_host.py:82-105`). It declines incompatible shapes before launching, allowing the map fallback (`:1040-1043`). Larger batches use `_split_translate()` followed by streamed prefill (`src/exllamav3/modules/block_sparse_mlp_cpu.py:344-346`). That host path computes a separate **CPU-local, current-call** histogram to select experts for temporary GPU streaming (`src/exllamav3/model/moe_cpu_host.py:1338-1378`). This temporary histogram and the resident GPU prefill histogram in `modules/block_sparse_mlp.py:1190-1191` are not the exchange policy's `_split_hist`.

### Which phases contribute

| Phase/path | Exchange histogram | Decision clock under histogram policy |
|---|---|---|
| Plain target decode, including c4/c8 | All B×S×K selections at each eligible split layer | One tick only if that invoked layer is registry entry zero |
| Target MTP verification | All verification positions, including speculative positions later rejected | One owner-layer invocation, regardless of verification width |
| MTP draft forward | Its own eligible split layer's histogram; no additions to a trunk layer's vector | Ticks only if an invoked draft layer is registry entry zero |
| Prompt prefill | All positions at split MLPs actually reached | Same owner rule; histogram policy does not exclude prefill |
| MTP accepted-state repair prefill | Split MLPs actually reached, subject to last-KV early exit | Same owner rule |
| Internal attention/shared-expert/native expert graphs | Split collection remains outside these component graphs | Python split hook still runs each model pass |
| Externally captured whole Python forward, if a caller adds one | Captured map/issue kernel would increment on replay, provided the whole CPU handoff were capture-safe | Capture guard skips tick; Python tick does not replay. This is not the shown serving forward path |

MTP drafts call `draft_model.forward()` for each draft depth (`src/exllamav3/generator/generator.py:1352-1372`). Verification/plain generation always calls `self.model.forward()` (`:1724-1760`); neither call instructs the histogram to distinguish phase or accepted tokens. Accepted-state repair calls `draft_model.prefill()` only when MTP is active (`:2185`, `:2272-2275`). A prefill is bounded by the last KV module (`src/exllamav3/model/model_ls.py:372-403`); `TransformerBlock` can return immediately after attention at that boundary (`src/exllamav3/modules/transformer.py:159-161`). Consequently the final block's MLP may not be visited, including a one-block draft head's MLP on repair prefill. Do not assume every accepted token gets counted twice at every layer. Instrument actual reached MLP rows to resolve that on the live configuration.

The selected-ID kernel has no batch>1 or MTP-off exclusion. The relevant exclusions are: no dynamic map, ineligible/unloaded split layer, an MLP not reached by prefill, and autosplit measuring forwards (`src/exllamav3/modules/block_sparse_mlp.py:1105`; `src/exllamav3/modules/block_sparse_mlp_cpu.py:607-608`). `tid2eid_key` disables dynamic placement. Warmup uses real forwards and can add initial signal; autosplit measuring forwards skip the split hook (`src/exllamav3/model/model.py:202-221`; `src/exllamav3/model/model_ls.py:130-133`).

### Graph distinction

The shown LS driver loops through Python modules on every forward (`src/exllamav3/model/model_ls.py:407-447`) and prefill (`:361-404`). `TransformerBlock` invokes `self.mlp.forward()` (`src/exllamav3/modules/transformer.py:187`), which submits the CPU split before calling the resident expert backend. `BC_BlockSparseMLP::run_bszN` explicitly runs two fused launches with **nothing captured or patched** (`src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:134-140`). Shared experts can use their own internal graph (`:149-178`; `src/exllamav3/exllamav3_ext/libtorch/mlp.cpp:117-130`); the resident single-expert fallback can also capture internally (`src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:560-574`). Those calls follow collection/ticking and return with capture ended. Thus “c4 uses CUDA graphs” alone does not explain skipped exchange ticks in this source.

### MTP suspension and the shared-registry trap

The max-batch setting checks `len(self.active_jobs)`, which may differ from ready sequences. Above the cap, ready jobs become `_mtp_suspended`; the condition also honors that flag, making suspension sticky for those jobs (`src/exllamav3/generator/generator.py:783-802`). It requires a recurrent target (`:798-799`). c4 invokes target-only generation (`:825-827`), and accepted-state draft repair is skipped (`:2185`). Falling back to one remaining stream need not resume MTP for a job already suspended; **new** sequential c1 requests can resume it.

Each dynamic split layer appends itself to `config.infer_params.moe_cpu_swap_modules`, with no component filter or reordering (`src/exllamav3/modules/block_sparse_mlp_cpu.py:241-247`). The per-layer split budget uses `infer_params.moe_cpu_split` for both text and MTP; it is not restricted to component `text` (`:163-180`). By contrast, whole-layer offload explicitly has separate component budgets (`:138-140`). The GLM MTP head contains a `BlockSparseMLP` using the same expert dimensions/config (`src/exllamav3/architecture/glm5_next_mtp.py:30-36`, `:103-124`), so an eligible head can register before text. Separate worker processes do **not** imply separate histogram registries.

The included initializer establishes shared MTP config (`src/exllamav3/model_init.py:225-228`) and draft-first loading (`:345-360`). Every subsequent text tick returns at `reg[0] is not self` (`src/exllamav3/modules/block_sparse_mlp_cpu.py:733-736`). If `reg[0]` is MTP and MTP is suspended, no owner calls occur, no interval is reached, and no new pending sweep is created. A queue-drained hook cannot manufacture a missing decision: `run_pending_swap_sweeps()` immediately returns when pending is false (`:91-96`). This is a concrete mechanism for the measurement, conditional on live registry order and split eligibility. TabbyAPI's actual initialization code/logs are absent from this checkout. Initialization, reached prompt-prefill MLPs, a residual pending sweep, or brief unsuspended MTP activity can still cause occasional decisions around a c4 run; the claim is loss of the steady c4 decision clock, not proof that every startup/request-boundary sweep is absent.

## 2. Exchange cadence, thresholds, and c4

### INTERVAL=64

The clock counts **calls to the first registered module's split submission**, not requests, output tokens, CPU jobs, all layers, or aggregate batch rows. It is called before this submission's map/issue kernel (`src/exllamav3/modules/block_sparse_mlp_cpu.py:306-335`). Histogram policy also ticks on reached prefill MLP invocations. The docstring's “decode steps” wording (`:728-729`) is narrower than the executing histogram branch (`:318-323`).

With `exchange` + `exact`, the owner checks whether its prior count is at least 64, sweeps immediately if so, then increments (`:737-755`). Starting at zero, the first sweep happens on owner invocation **65**, before counting that invocation's selections. The exchange sweep resets the registry-zero counter (`src/exllamav3/model/moe_exchange.py:266`; `src/exllamav3/modules/block_sparse_mlp_cpu.py:771-772`); subsequent sweeps are 64 owner invocations apart. Capture returns without increment (`:738-739`).

The sweep is synchronous and inline at this boundary. It is not solely a request-end operation. `served` cadence, which is not the configured cadence, sets pending and defers until idle, with an overdue inline fallback at 4×interval (`:757-769`). Queue drain also executes an already-pending sweep in either mode (`src/exllamav3/generator/generator.py:874-878`, `:2312-2313`). Under exact cadence a full queue need not prevent adaptation **if its clock owner runs**.

If text owns the clock, ~6,000 ordinary c4 target steps provide ~93 sweep opportunities, apart from extra prefill calls and residual count. If suspended MTP owns it, those same text steps provide **zero** owner ticks. At c1, multiple draft-depth forwards can give several owner ticks per generator round if the owner is MTP; target-only ownership gives one per target verification forward. There is no accepted-token accounting.

### HYST=2, FLOOR=4, MAX=64, global scope

For each visited layer let E be routed expert count, H=sum(hist), hot the candidate CPU expert count, and cold the candidate GPU count. The executing condition is:

```text
floor_mass = 4 * H / E
required   = max(2 * max(cold, 1), floor_mass)
eligible   = hot >= required
```

This comes directly from `src/exllamav3/modules/block_sparse_mlp_cpu.py:787-805`. **FLOOR=4 means four times the layer's uniform mean, not four hits.** Equality passes. The `max(cold,1)` guard imposes at least two hits at HYST=2. Cold GPU experts are sorted ascending and hot CPU experts descending; disjoint pairs are examined in that order (`:799-804`). A threshold failure breaks, since subsequent hot counts cannot improve and cold counts cannot decrease. Failed compatibility exchanges return false and are skipped without incrementing successful swap count (`:807-808`; `src/exllamav3/model/moe_exchange.py:148-171`); this does not search all alternate compatible pairings.

MAX limits successful expert pairs, not bytes, tokens, decisions, or experts resident on a device. With global scope the registry is walked in insertion order with remaining budget `64-total`; reaching the budget breaks out before visiting further layers (`src/exllamav3/model/moe_exchange.py:263-279`). There is no global histogram, global candidate ranking, or round-robin fairness. A draft-first registry also gives that head first budget priority. Earlier text layers can consume the rest. The code accepts layer scope, but the supplied config is global.

### Can mixed c4 never cross the thresholds?

Yes, even with perfectly collected counts and a functioning clock. Once the histogram is large enough for the ratio/floor regime, collecting more copies of the same distribution raises both `hot` and `floor_mass` proportionally. Waiting longer does not overcome a relative-rate failure. At approximately constant routing rates, if an expert is selected with probability p per token and each token selects K experts:

```text
hot / (H/E) ≈ p * E / K
FLOOR=4 requires p >= 4*K/E.
```

For an **illustrative**, not source-proven live config E=256, K=8, the required per-token selection probability is 12.5%. An expert selected on 20% of one domain's tokens clears the floor (6.4×mean). If it is absent from three other equally weighted domains, its c4 probability is 5%, only 1.6×mean, so it fails indefinitely. The live E and K are config-derived (`src/exllamav3/architecture/glm5_next.py:93-94`); substitute their actual values. Hysteresis can independently reject a candidate even after the floor passes.

c4 gets four times the **raw histogram mass per plain decode call** relative to c1 plain decode, not one quarter. The effect is distribution mixing and a window measured in owner calls. Sequential c1 runs also separate content domains over time; the halving mechanism forgets old domains after a few sweeps. MTP verification uses more than one token row per target call and counts rejected paths; that changes observed traffic weighting relative to accepted outputs, but does not create a trunk-layer signal from the draft-head histogram. Neither mix nor speculation proves why the live 22% gain occurred without the diagnostic counts.

## 3. Decay, window, and persistence

For **histogram** policy, every layer actually visited by `_split_sweep_layer()` multiplies its GPU histogram by 0.5 after candidate evaluation, even on a zero-swap result (`src/exllamav3/modules/block_sparse_mlp_cpu.py:809-815`). There is no per-token decay, request-end histogram clear, time-based expiry, or finite sliding window. Score policy has separate prompt/prior terms and per-submit decay, but `settings()` returns None for histogram (`src/exllamav3/model/moe_score.py:9-13`); its HALF_LIFE/prefill/prior settings do not apply here (`:75-109`). The stats file initializes physical placement, while `_split_hist` starts at zero (`src/exllamav3/modules/block_sparse_mlp_cpu.py:223-236`, `:623-644`). Its prior is used by score policy, not the histogram sweep.

Let A be the new per-layer count vector between visits. Just before successive regular sweeps, `h_next = 0.5*h_previous + A`. With constant traffic, pre-sweep mass tends to 2×the latest interval's mass; old signal halves per visit. If the owner is text and every layer is visited, the history half-life is approximately **64 target split calls**, not 64 output tokens. At plain c4 that corresponds to 256 new token rows per interval; at c1 plain decode, 64. Large prefill chunks inject many rows for a single tick. If the owner is MTP, elapsed target calls between sweeps depend on draft scheduling; if MTP stops, there is no finite call/time half-life for text histograms.

Global budget exhaustion also matters: skipped later layers are neither evaluated nor halved, because `run_sweep()` breaks before calling them (`src/exllamav3/model/moe_exchange.py:270-277`). Thus they can retain older signal longer than earlier layers even while sweeps continue. MAX=0 similarly prevents all layer visits/decay.

**Placement itself never decays.** Maps and live GPU/arena contents persist across requests and idle periods until a successful later exchange or unload/reload. Idle only handles an already-pending sweep; it does not restore the stats profile (`src/exllamav3/modules/block_sparse_mlp_cpu.py:91-100`; `src/exllamav3/generator/generator.py:874-878`). Unload removes the module from the registry and clears map/histogram state (`src/exllamav3/modules/block_sparse_mlp_cpu.py:271-283`). Load reconstructs the profile-based order. Therefore both a frozen clock and threshold-driven no-op sweeps can preserve c1's improved c4 placement indefinitely.

## 4. Ranked explanations and the minimum confirming evidence

These are ranked by how directly the code explains the c4→c1→faster-c4 sequence. Multiple mechanisms can coexist. The throughput observations alone do not show map contents, owner identity, thresholds, or worker stall time.

### 1 — MTP owns the exchange clock and is suspended at c4

This is the strongest specific mechanism: draft-first shared registration is an executable path in the supplied initializer, and owner gating is unconditional. c4 fills text histograms but creates no decision events; c1 restores owner calls, sweeps all layers, and subsequent c4 preserves the new maps. ~6k c4 steps can remain inert without any threshold problem. The first c1-triggered sweep may use a large accumulated c4 text histogram; later sweeps progressively replace it with c1/verification signal. The observed speedup after ten c1 requests cannot distinguish which of those sweeps mattered.

**Existing logs:** with `EXL3_MOE_CPU_SWAP_DEBUG=1`, exchange mode prints even a **zero-swap** completed sweep:

```text
 -- exchange sweep: <N> swaps wall_ms=<...> fence_ms=<...>
 -- exchange bytes: layer=<key> bytes=<payload> shapes=<...>
```

These are at `src/exllamav3/model/moe_exchange.py:281-282` and `:162-164`. No sweep lines through steady c4, followed by lines during fresh c1, supports a missing clock. Startup split lines can suggest registration order, but logs before `cpu_post_load()` are not a definitive final registry dump.

**Smallest conclusive extra line:** once after both models have loaded, print:

```text
[SWAP-OWNER] owner=<reg[0].key> component=<reg[0].cpu_component> modules=<len(reg)>
```

Then print, once per 64 generator iterations **outside `_split_swap_tick`** and again at queue drain:

```text
[SWAP-CLOCK] ready_rows=4 no_draft=1 owner=<key> component=mtp owner_tick=<count> pending=0 sweeps=<cumulative_count>
```

Read owner via `self.model.config.infer_params.moe_cpu_swap_modules[0]`; component is stored on the module (`src/exllamav3/modules/block_sparse_mlp_cpu.py:677-678`). Use Python-only counters; do not read CUDA tensors for this test. Observe owner_tick unchanged and sweep count unchanged while text decode_steps advance, then changing during fresh c1. A diagnostic only inside the owner tick would never print during the fault. Capture current effective mode/cadence/interval and MTP cap in the same startup diagnostic. If owner is `text`, this hypothesis is falsified for this session.

### 2 — Mixed-domain rates fail FLOOR and/or HYST

With a functioning text clock, c4 can perform ~93 sweeps that all make zero changes. Four content kinds dilute domain-specific hot experts in the pooled histogram; sequential c1 raises their recent rates. After promotion, c4 need not have enough threshold-qualified reverse candidates to undo it. Initial profile/user placement can therefore be a suboptimal but stable fixed point for c4.

**Existing log:** repeated `exchange sweep: 0 swaps` during c4, and positive sweep counts during c1, establishes decisions occur but does not identify the rejecting condition.

**Smallest conclusive extra line:** in `_split_sweep_layer()`, after sorting and before evaluating pairs, log from the **already copied CPU tensors** (`src/exllamav3/modules/block_sparse_mlp_cpu.py:789-805`):

```text
[SWAP-CAND] layer=<key> budget=<remaining> H=<sum> E=<E> hot=<id>:<count> cold=<id>:<count> floor=<4*H/E> ratio_min=<2*max(cold,1)> eligible=<0|1> swaps=<N>
```

At least one line per visited layer per sampled sweep is needed; an early-layer sample cannot establish why all layers stay fixed. If the best pair fails, sorted ordering proves every subsequent pair in that layer fails the same selection test. `hot < floor` confirms the floor block; `hot < ratio_min` confirms hysteresis. Log both when both apply. Compare c4 with each c1 kind under the same initialized map. If qualified pairs exist but swaps stay zero, inspect compatibility next.

### 3 — Registry-order budget starvation, potentially compounded by MTP-first priority

MAX=64 global can be consumed by early layers; late layers then receive no decision and no decay (`src/exllamav3/model/moe_exchange.py:270-277`). This can limit adaptation and retain stale signal. It is weaker as a standalone explanation for “barely moved,” because exhausting 64 swaps per sweep implies substantial movement somewhere.

**Existing log:** frequent `exchange sweep: 64 swaps` and `exchange bytes` lines restricted to early registered layers support this mechanism.

**Smallest extra line:** at sweep completion, print `visited=<n>/<len(reg)> remaining=<64-total> last=<key> skipped=<n>`, or log each layer's incoming budget and completed swap count. Full-budget sweeps plus persistent skipped late layers confirm starvation. If total is repeatedly zero, budget exhaustion is not the cause.

### 4 — Shape/layout/auxiliary incompatibility rejects the selected hot/cold pairs

Exchange requires matching contiguous GPU and host shapes/dtypes and compatible mirrors (`src/exllamav3/model/moe_exchange.py:48-64`, `:148-171`). Uniform eligibility of the tail at load does not prove every resident/tail pair has identical quant widths. Threshold-qualified pairs can return false, and the algorithm does not rematch candidates across compatibility classes.

**Existing log:** a `-- exchange bytes` line describes an attempted exchange before compatibility returns. It does **not** prove a committed swap; compare with the sweep total. Some auxiliary-descriptor mismatches return before the byte line (`:151-155`), so absent byte lines do not rule incompatibility out.

**Smallest extra line:** on a false return, log `layer`, `r_cold`, `r_hot`, projection/field, both shapes/dtypes/contiguity, and a rejection reason. A per-layer `eligible_pairs`, `compatible_pairs`, `successful_swaps` counter is enough to separate policy rejection from copy incompatibility; the shape line identifies the fix. c1 may nominate different compatible pairs. This is a lower-ranked possibility until candidates demonstrably pass the thresholds.

### 5 — Throughput gain partly comes from a cause other than resident placement

The brief's interpretation is plausible, but throughput alone cannot isolate placement from actual batch width, prompt/cache reuse, context-dependent attention cost, worker contention, clocks, or other warm state. Three prior c4 runs make simple first-use warmup less persuasive. MTP drafting is still expected to be suspended on new c4 jobs under this config; it should not itself supply c4's direct speculative gain (`src/exllamav3/generator/generator.py:792-827`). These are controls, not source-proven alternative causes.

**Minimum measurements:** record successful swap totals and per-layer GPU membership before/after c1; then replay the same forced target inputs at c4 and measure CPU-selected assignment fraction and CPU collect stall. Membership can be computed by reading `_split_map` at a safe boundary: resident router IDs satisfy `map[r] < cpu_split_first`. Keep a stable membership digest and moved-ID count, not only a sum of map entries (every permutation has the same sum). A GPU-slot permutation alone is not enough: record **which router experts are GPU-resident**.

For exposed CPU wait, existing `EXL3_SPLIT_PROF=1` prints ` -- split prof [wait] ...` (collect/readback stream-time brackets, `src/exllamav3/modules/block_sparse_mlp_cpu.py:36-65`, `:403-410`). Compare the same B=4 inputs and context lengths. This profiling changes overhead, so use it for diagnosis, then reboot without it for speed measurement. Optional `EXL3_MTP_PHASE_PROF=1` provides mode/batch/depth (`src/exllamav3/generator/generator.py:803-809`; `src/exllamav3/util/mtp_phase.py:10-12`, `:108`), but DEBUG alone does not.

### Minimal live diagnostic order

1. Test boot with the supplied config plus `EXL3_MOE_CPU_SWAP_DEBUG=1`. Record startup split ranges, effective mode/cadence/policy, and final owner/component.
2. Run steady c4 with new jobs. Count completed sweep lines. If none, compare the owner's Python counter across c4 and fresh c1. If owner is MTP and frozen, cause 1 is confirmed; no histogram instrumentation is needed to establish the clock fault.
3. If c4 sweeps occur, collect best-pair/threshold lines for visited layers and budget/skip totals. This distinguishes causes 2–4.
4. To prove text histogram collection despite a frozen clock, make two safe, fenced snapshots of one target layer before/after N fixed-shape calls. Before any halving, expect ΔH=N×B×S×K (plus separately measured prefill/replayed rows). If sweeps occur, account for each halving rather than expecting that raw delta. An isolated probe with MAX=0 avoids both changes and halving, but changes the test configuration and still allows sweep-clock diagnostics.
5. Record placement membership and CPU wait before/after c1 to connect actual exchanges to the c4 speed change. Logs/diagnostic lines above are proposals where explicitly labeled extra; no such new lines were added to the serving source in this analysis task.

## 5. Minimal opt-in patch design and A/B plan

### Patch the decision-clock exclusion, not the already complete histogram

Propose **`EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1`**, default **0**, initially supported only for `MODE=exchange`, `POLICY=histogram`, `CADENCE=exact`. Reject unsupported combinations when explicitly enabled. If the owner diagnostic confirms MTP-first registration, this is the smallest targeted change: choose the first registered dynamic split module whose saved `cpu_component == "text"` as clock owner. Keep registry order, candidate ordering, histograms, weights, copy implementation, and global budget unchanged. Do not multiply c4 hits by four; every row is already counted.

Use one shared helper in `modules/block_sparse_mlp_cpu.py`, called by both tick gating and reset:

```python
def swap_clock_owner(reg):
    if text_clock_enabled:
        return next((m for m in reg if m.cpu_component == "text"), reg[0])
    return reg[0]
```

`text_clock_enabled` is fixed at process initialization from the new flag after validating mode/policy/cadence. The fallback preserves initialization/draft-only behavior before text layers exist. Resolve ownership from the current registry, so unload/rollback cannot leave a stale pointer. The practical changes are:

1. In `_split_swap_tick()` (`src/exllamav3/modules/block_sparse_mlp_cpu.py:733-736`), replace the `reg[0] is not self` gate with `swap_clock_owner(reg) is not self`.
2. In `moe_exchange.run_sweep()` (`src/exllamav3/model/moe_exchange.py:266`), reset **that same chosen owner**, not unconditionally `reg[0]`. Forgetting this is a real bug: once the text counter reaches 64, every subsequent text call would trigger another sweep. Queue-triggered sweeps use the same reset path.
3. Add the owner startup line and a DEBUG sweep counter/owner-tick field. Retain the existing first-registration counter initialization (`src/exllamav3/modules/block_sparse_mlp_cpu.py:241`). When ownership switches from draft-only initialization to text, the newly initialized text counter starts its own clock.

This restores one decision opportunity per 64 **target split invocations**, including existing histogram prefill behavior; c4 now advances the clock independently of MTP suspension. It does not redefine the interval as 64 accepted tokens. MTP draft histograms stay separate and may be evaluated with stale signal while drafting is off, because the full registry still sweeps; that is existing registry behavior and should be visible in logs. A future component budget/decay policy would be a separate change.

With the flag off, both gating and reset remain registry-zero behavior. With a text-first registry, it gives the same owner and should leave the execution schedule unchanged. With MTP-first, c1 sweep timing also changes from draft invocations to target invocations. Do not silently alter cadence, thresholds, budget allocation, or prefill weighting as part of this experiment.

If diagnostics instead confirm text-first ownership and threshold failure, this flag will not improve adaptation. First use the existing FLOOR/HYST settings for an isolated policy A/B, for example FLOOR=1 or 2 while retaining HYST=2, against the baseline FLOOR=4. A uniform batch multiplier or changing the interval alone cannot solve a relative-floor failure. Lower floor can increase churn and needs the latency/exactness validation below. No extra histogram collection patch is justified by this source. If an external whole-forward graph is later proven to bypass Python ticks, a host-side target-forward clock outside the graph would be needed instead; that larger design must suppress duplicate ticks and preserve the same pre-submission fenced sweep boundary.

### Exactness and safety requirements

There are two different claims to verify:

- **Bit-exact exchange of stored expert state:** preserve the existing transaction unchanged. It validates compatible tensor layouts; preallocates scratch before writes; copies GPU→scratch and arena→GPU; converts only physical trellis layout; updates arena, mirrors, and unfolded resident/streamed reconstruction scales; waits for transfer completion; then updates router mapping (`src/exllamav3/model/moe_exchange.py:53-106`, `:132-181`, `:226-249`). Reader fences cover current/recorded streams and prefill copy streams, with worker-health checks and fail-closed poisoning (`:109-129`, `:258-285`). Keep sweeps outside capture and before new CPU submissions. Fixed GPU tensor/map addresses must remain stable for native pointer tables/graphs.
- **Bit-exact model outputs across changed CPU/GPU placement:** the source does **not** establish this. It explicitly notes different device numerics (`src/exllamav3/modules/block_sparse_mlp_cpu.py:761-764`; `src/exllamav3/generator/generator.py:874-876`). CPU experts use int8 activations and their own accumulation implementation (`src/exllamav3/exllamav3_ext/cpu/moe_mul1.cpp:35-43`); preserving bytes does not prove identical arithmetic when residency changes. Exact cadence already permits changes mid-generation in exchange mode. A clock fix changes when those changes happen. Treat bit-exactness as a required test result, not a promise inferred from byte copies.

For a diagnostic boot, `EXL3_MOE_CPU_SWAP_VERIFY=1` checks map permutation (`src/exllamav3/modules/block_sparse_mlp_cpu.py:809-812`) and demoted host bytes/aux mirror (`src/exllamav3/model/moe_exchange.py:100-105`). It does **not** compare every promoted tensor to its pre-swap host source or prove post-swap logits equal pre-swap logits. Add a test-only pre-swap host snapshot and compare promoted tensors after inverse layout conversion, verify a swap-back restores both sides, and check all refreshed scales. Optional `EXL3_MOE_CPU_SPLIT_CHECK=1` validates layout/index ranges (`src/exllamav3/model/moe_split_check.py:39-70`); it also does not validate numerical equivalence.

For the clock-only change, compare flag off/on with identical initial maps and no successful swaps (e.g. MAX=0), forced input IDs, cache state, shapes, and sampling state. Logits should be bitwise identical because only Python timing/counters changed. Then force actual compatible swaps at c4: byte/state invariants must remain exact; compare fixed-input layer outputs and logits before/after relocation using strict equality. Also compare the new-clock implementation with an explicit manual sweep schedule producing identical placements at identical call boundaries. That isolates transaction correctness from a changed placement policy. If output equality across relocation fails, this analysis cannot certify the stronger bit-exact-output requirement; implementing device-arithmetic equivalence would exceed this minimal clock patch.

### c4 latency risks

The sweep serially fences each visited layer, copies histograms/maps to CPU for decisions, and can exchange up to 64 expert pairs. Transfers are bidirectional and synchronously completed (`src/exllamav3/model/moe_exchange.py:69-91`, `:270-279`). All active c4 streams can therefore see an inter-token stall, even when throughput eventually improves. The budget caps pair count, not total payload or milliseconds. DEBUG's `bytes=` is one expert payload; exchange moves data in both directions, plus GPU scratch/auxiliary work. Measure rather than infer an exact PCIe duration.

Even zero-swap sweeps incur layer fences/readbacks. More frequent c4 sweeps also accelerate decay compared with a frozen clock and can cause additional promotions/demotions. Scratch allocations create transient VRAM pressure; existing fail-closed behavior must stay intact. A draft-first global registry can still consume budget before text, so the clock fix does not guarantee fair adaptation. Do not enable per-layer budgets or larger MAX in the first A/B.

### Controlled A/B

1. Diagnose owner/candidates first. Save startup config, profile hash, final registry/component order, split ranges, and initial GPU membership. Use two fresh boots from the same checkpoint/profile and the same fixed prior-traffic script. A begins with flag=0; B with flag=1. Identical boot and warmup matter because maps and histograms persist. Do not run A then B in a shared warmed process.
2. Run the supplied cold c4 workload with exactly the same four content kinds, two rounds, 1,024 outputs per stream, thinking off, repeated three times. Record actual ready sequence count and MTP suspension. In B, a functioning text clock should show about one sweep per 64 target invocations; cold c4 should now be able to change maps without a c1 intervention **if candidates qualify**. A frozen MTP-owned clock should remain silent in A. If B shows only zero-swap sweeps, use the threshold diagnostics rather than declaring the patch ineffective at counting.
3. Then run the same ten sequential c1 requests with MTP enabled and rerun c4 twice. Compare the size of the additional c1 boost, cumulative swaps, GPU membership changes, per-layer CPU selection rate, and worker exposed wait. In the clock-fault explanation, the pre-c1 placement gap should narrow in B; a remaining c1 boost may reflect thresholds/domain mixing or verification weighting.
4. Also run c1 with MTP disabled in a separate controlled branch and same text inputs. This separates “one domain at a time” from “draft-head owner active.” Under the old MTP-owned clock this branch should remain unable to tick; under the new text-owned clock it should tick. Existing MTP suspension is sticky, so use fresh requests/boots when changing the policy.
5. Report aggregate output tok/s, per-stream inter-token latency p50/p95/p99 and worst gap, total wall time, TTFT, sweep count/swaps/wall_ms/fence_ms, and bytes/visited/skipped layers. Compare throughput with equal-length generated outputs; use forced-token replay separately for exact routing/numerical comparisons, because adaptive placement can change sampled continuations.
6. Complete byte, map, scale, fixed-input numerical, and failure-path checks on the debug boot; then reboot with VERIFY/CHECK/profiling off for timing, retaining only sampled DEBUG if desired. Accept the clock fix only if owner behavior matches the design, exchange invariants pass, and measured throughput gains justify c4 latency. If strict output bit-exactness is required, a strict equality failure blocks that acceptance regardless of throughput.

## Local validation and remaining uncertainty

A stdlib-only harness extracted and executed the **actual `_split_swap_tick` AST** with a fake non-capturing CUDA guard and a sweep/reset callback. It did not import Torch, execute expert math, or mutate serving files. With a shared registry `[mtp,text]`, 6,000 text calls produced zero owner ticks and zero sweeps; the next 65 MTP calls produced one sweep and left owner count at 1. Reordering the synthetic registry to `[text,mtp]` produced 93 sweeps over 6,000 text calls. The E=256/K=8 probability example was checked separately: 20% single-domain routing gives 6.4×mean; a four-way mixture gives 1.6×mean and fails FLOOR=4. A second AST harness applied only the proposed owner-gate substitution and matching reset callback to an MTP-first synthetic registry: flag=0 gave zero sweeps over 6,000 text calls; flag=1 gave 93 and left the text counter at 48. These checks verify control flow and threshold arithmetic, not the live server's registry, numerical exactness, or performance.

No GPU/model weights, TabbyAPI loader, live config JSON/profile contents, Docker logs, or live histogram/map snapshots were available in this checkout. The decisive missing evidence is: **final registry owner/component plus tick delta and sweep count during steady c4**. If that falsifies the clock explanation, collect **best hot/cold counts versus floor/ratio and remaining budget per visited layer**. Stored-state and model-output exactness require the explicit comparisons described above. No network/SSH was used; the requested patch is a design only, and source files remain unchanged.

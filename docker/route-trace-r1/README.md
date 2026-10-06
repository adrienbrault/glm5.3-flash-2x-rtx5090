# Route trace overlay, revision 1

The served source is untouched. `base/` holds the five original Python files; `src/` holds their replacements plus `exllamav3/route_trace.py`. `route-trace.patch` is exactly `diff -ruN base src`. No native extension rebuild. `SHA256SUMS` fingerprints the package, tests and instructions. Use a **fresh output directory for every server run**. Set env before importing the model and restart the server after applying the overlay.

## Operator landing

From this overlay directory, with PACKAGE_PARENT pointing to the served Python site-packages directory (containing exllamav3):

```sh
sha256sum -c SHA256SUMS
python3 landing_assert.py "$PACKAGE_PARENT"
patch --dry-run -p1 -d "$PACKAGE_PARENT" < route-trace.patch
patch -p1 -d "$PACKAGE_PARENT" < route-trace.patch
python3 landing_assert.py "$PACKAGE_PARENT" --landed
python3 -m unittest discover -s tests -v
```

Do not apply to a differently fingerprinted served tree. The mirrored `src/opt/venv/...` can be used as a landing target only if its assert passes. No changes are needed to TabbyAPI backend settings. The environment entry the R858 operator can append to its Docker env allowlist or pass with `--env-file` is in `docker-env.list`:

```text
EXL3_ROUTE_TRACE=/app/route-traces/r858
```

Mount a host output directory writable by the container at `/app/route-traces`; for example add `--mount type=bind,src=/ABS/HOST/route-traces,dst=/app/route-traces` and `--env EXL3_ROUTE_TRACE=/app/route-traces/r858` to the operator's existing invocation. These are runbook fragments, not Docker commands run in this workspace. Unset/empty env is off. Trace off does not import numpy/torch from the trace helper, allocate tensors/events, start a worker or mutate forward params; AST regression tests remove the gated blocks and establish that **every original statement is unchanged**. GPU bitwise identity when off still needs the operator check below.

## What is recorded

One uncompressed NPZ per executed MoE layer per model call, committed with atomic rename by a bounded background writer. `route-PID-WRITERTHREAD-SEQUENCE.npz` arrays:

- `meta`: scalar JSON Unicode; version=1, kind=actual, call=`origin_pid:call_counter`, component, phase, layer module key, device, layer_instance, batch, q, expert_space=checkpoint.
- `ids`: `[batch*q,topk]` int32 checkpoint expert IDs, before physical split translation. Static stats permutations are inverted by indexing the router-to-checkpoint permutation. Dynamic slots never leak into this stream.
- `weights`: identical fp16 router values (not renormalized), same shape/order as ids.
- `positions`: int64 logical cache token position, sequence start + column. -1 if unavailable.
- `request_keys`: Unicode generator job serial for each flattened token, empty if caller provides none. This identifies requests inside one generator instance, not an HTTP request UUID. `batch_rows` identifies sequence rows, including multiple rows for one request. Keep model component and process/call together when joining.

Model `prefill` and `forward` establish call context; target generator decode includes job serials and explicit decode/verify phase; Job.prefill sets serial and phase even for its MTP-driven target `forward`. Pipeline stage0 establishes new context for lookahead chunks using the captured cache start. Generic multi-column `Model.forward` is labelled `forward` because it may be verify or prompt; direct module calls have unknown metadata. Standalone draft calls record component and positions where available but request keys can be unavailable. `tp_warmup` and `autosplit_measure` are excluded. Other model warmup calls may be recorded: use dedicated trace runs after warmup or filter their empty request keys. Prefill stops at the final cache-writing attention in this engine, so a last-layer MoE that is not executed has no invented record.

CUDA recording enqueues **same-producer-stream**, nonblocking D2H into owned pinned tensors, records an event, and queues handles. This ordering protects c1 router statics from the subsequent in-place split map and from next calls. Only the writer thread synchronizes the event, maps static IDs, expands row metadata and writes NPZ. Host staging cache lengths are cloned when the call starts. There is no `.cpu()`, `.item()`, event wait or disk serialization on the GPU forward path. Costs while enabled include two small D2H enqueues, pinned allocation and event per layer, context copying and queue enqueue; it is a diagnostic run, not a speed benchmark. At c1 each record has roughly 80 bytes of route payload plus metadata/NPZ headers; prefill payload scales with rows. Small files can tax a slow volume. Queue capacity is 256 records; saturation or writer failure explicitly fails the audit instead of silently losing rows. A clean process exit drains and writes `complete-PID-THREAD.json` with record count. SIGKILL can lose queued records; absence of that marker is an incomplete audit. Already committed NPZ files remain usable for partial inspection.

No predictor is added in this minimal trace patch. The simulator schema allows additional `kind=predicted` records: call/layer refer to the **target** call and next layer; ids are ranked candidate IDs and weights their confidence scores, with the same positions/request arrays. DESIGN-PREFETCH.md specifies the independent scratch/HC/norm/router probe needed to produce these. Never substitute actual next-layer IDs for predictions.

Inner native/shared-expert graphs can remain enabled: the Python hook is outside them. An outer CUDA graph that bypasses Python forward hooks cannot emit per replay metadata; capture-time hook execution fails explicitly. Use ordinary Python-driven EXL3 forwards for the routing audit.

## Validation and flan probes

CPU tests cover the off gate and AST equivalence, token row metadata, event-before-write behavior, permutation recovery, atomic file commit, explicit error/full-queue handling, and placement of the hook after TP routing/before split mapping. These do not test real CUDA D2H ordering, kernel execution or GPU numerical parity.

R858: fixed token IDs, fixed placement (disable swaps), deterministic served kernels; compare baseline and patched env-off hidden states/logits using torch.equal and token outputs for c1/c4/prefill. Then env-on: compare same outputs, verify every expected **executed** layer/call/row is present and logged selections match a short synchronous debug dump. Include static stats permutation and dynamic swap cases. Force rapid back-to-back c1 calls to exercise reused routing buffers, and a large prefill followed by decode. Test both cards and, separately, TP deduplication by device. Stop server gracefully and check complete markers and queue/error stderr. Record trace-on overhead independently of production latency.

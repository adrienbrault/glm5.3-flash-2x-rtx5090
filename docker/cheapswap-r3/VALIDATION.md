# Validation performed here

- CPU only, Python 3.12 with NumPy 2.5.3 from an existing cached environment. No dependency
  downloads, GPU, Docker, SSH or git were used.
- 38/38 package CPU tests pass, no skips in this workspace. Includes the 27 inherited r2
  tests and 11 new score/initialization/replay tests. `cpu-tests.log` is the full log.
- Default-off AST specialization of both integration modules and the new phase-caller
  module matches the complete pristine executable ASTs (removing cached opt-in assignments
  and the new flag import). Legacy checkpoint-swap function body is unchanged.
- `model/moe_exchange.py` and `model/moe_cpu_host.py` are byte-identical to r2; native
  sources are untouched. Byte exchange and failure/fence tests are inherited unchanged.
- Real 512-call chronological R860b c1 slice, starting at call index 98, runs through the
  production `_split_sweep_layer` score branch and the unchanged r2 global `run_sweep`.
  CPU doubles stand in for tensor/CUDA operations and actual expert-byte writes. Per-call
  CPU picks and exchanges match the independent vectorized simulator exactly; maps remain
  permutations and the global cap is enforced. No autoregressive routing feedback is claimed.
- The actual loader's profile initialization statements and `cpu_post_load` are executed
  against doubles; physical expert order, inverse map and original-ID prior are verified.
- Real submit body redirects prefill histogram counts and decays only decode counts;
  False-valued prefill markers, inline sweep preservation and queue-boundary expiry are tested.
- Fresh copy of pristine tree: `patch --batch --forward --fuzz=0 -p1` succeeds; all five
  changed/added Python files equal generated `patched_hashes.json`; untouched files match
  pristine bytes. Landing assertion verifies syntax, default-off gates, score defaults and
  inverse profile map. `landing-tests.log` records this check.
- Every packet Python file was syntax parsed, including all inherited operator GPU/native
  scripts and the new policy-cost/profile-inspection probes. They were not run on hardware.
- SHA256SUMS is regenerated from actual packet files, excluding itself, last.txt and generated
  Python caches. The top-level out manifest covers the complete simulation and packet.

The six standalone simulator tests also pass (`../simulator-tests.log`). All 1,458 grid
policies and both shift directions completed with incremental output. Final trace audits
report zero incomplete decode calls discarded; legitimately partial prefill calls retained.
`../grid.jsonl` holds every row; `../trace-inputs.json` identifies cache hashes and input
completion-manifest hashes. Full traces were not copied into the packet.

Tests that compare with pristine/r2 sources skip when those inputs are absent in the
operator's standalone image. Their completed source-differential evidence is recorded here;
policy, replay, transaction-double and saved-cost arithmetic tests remain CPU runnable.

Pending operator validation: actual pinned arena/CUDA/native fixture and worker ring runs,
profile-initialized full model numerical/vision/graph/prefill/queue/c4 lifecycle, fitting MTP
accept/reject lifecycle, allocator reserve, live histogram-decay overhead, real D2H/DDR
contention and A/B/A2 throughput. HOW-TO-RUN.md provides the commands and conditions.

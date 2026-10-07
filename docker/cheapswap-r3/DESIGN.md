# R3 selection and initialization

`moe_exchange.py` and `moe_cpu_host.py` are byte-identical to reviewed r2. No native source,
transaction ordering, reader fence, arena layout, scale-cache refresh, poison boundary or
commit event changed. The pristine patch includes the whole r2 exchange package plus r3.

Profile initialization composes with both histogram and score exchange. Load expert lists
hot-to-cold as the static loader does, retain original router columns, and initialize
`router_id -> physical_slot` to the inverse profile order. Original router IDs also index
profile and online counts; deterministic router ties retain their original order. Without
exchange or without SWAP=1, the served loader behavior remains unchanged. Unset policy is
histogram: r2 selection, floor, hysteresis and 0.5 sweep decay are preserved.

The standalone `model/moe_score.py` selects the disjoint cold/hot expert pairs. It runs
inside r2's already-fenced `_split_sweep_layer` boundary, using the same `_split_swap_experts`
and map publication operations. Global MAX is enforced by r2's existing ordered sweep loop,
so later-layer starvation remains. Failed or incompatible pairs do not consume budget;
shape-rejected pairs are not reallocated during that round, as in r2. Fixed-route replay
assumes compatible projections; actual quantization-shape differences require the operator
probe. Publication failure still poisons the whole registry under r2's exception boundary.

For score, the existing float32 GPU decode histogram is multiplied by `2**(-1/H)` once per
Python decode call after the pre-call sweep trigger, before the existing count kernel.
There is no 0.5 sweep decay. Float32 replay follows this exact order. This adds one GPU
vector-decay launch per layer per decode call; its live cost is unmeasured. c4/verify rows
add all their uses but advance cadence once per model call, following r2's cadence contract.
Externally replayed whole-model graphs that bypass Python ticks are not supported.

Prefill uses a separate fixed-address float32 288-bin GPU vector: the existing fused/map
count kernel writes it instead of the decode vector. The original expert-counting kernels
remain untouched. The routed caller passes phase by **presence** of `params['prefill']`;
False still marks a prefill module. Consecutive prefill chunks accumulate counts and observed
rows. Queue-drain hooks expire the prompt prior, including no-sweep boundaries; inline
score sweeps preserve it. No router IDs are copied to the host at prefill. At an ordinary
fenced sweep only, 288 prompt counts are copied beside the existing map/decode counts.
The final MoE layer, not reached by prefill, uses zero prompt signal. Fully cached prompt
routing is unavailable; no historical cached-prefix counts are invented.

This prompt signal is a single most-recent prefill aggregate, suitable for the c1 replay.
Concurrent c4 arrivals/prefix reuse and draft/MTP phases need operator lifecycle checks;
there is no per-request history table, and this replay does not claim accepted-token cadence
or c4 speed. Module-local BC graphs retain all tensor addresses. Statistics across externally
rotated producer streams may differ; the engine's normal single-producer ordering is the
numerical replay contract. The byte-transfer and failure-fencing contract remains r2's.

Score defaults are the offline winner: exact cadence 32, H=512, wp=32, wb=256, rho=2,
global MAX=32. The startup settings helper supplies MAX=32 if absent before the unchanged
r2 sweep reads its environment. Explicit env settings override each knob. Invalid score
weights, half-life, scope or cadence fail load. Runtime environment toggling is unsupported.
Score needs exchange plus dynamic split; SWAP=0 is still static. A profile missing a layer
falls back to the served identity-tail initialization and a zero broad prior for that layer.

CPU tests cover real profile-loader statements and post-load map setup, the default-off
whole-module AST differential, the exact admission threshold and tie ordering, prefill
redirect/expiry, decode decay, fail-closed publication and a 512-call real trace replay through
the production selector and r2's real global sweep loop. CUDA and worker operations in that
replay use explicit CPU doubles. The inherited GPU/native fixtures are preserved for the
operator; they are not substitutes for full-model score/prefill/graph testing.

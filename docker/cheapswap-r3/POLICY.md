# R3 placement replay

The exhaustive replay completed before building the patch. Raw traces and the pristine/r2
packages remained read-only inputs. `simulation.log`, `grid.jsonl`, and trace cache files flush incrementally.

## Conventions fixed before scoring

- Decode c1 model calls use the first row of each actual text routing record, as in
  `r2/sim_cost.py`. Additional rows contribute to profile fitting only; trace audits report
  their count and a separate all-row static CPU-share check. Calls are grouped by `meta.call`.
- Router IDs are checkpoint expert IDs. The target budget is 192 GPU + 96 CPU experts per
  layer. Static placement selects the hottest 192, ties by ascending ID.
- Both directions fit the broad prior exclusively on the other trace's complete decode
  records, using all observed rows. Counts normalize to uses per row (sum 8 per layer).
  Generated `*-split-stats.json` files are real counts, compatible with the served loader.
- Prefill signal sums all observed actual chunks of the current prompt, normalized by
  observed rows independently for each layer. The last MoE layer is absent during prefill
  because the served path stops at the last KV module; its signal is zero. Cached prefix
  tokens have no fabricated routing. A new prompt replaces the prompt prior; decode memory
  and placements survive across prompts.
- Score: `D + wp * prompt_rate + wb * broad_rate`. `D` decays by `2**(-1/H)` once per decode
  call before new uses are added. Thus H is a half-life in model calls, not accepted MTP
  output tokens. Sweep before the next call after k completed calls; no free prompt swap.
- M is the **global** cap per round across the ordered 42-layer registry, just as in r2.
  Pair each layer's hottest CPU expert with its coldest GPU expert, accepting only strict
  improvement and `new >= rho * old`. Cold ties ascend, hot ties descend. Unvisited layers
  still decay at submission. Registry-order starvation remains a known limitation.
- Grid: k={8,16,32}, H={128,512,2048}, wp={0,8,32}, wb={0,64,256}, M={8,16,32},
  rho={1.2,1.5,2}; identity and static initialization. Total 1,458 policies / 2,916 directional evaluations.
- Compare static with B-safe (64/4/2/64 global exact) and B-fast (16/2/1.2/64 layer served).
  Main histogram baselines follow r2's decode-only replay. A labelled sensitivity injects
  prompt counts at first decode, approximating prefill cadence at one boundary; it is not
  an exact replay of every served prefill tick. Static-initialized histogram arms also shown.

## Cost calibration

Use `r2/sim_cost.py:price` unchanged, with the calibration stored in `r2/results.json`:
F=14.128555 ms, exposed CPU cost=0.050493888 ms/pick, alpha=0.385717203;
CPU demand=0.36/2.75 ms/pick. Anchors were 48.75 tok/s served, 39.1 interval64/floor4,
63.2 in-sample static, paired with chronological first-half R860b proxy route counts.
This is a conditional fit; it is not a fit to these cross-trace evaluation results.

Serial exchange assumptions: 6.33 decimal MB/expert, H2D=28.8 GB/s, D2H=28.8 GB/s
(unmeasured), two permutations at 600 GB/s (unmeasured), launch allowance 0.08 ms/exchange,
and total selection/fence/commit overhead 10 ms/sweep (unmeasured). Thus each exchange is
0.561783 ms plus amortized sweep cost. No ideal overlap is credited. GPU decay-kernel
and optional prefill accounting overhead are not measured and must be added after the
operator probe; neither the model nor the simulator can certify their cost.

The fit's exposed CPU coefficient is especially fragile under changed context, batches,
new routing after device moves and DDR contention. Do not interpret predicted tok/s as
measured gains. The R869 observed 57.5 tok/s static gain remains the serving evidence.

## Shift and selection protocol

Shift replays concatenate the full decode streams in both orders. Initialize from a prior
fitted on the first stream; freeze that prior across the shift. Preserve online counts,
placement and sweep phase, replacing only the prompt signal at observed boundaries.
The prefix-fitted prior deliberately makes static a strong first-content reference.

Recovery uses the terminal 512-call CPU share as the finite-trace steady-state proxy.
Report the earliest 128-call rolling mean within +/-5% of that proxy for 128 successive
windows (time is the end of confirmation); null means not observed. Also report a one-sided
no-more-than-5%-above proxy so a better-than-terminal share is not treated as a failure.
Multiple prompt changes in the suffix mean this is a content-stream proxy, not an actual
stationary synthetic-word-list experiment. R870 routing is absent; the operator must
collect it to settle recovery on that particular failure case.

The recommended knobs maximize equal-weight mean predicted tok/s across the two complete
cross-trace evaluations. That is offline tuning on both traces, not an untouched third
holdout. Direction-specific winners expose tuning asymmetry. Sensitivity reprices all grid
rows at sweep={0.1,10,50} ms and D2H={10,20,28.8} GB/s, both with the locked recommendation
and with reselection. Sensitivity reuses the same route scores without a new routing replay.

## Completed results and recommendation

All 1,458 policies / 2,916 cross-trace evaluations completed. The score winner is
`k=32 H=512 wp=32 wb=256 M=32 rho=2`, static-initialized. It does **not** beat static plus
a modest histogram budget under the inherited calibration. Default-off score support is
provided for operator experiments; profile initialization is the more useful first change.

| Policy | R869 prior -> R860b: CPU share / swaps per call / predicted tok/s | R860b prior -> R869: same |
|---|---|---|
| Static | 0.2566 / 0.000 / 54.11 | 0.2387 / 0.000 / 55.01 |
| R2 B-safe identity | 0.2014 / 0.485 / 55.64 | 0.1832 / 0.447 / 56.68 |
| R2 B-fast identity | 0.1295 / 1.387 / 57.92 | 0.1326 / 1.441 / 57.63 |
| R2 B-safe static-init | 0.1521 / 0.371 / 58.57 | 0.1520 / 0.387 / 58.55 |
| R2 B-fast static-init | 0.1200 / 1.178 / 58.87 | 0.1287 / 1.323 / 58.07 |
| Static + small histogram (M=8) | 0.2021 / 0.250 / 55.52 | 0.2010 / 0.249 / 55.59 |
| Static + small histogram (M=16) | 0.1587 / 0.452 / 57.51 | 0.1653 / 0.456 / 57.14 |
| Static + small histogram (M=32) | 0.1388 / 0.611 / 58.35 | 0.1424 / 0.662 / 58.04 |
| Best score | 0.1380 / 0.765 / 58.10 | 0.1439 / 0.770 / 57.76 |

Best score predicts +7.38% / +4.99% versus static, at about
0.77 exchanges per call (9.7 MB bidirectional exchange traffic/call). The M=32 histogram
comparator beats it in both directions; B-safe static-init reaches 58.57/58.55 tok/s at
only 0.37/0.39 swaps/call. These are predictions, not R872 numbers.

The grid includes the warmup prefix (first 98 calls) and multirow decode records with
first-row scoring. Profile fitting uses every row. Routing feedback, speculative accepted
tokens, quantization-shape pair rejection and measured per-call GPU decay overhead are absent.
The reconstructed R869 prior is not established as byte-identical to the measured serving
profile, so the 54.11 predicted static rate must not be relabelled the observed 57.5 tok/s.

| Shift / policy | Suffix CPU share | swaps/call | predicted tok/s | +/-5% recovery confirmation calls | one-sided calls |
|---|---:|---:|---:|---:|---:|
| r869-to-r860b / score | 0.1370 | 0.695 | 58.29 | not observed | 255 |
| r869-to-r860b / static | 0.2566 | 0.000 | 54.11 | 5118 | 255 |
| r869-to-r860b / B-safe | 0.1544 | 0.388 | 58.40 | 3425 | 255 |
| r869-to-r860b / B-fast | 0.1214 | 1.185 | 58.78 | not observed | 255 |
| r860b-to-r869 / score | 0.1447 | 0.763 | 57.72 | 3351 | 2366 |
| r860b-to-r869 / static | 0.2387 | 0.000 | 55.01 | 1414 | 255 |
| r860b-to-r869 / B-safe | 0.1552 | 0.396 | 58.34 | not observed | 3509 |
| r860b-to-r869 / B-fast | 0.1308 | 1.345 | 57.91 | 3380 | 961 |

The two-sided score recovery is censored R869 -> R860b and 3,351 calls in reverse.
In the first direction the initial suffix share is already below its terminal proxy; that
is why the one-sided result is 255 calls and the two-sided result is censored. Static
can enter the same terminal band merely because content changes, without adapting at all.
These diagnostics do not establish fast recovery of R870. A stationary misfit trace and
matching stationary tail are needed before selecting for that claim.

Sensitivity is in `sensitivity.json`. At 50 ms/sweep the locked score recommendation
falls below static on mean predicted throughput; cheaper sweeps favor shorter cadence. R872 should
replace these assumptions before any promotion. Recommended first exchange experiment:
`out/r3/B-initialized.env` (profile + unchanged r2 B-safe selection). Keep the measured
static serving setup while those exchange and lifecycle checks remain unmeasured.

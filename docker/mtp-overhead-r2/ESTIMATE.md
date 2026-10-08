# Expected costs, with uncertainty

No new GPU timings were collected. These are arm-budget estimates conditioned on R892's **dynamic-placement** N104 MTP-fast baseline. The exactness experiment uses static placement; first measure its own flags-off reference, then apply only the proposed **deltas**, not these absolute costs.

For c1 depth one, throughput is `1000 * E / T`, with measured `E≈1.74` tokens/step. At T27.4 this is 63.5 tok/s. Shape/placement/content can change both E and T. Report R828 counters and content-specific rates; do not substitute client request wall time for server decode step time.

| Flag set (MTP_FAST=1 retained) | Expected change in complete c1 step | Planning c1 ms/step / tok/s at E1.74 | c4 expectation |
|---|---|---|---|
| New flags off | Reference | 27.1–27.7 / measured 62.6–63.9 | Measured 89.8–91.5 aggregate, depending on arm/workload |
| PHASE_PROF only | Instrumentation overhead; no saving | Could add .5–3+ ms; unknown until P versus A | Observer cost; do not score this arm as an optimization |
| DRAFT_PINNED_STAGING only | 0–.2 ms saved; possibly neutral | 27.2–27.4 / 63.5–64.0 on T27.4 anchor | Likely neutral; .0–.4 ms/round unmeasured |
| MTP_GPU_DRAFT only | 0–.4 ms saved if eligible; possible .1 ms regression | 27.0–27.5 / 63.3–64.4 | c4 declines c1-only path; zero direct saving |
| MTP_GPU_EMBED only | 0–.2 ms saved if mirror fits; possible .1 ms regression | 27.2–27.5 / 63.3–64.0 | 0–.4 ms/round conditional; single mirror device can limit use |
| MTP_GREEDY_ACCEPT only | 0–.2 ms saved; neutral is plausible | 27.2–27.4 / 63.5–64.0 | 0–.4 ms/round, roughly <1%; second readback tails, not worker/GPU verify time |
| MTP_CACHED_REWIND only | 0–.2 ms saved; validation could regress .1 | 27.2–27.5 / 63.3–64.0 | 0–.4 ms/round, <1%; same batched GPU rewinds as base |
| All c1 optimization flags, profiler off, mirrors decline | Shared-budget net saving 0–.4 ms; possible .2 regression | 27.0–27.6 / 63.0–64.4 | Likely within noise, about 90–92 aggregate on R892 anchor |
| All c1 flags, profiler off, mirror active | Shared-budget net saving 0–.8 ms; possible .2 regression | 26.6–27.6 / 63.0–65.4 | Without batch cap still about 90–92; unmeasured |
| MTP_MAX_BATCH=1, sustained c1 | Cap inactive; same as selected c1 flags | Reference or combined row above | At c2/c4: ordinary target decode, no learned draft or repair in steady state |
| MTP_MAX_BATCH=1, sustained c4 at N104 | Different mode: E=1, plain (4,1), MTP cache dormant | Not a c1 speed forecast after a job has been suspended | Unknown matched N104 plain anchor. Planning T36–42 ms/round → 95–111 aggregate; N96 daily's measured ~110 is only an external anchor |
| Repair merge or new whole-forward graphs | No production optimization shipped | 0 delivered saving | 0 delivered saving; exact diagnostic probes supplied |

Ranges are deliberately modest: R892 already found pinned+batch verification neutral, and the kernels doing verification/CPU experts are unchanged. Independent flag maxima must not be summed: several remove the same host/readback dependency. Full mirrors may not fit at N104 and do not automatically change the critical path.

A complete MTP step at **24.3** ms with E1.74 would yield 71.6 tok/s, but there is no evidence these changes can deliver it; the 24.3 number is a different-shape plain batch. A claimed ~10 ms reduction to ~17.4 ms would imply 100 tok/s at E1.74, require dramatically cheaper verification, and is unsupported by this source or the supplied logs. The first value of this overlay is the phase breakdown that can discriminate where a larger reduction could exist.

For cap c4, compare to plain N104 with the same MTP-fast image, static placement file, cache/vision and prompt mixture. MTP memory remains allocated in the cap arm, so a model-only MTP-off arm can differ in placement and workspace pressure; record actual layer devices, expert counts and warm graph configurations. Do not present the 95–111 planning range as a measured recovery to N96's 110.

# R969: the text-owned expert-exchange clock with single-stream requests first on fresh boots: single-stream median of ten 64.8 against 63.8 tok/s (+1.6 %; c1 score −1.6 %), sums at 2, 3 and 4 streams +2.4, +6.3 and +5.7 %; kept

Results directory on the box: `/srv/qwen5090/results/2026-10-10-r969-glm53-textclock-170158/`. Driver: `scripts/r969-glm53-textclock-c1.sh` with `scripts/r969_summary.py` and `scripts/glm_arms.sh` (the published driver differs from the one that ran only in its header comments). The box's copies of `glm_arms.sh` and `glm53_probe.py` also carry a power sampler that is not in this repository; R969 ran with it off (`FLAN_POWER=0`). Raw records: `results/2026-10-10-r969-glm53-textclock/` (`<arm>/c1.jsonl` to `c4.jsonl`, the probe's records per phase; `vram-boot.csv` after boot; `effective-env.json` derived from the container's environment and command; `Bdebug/sweeps-c1.txt` and `sweeps-c2.txt`, the last `[SWAP-OWNER]` line of the debug boot and its `exchange sweep` lines during each phase; `summary.txt`, `gate.json`, `phases.jsonl`, the order the phases completed in; `daily.env`, the served configuration). Engine and boot logs are not published. Overlay: `docker/textclock-r1/`.

## Question and rule

R968 put the text clock into service (`EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1`, `r968-glm53-textclock.md`) with a single-stream median 1.2 to 1.9 % below the image without it. R968 ran its single-stream requests after a cold 4-stream round, so the text-clock arms entered them with a placement already moved toward 4-stream traffic (16 sweeps in one 4-stream round on its debug boot), and the arms without it with the boot placement. R969 runs the single-stream requests first on every boot.

Arms, all on the served image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1_textclock1` (image ID `sha256:f761883c72d4…` in every boot) and `scripts/glm-daily.env` as served: A with `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=0`, B with `=1` (served). With 0 the overlay's clock owner is registry entry 0, the clock before the overlay; the review before the run checked this in the patched source (`_text_clock_enabled` is true only for `1`). The driver strips the clock and debug entries from the served `EXL3_EXTRA` and appends `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=<0|1>;EXL3_MOE_CPU_SWAP_DEBUG=`; the summary requires the effective environment and command of all seven boots to be identical apart from those two names (`effective-env.json`).

Rule, registered in the brief before the run: per boot the median of its ten single-stream requests; per side the median of its three boots; KEEP if B/A is at least 0.985; REVERT candidate if B/A is below 0.985 and every B boot is below every A boot; INCONCLUSIVE otherwise. The 2-, 3- and 4-stream ratios are information only. The unit does not promote or roll back.

## Runs

One session on 2026-10-10, every arm a fresh boot; between the launcher's boot warmup (short requests at 1, 2 and 4 streams, the same for every arm) and the first measured request nothing ran.

- Bdebug (container started 17:02:37 UTC, discarded): image B with `EXL3_MOE_CPU_SWAP_DEBUG=1`. Last `[SWAP-OWNER]` line: `owner=model.language_model.layers.3.mlp component=text modules=43 text_clock=1`. During the ten single-stream requests 112 sweeps with 2,292 swaps (4,594 ms of sweep time in total); during the two 2-stream rounds 32 sweeps with 261 swaps (996 ms). For comparison, R967 counted 113 sweeps during ten single-stream requests with the clock before the overlay (`2026-10-10-r967-glm53-swap-clock-114438`, no raw records published) and 0 during 4-stream rounds.
- A1 (17:08 UTC), B1 (17:17), B2 (17:25), A2 (17:34), B3 (17:42), A3 (17:51 to about 17:59). Per arm: single stream (`--phase c1 --runs 2 --kinds code,prose,chat,html,edit`: the c1-score method of `bench/RESULTS.md`, 1,024 forced tokens, html 2,048, temperature 0, `reasoning_effort: low`), then `--phase decode --distinct --runs 2` at 2, 3 and 4 streams (one prompt per stream in the order code, prose, chat, html; 1,024 forced tokens; temperature 0; no token is drafted above one stream). The served configuration was restored after A3.

Every request ended with `finish_reason: length`. Free VRAM after boot 563 / 297 MiB in A1, B1, A2 and B3, 559 / 293 in B2 and Bdebug, 571 / 303 in A3. The results review found no other GPU work on the box between 17:02 and 18:01 UTC: one queued unit ran a CPU-only container check at 17:04 during Bdebug's boot, and a scheduled image prune reclaimed 0 bytes.

## Speed

Rates in tok/s. Single stream: the median of the ten requests (the rule's figure) and the c1 score (mean of the five per-kind medians of 2 runs). 2 to 4 streams: sum of the stream rates over the common decode window, per round, and the median of the 2 rounds.

| arm | clock | single stream, median of 10 | c1 score | 2 streams, sum (per stream) | 3 streams, sum (per stream) | 4 streams, sum (per stream) |
|---|---|---|---|---|---|---|
| A1 | before the overlay | 63.54 | 64.09 | 72.68 (36.34) | 85.52 (28.51) | 92.09 (23.02) |
| B1 | text | 63.69 | 62.18 | 74.41 (37.20) | 91.99 (30.66) | 98.83 (24.71) |
| B2 | text | 64.77 | 62.94 | 74.33 (37.16) | 90.33 (30.11) | 95.57 (23.89) |
| A2 | before the overlay | 64.24 | 63.96 | 74.54 (37.27) | 89.01 (29.67) | 92.76 (23.19) |
| B3 | text | 65.57 | 63.16 | 75.47 (37.74) | 92.49 (30.83) | 98.02 (24.51) |
| A3 | before the overlay | 63.76 | 62.89 | 71.71 (35.86) | 86.58 (28.86) | 93.49 (23.37) |
| A, median of 3 | | 63.76 | 63.96 | 72.68 (36.34) | 86.58 (28.86) | 92.76 (23.19) |
| B, median of 3 | | 64.77 | 62.94 | 74.41 (37.20) | 91.99 (30.66) | 98.02 (24.51) |
| B / A | | 1.016 | 0.984 | 1.024 | 1.063 | 1.057 |

Single stream by kind, median of 2 runs, code / prose / chat / html / edit: A1 62.5 / 66.0 / 71.6 / 66.7 / 53.7; B1 59.6 / 65.5 / 67.8 / 64.0 / 53.9; B2 60.7 / 67.0 / 68.5 / 64.8 / 53.7; A2 61.0 / 66.0 / 68.4 / 63.3 / 61.2; B3 59.9 / 66.7 / 68.6 / 66.3 / 54.3; A3 60.2 / 67.4 / 68.9 / 63.7 / 54.3. Median of the three boots, B: 59.9 / 66.7 / 68.5 / 64.8 / 53.9. Time to the first token, median per boot: single stream 0.47 to 0.49 s in every arm; per round at 2 streams 0.75 to 0.82 s, at 3 streams 1.11 to 1.23 s, at 4 streams 1.57 to 1.72 s in every arm.

Gate (`summary.txt`, `gate.json`): single stream B/A 1.0159, every B below every A false; DECISION: KEEP.

## Reading

- Single stream, by statistic: +1.6 % on the rule's median of ten; −1.6 % on the c1 score; −1.5 % on the geometric mean of the per-kind B/A ratios of the six-run means, −0.6 % with A2's two edit requests left out. A2's edit requests accepted 0.65 and 0.74 of their drafts against 0.93 to 0.97 for every other edit request and ran at 59.2 and 63.2 tok/s against 52.8 to 55.2: a different generation, which raised A's edit rate. R968's A2 had one such edit request (acceptance 0.64, 61.0 tok/s). Requests are greedy, yet every content hash differs between boots and between the two runs of one boot. The per-boot spread on this stack is about 2.7 % (R961b); the three B boots span 3.0 % on the median of ten. The statistics do not agree on the sign; every one of them is at or above 0.984.
- Code is the one kind lower in both rounds: −1.85 % here on the six-run means, −2.1 % in R968. Without A1, the first measured boot, the code and chat gaps are −0.8 and −0.5 %. Three boots per side do not resolve a difference of this size.
- The text clock ticks at one stream as the clock before it did: 112 sweeps during the ten single-stream requests on Bdebug against R967's 113. With the single-stream requests first, both arms enter them from the boot placement, and the rule's median moves from R968's −1.9 % to +1.6 %. R968's single-stream result is read as an order effect of the 4-stream round before it, together with the recurring A2 edit divergence; R969 does not measure that order directly.
- 3 and 4 streams: every B boot is above every A boot (3 streams 90.3 to 92.5 against 85.5 to 89.0; 4 streams 95.6 to 98.8 against 92.1 to 93.5). 2 streams: +2.4 % on the medians, not separated (A2 74.54 above B2 74.33). Bdebug logged 32 sweeps during the 2-stream rounds; R967 logged 0 at 4 streams with the clock before the overlay.
- A's 4-stream median, 92.76, is within 0.6 % of R968's 92.2 for the image without the overlay after its single-stream requests, consistent with `=0` reproducing the clock before the overlay.
- Kept: the served configuration is unchanged (`daily.env`). The independent review of the results confirmed KEEP and asked for the single-stream statistics above to be stated together. For future single-stream gates it recommends a per-kind statistic with diverged generations flagged by their draft acceptance.

# R968: the expert-exchange clock owned by the first text MoE layer (`EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1`): 4-stream sum on a fresh boot 84.9 against 76.5 tok/s (+10.9 %), after ten single-stream requests 96.5 against 92.2 (+4.6 %), single-stream median 64.4 against 65.6 tok/s (−1.9 %); served

Results directory on the box: `/srv/qwen5090/results/2026-10-10-r968-glm53-textclock-131424/`. Driver: `scripts/r968-glm53-textclock.sh` with `scripts/textclock_summary.py` and `scripts/glm_arms.sh` (the published driver differs from the one that ran only in its header comments). The box's copies of `glm_arms.sh` and `glm53_probe.py` also carry a power sampler that is not in this repository; R968 ran with it off (`FLAN_POWER=0`). A first attempt (unit started 13:00 UTC) stopped at its container check before any measurement: the check read the `EXL3_*` flags from the container's `Config.Env` only, and the launcher passes them as `/usr/bin/env` arguments in `Config.Cmd` (`docs/GOTCHAS.md`); the check now merges both. Raw records: `results/2026-10-10-r968-glm53-textclock/` (`<arm>/c1.jsonl`, `c4-cold.jsonl`, `c4-after.jsonl`, `fp.jsonl`, `vram-boot.csv` after boot, `effective-env.json` derived from the container's environment and command; `Bdebug/c4-debug.jsonl` and `Bdebug/sweeps.txt`, the last `[SWAP-OWNER]` line and the 16 `exchange sweep` lines of the debug boot's engine log during its 4-stream round; `summary.txt`, `gate.json`, `daily.env`, the served configuration before the round). Engine and boot logs are not published. Overlay: `docker/textclock-r1/`.

## Question and rule

The exchange of `docker/cheapswap-r3/` moves the experts the traffic uses most onto the GPUs. Two earlier measurements on the served configuration showed that 4-stream traffic did not move them:

- R966b (2026-10-10, `2026-10-10-r966b-glm53-c1-priming-104516`, no raw records published): on the running daily, a 4-stream round with distinct prompts summed 76.8 tok/s; ten single-stream requests (five kinds, 2 runs) followed; the next two 4-stream measurements summed 93.96 and 93.40.
- R967 (2026-10-10, `2026-10-10-r967-glm53-swap-clock-114438`, no raw records published): the served configuration with `EXL3_MOE_CPU_SWAP_DEBUG=1` on a fresh boot: 76.69 tok/s at 4 streams with 0 exchange sweeps, 113 sweeps during ten single-stream requests, 94.24 tok/s at 4 streams again with 0 sweeps. The first registered split module was `model.language_model.layers.45.mlp`, the MTP layer.

Mechanism (`docker/textclock-r1/ANALYSIS.md`, section 4.1): every split layer adds its routed picks to its own histogram at every batch size, but a sweep is decided every 64 invocations of registry entry 0, and with `EXL3_MTP_MAX_BATCH=1` the MTP layer does not run while two or more requests are active. `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1` gives the clock to the first text split module.

Rule, registered in the summary before the run: B's cold 4-stream sum at least 1.10 times A's; B's single-stream rate between 0.98 and 1.02 times A's; no `FATAL` line. Each side's value is the median of its two arms, each arm's value the median of its two 4-stream rounds or of its ten single-stream requests. Greedy fingerprints are reported as information only: the review before the run (`docker/textclock-r1/REVIEW-r2.md`) found that greedy output differs between boots of one image under the exchange. The unit does not promote.

## Runs

One session on 2026-10-10, every arm a fresh boot of `scripts/glm-daily.env` as served before the round (image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1`, `EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,102`, `CHUNK=4096`). A is that image; B is `..._draftchunk1_textclock1` built from it with `--network=none`, with `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1` and `EXL3_MOE_CPU_SWAP_DEBUG` empty appended to `EXL3_EXTRA`. The two differ in nothing else (`effective-env.json`; A1 and A2 identical, B1 and B2 identical). Free VRAM after boot 563 / 297 MiB in A1, B2 and A2, 561 / 295 MiB in B1.

- Bdebug (13:15 to 13:17 UTC, discarded): image B with `EXL3_MOE_CPU_SWAP_DEBUG=1`, one 4-stream round with distinct prompts after boot. Last `[SWAP-OWNER]` line: `owner=model.language_model.layers.3.mlp component=text modules=43`. 16 sweeps with 326 swaps during the round (0 to 64 swaps per sweep, 24 to 68 ms each, 648 ms in total), summed rate 81.6 tok/s.
- A1 (13:17 to 13:27), B1 (to 13:37), B2 (to 13:46), A2 (to 13:56 UTC). Per arm, after the launcher's boot warmup and nothing else: cold 4-stream (`--phase decode --concurrency 4 --runs 2 --distinct`: streams with the code, prose, chat and html prompts, 1,024 forced tokens, temperature 0, the template's default reasoning effort), single stream (`--phase c1 --runs 2 --kinds code,prose,chat,html,edit`: the c1-score method, 1,024 forced tokens, html 2,048, temperature 0, `reasoning_effort: low`), the 4-stream rounds again, then the greedy fingerprint (five kinds, one run, fixed salt).

Every request ended with `finish_reason: length`. The results review counted 45 server-side requests in each arm's engine log, all of them probe requests in the same order, and found no other unit on the box in the window.

## Speed

Rates in tok/s. 4 streams: sum of the stream rates over the common decode window, per round, and the median of the 2 rounds. Single stream: the median of the ten requests (the rule's figure) and the c1 score (mean of the per-kind medians, `bench/RESULTS.md`).

| arm | image | 4 streams, fresh boot (rounds) | single stream, median of 10 | c1 score | 4 streams after the single-stream requests (rounds) |
|---|---|---|---|---|---|
| A1 | without the overlay | 77.28 (76.13, 78.43) | 65.72 | 64.4 | 94.38 (94.11, 94.65) |
| B1 | text clock | 84.12 (80.24, 87.99) | 64.07 | 63.7 | 96.74 (96.16, 97.31) |
| B2 | text clock | 85.63 (80.50, 90.77) | 64.71 | 64.0 | 96.19 (94.90, 97.48) |
| A2 | without the overlay | 75.81 (75.41, 76.20) | 65.56 | 65.3 | 90.09 (91.14, 89.04) |
| A, median of A1 and A2 | | 76.54 | 65.64 | | 92.23 |
| B, median of B1 and B2 | | 84.87 | 64.39 | | 96.46 |
| B / A | | 1.109 | 0.981 | | 1.046 |

Single stream by kind, median of 2 runs, code / prose / chat / html / edit: A1 63.0 / 69.2 / 69.8 / 65.7 / 54.5; B1 62.6 / 66.8 / 70.4 / 64.1 / 54.4; B2 62.1 / 67.8 / 70.4 / 64.7 / 55.0; A2 64.3 / 68.7 / 71.4 / 65.4 / 56.8. Draft acceptance per request 0.55 to 0.97; A2's second edit request accepted 0.64 of its drafts against 0.92 to 0.97 for the other seven edit requests and ran at 61.0 tok/s. Time to the first token at 4 streams, median per round, 1.60 to 1.80 s in every arm.

Gate (`summary.txt`, `gate.json`): cold 4 streams +10.88 % PASS, single stream −1.90 % PASS, no `FATAL` line PASS; OVERALL PASS. Fingerprints: all five kinds differ between every pair of arms, A1 and A2 included.

## Reading

- On a fresh boot the 4-stream sum is 10.9 % higher with the text clock, and every B round is above every A round. The gap grows within the phase: about +6 % in the first round (80.2 and 80.5 against 76.1 and 75.4) and about +16 % in the second (88.0 and 90.8 against 78.4 and 76.2), as the sweeps move the placement toward the 4-stream traffic. Before the overlay this gain needed single-stream requests in between (R966b, R967).
- After the single-stream requests the 4-stream sum is 4.6 % higher on the medians and 2.2 % higher against A1 alone; A2 read 4.5 % below A1, so the gain at this point lies between about 2 and 5 %.
- Single stream: −1.9 % on the medians of the ten requests and −1.2 to −1.3 % on the geometric mean of the per-(kind, run) B/A ratios with A2's second edit request left out or replaced by A1's (−1.6 % with it). B is below A in 8 of the 10 (kind, run) cells. The per-arm drift on this stack is about 2.7 % (R961b); the two A boots differ by 0.2 % here, the two B boots by 1.0 %. Single-stream decode is the shape most real traffic has, so this change is stated with the numbers above and not taken as zero. Its cause is not measured; two candidates: B enters the single-stream requests with a placement adapted to the 4-stream rounds, or B's sweeps run inside layer 3's forward instead of at the MTP layer.
- The sweep path is the same code as before; the flag changes when it runs. Greedy fingerprints differ between the two A boots as much as between A and B, so they say nothing about the flag; output equality under the exchange is not tested by this round.
- Served from 2026-10-10 about 14:06 UTC: `scripts/glm-daily.env` with image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1_textclock1` and `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1`, the rest unchanged. The served container's last `[SWAP-OWNER]` line names `layers.3.mlp component=text modules=43` (operator check, no published record). The independent review of the results recommended serving it with the single-stream note. Next: fresh boots with the single-stream requests first and no 4-stream round before them, 2 and 3 streams, and one debug boot that counts the sweeps during single-stream requests.

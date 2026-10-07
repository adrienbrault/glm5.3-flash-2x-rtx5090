# R870: decode at 128k context runs at the short-context rate; the slow requests are the first ones after a boot

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r870-glm53-depth-063536/`. Driver: `scripts/r870-glm53-depth.sh`. Raw records: `results/2026-10-07-r870-glm53-depth/` (`ladder/depth.jsonl`, `prof32k/depth.jsonl`, `prof32k/prof-tail.txt`, configs, `summary.txt`).

## Configuration

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, 96 experts per layer on the CPU, dynamic placement, MTP off. One boot; for each depth one completion request with a calibrated random word-list prompt, cold prefill timed by the engine, then 256 forced decode tokens. A second boot repeats 32k with `PROFILE=1`.

| request (in order) | prompt tokens | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|
| 1 | 4,089 | 1,549 | 21.7 |
| 2 | 8,194 | 1,914 | 25.8 |
| 3 | 16,455 | 1,894 | 45.9 |
| 4 | 32,804 | 1,948 | 42.6 |
| 5 | 65,631 | 1,949 | 43.9 |
| 6 | 131,089 | 1,848 | 48.7 |
| profiled boot, first request | 32,685 | 1,917 | 21.4 |

- Decode does not fall with context: 128k decodes at the short-context rate. The slow points are the first requests after each boot. The word-list filler routes to other experts than the probe prompts, and dynamic placement adapts over the next generations.
- The profile agrees (`prof32k/prof-tail.txt`): the running mean of the gate/up GEMV per CPU job falls from 750 to 546 µs over the request, against 239 µs in R860b, i.e. more CPU picks per job than in steady decode. One 0.73 s stall mid-decode at 128k is consistent with an inline overdue sweep.
- Prefill is flat at 1,850 to 1,950 tok/s from 8k up.

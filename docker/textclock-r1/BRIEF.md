# GLM-5.3-Flash expert exchange: why does c4 traffic not re-place experts? (analysis task, real measurements)

Source: src/exllamav3 = the exact Python/CUDA source of the serving image (TabbyAPI + ExLlamaV3 with our overlays).
Serving config (env): EXL3_MOE_CPU_SWAP=1 EXL3_MOE_CPU_SWAP_MODE=exchange EXL3_MOE_CPU_SWAP_POLICY=histogram
EXL3_MOE_CPU_SWAP_CADENCE=exact EXL3_MOE_CPU_SWAP_INTERVAL=64 EXL3_MOE_CPU_SWAP_FLOOR=4 EXL3_MOE_CPU_SWAP_MAX=64
EXL3_MOE_CPU_SWAP_BUDGET_SCOPE=global EXL3_MOE_CPU_SWAP_HYST=2.0 EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,102
EXL3_MOE_CPU_SPLIT_STATS=/app/split-stats.json EXL3_MTP_MAX_BATCH=1 (MTP drafting only at batch 1), max batch 4..8.
GPU-resident experts per layer = the split; the tail runs on 8 CPU worker threads. Exchange swaps hot CPU experts with
cold GPU experts.

Measured 2026-10-10 on the live server (thinking-off, same prompts, 1024 output tokens per stream):
- c4 (4 concurrent streams, code/prose/chat/html, 2 rounds) "cold" after real user traffic: 76.2, 77.1, 76.8 tok/s
  aggregate, across three consecutive c4 runs (~6k batch-4 decode steps) -> placement barely moved.
- then 10 c1 requests (one stream at a time, kinds code/prose/chat/html/edit, 1024 tokens each, MTP on) -> c4 93.96
  and again 93.40. So ten batch-1 requests re-placed the experts and c4 gained 22 %; c4 traffic itself did not.

Questions (answer from the code, with file:line):
1. Where is the routing histogram accumulated, and under which conditions (batch size, MTP draft vs verify steps,
   prefill vs decode, CUDA-graph path vs eager, per-layer vs global)? Is any path skipped when batch > 1 or when
   MTP is off (batch > MTP_MAX_BATCH)?
2. What triggers an exchange decision (INTERVAL counted in steps, tokens, requests?), what does HYST/FLOOR/MAX do,
   and can those thresholds make batch-4 traffic (counts spread over 4 different content kinds) never cross them?
3. Is there decay / a window, and how long does placement persist?
4. The most likely explanation for the measurement above, ranked, and for each the smallest log line or counter that
   would confirm it on the live server (the operator can read `docker logs` and set EXL3_MOE_CPU_SWAP_DEBUG=1 on a
   test boot).
5. If c4 traffic is excluded or underweighted: a minimal patch design behind a new env flag (default off), what it
   changes, risks (exactness: exchange must stay bit-exact; latency of swaps at c4), and how to A/B it.
Deliverables: out/ANALYSIS.md (written as you go), out/last.txt LAST. No network, no ssh. If the code cannot decide a
question, say exactly which log line or measurement would.

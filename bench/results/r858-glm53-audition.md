# R858: GLM-5.3-Flash 2.05 bpw boots with 104 of 288 experts per MoE layer on the CPU; c1 decode 49.5 tok/s

Results directory on the box: `/srv/qwen5090/results/2026-10-06-r858-glm53-233400-2023872/` (first attempt `2026-10-06-r858-glm53-232616-2002321/`). Driver: `scripts/r858-glm53-audition.sh` with `PLAN_SPLIT=28,28 KEEP_GLM=1`. Raw records: `results/2026-10-06-r858-glm53/` (`measure.jsonl` one line per stream and per round, `sanity.jsonl`, `config.yml`, `vram-boot.csv`, `audit.txt`).

## Configuration

Checkpoint `turboderp/GLM-5.3-Flash-exl3` revision `2.05bpw` (2.05 bpw, head 5 bits, MTP 2 bits). Image `tabbyapi:r828-prompt-lookup-r3`. `cpu_moe_split_experts: 104`, `EXL3_MOE_CPU_THREADS=8`, `EXL3_MOE_PINNED_ARENA=0`, `gpu_split: [31, 31]`, cache 65,536 tokens at 8-bit K and V, `max_seq_len` 49,152, chunk 512, MTP draft depth 1, vision tower on the GPU. Stock power limits 600 / 575 W, memory clock offset +4,500 MHz, core offset 0.

## Fit ladder

The first run planned against 31 + 31 GiB. Attempt 1 (80 experts per layer on the CPU, MTP on) and attempt 2 (12 whole MoE layers on the CPU, MTP on) both stopped at module 45 of 50 with `Insufficient VRAM in split for model and cache`, both cards full. The second run planned against 28 + 28 GiB and booted its first attempt, 104 experts per layer on the CPU (estimated 33.0 GB of routed expert bytes in RAM).

## Sanity

Three prompts (17 × 23, the capital of France, repeat a word) at `reasoning_effort` low and high returned 391, Paris and RED in all six requests.

## Decode

One chat request per stream, forced to 1,024 tokens with `min_tokens`, greedy, `reasoning_effort: low`, prompt "Write a long detailed Python tutorial with complete code and tests. Keep elaborating." with a per-run nonce. Three rounds per concurrency after a 32-token warm-up of each batch shape.

| concurrency | per-stream rate by round (tok/s) | sum of stream rates, median (tok/s) | MTP acceptance |
|---|---|---|---|
| 1 | 44.6, 49.5, 50.5 | 49.5 | 0.90, 0.80, 0.90 |
| 2 | 30.0, 30.0, 29.8 | 60.1 | 0.85 to 0.90 |
| 4 | 17.1, 17.1, 17.1 | 68.6 | 0.87 |

The content of the streams differs between rounds: some streams answered with a refusal instead of a tutorial, which moves the draft acceptance. R859 separates content kinds.

## Prefill

Not measured: the first cold 8,192-token prefill failed with `KeyError: 'dictionary is empty'` in the recurrent checkpoint cache, configured at 0 MB (`docs/GOTCHAS.md`).

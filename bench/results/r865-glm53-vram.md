# R865: 96 experts per layer on the CPU reproduces 48.2 tok/s; 92 does not fit even with a 31.8 GiB split

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r865-glm53-vram-022625/`. Driver: `scripts/r865-glm53-vram.sh`. Raw records: `results/2026-10-07-r865-glm53-vram/` (`ctl/c1.jsonl`, `S/depth.jsonl`, `S/vision.jsonl`, configs, `summary.txt`).

## Configuration and results

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, dynamic placement, MTP off. c1 score as in `bench/RESULTS.md`, five kinds.

| arm | change | result |
|---|---|---|
| ctl | 96 on the CPU | c1 score 48.2 tok/s (code 47.3, prose 48.5, chat 47.5, html 48.2, edit 49.3) |
| s92 | 92 on the CPU, `gpu_split` 31, 31.8 GiB | no boot (`Insufficient VRAM`) |
| c92 | s92 plus chunk 1,024 and `max_batch_size` 2 | booted, then the launcher's warmup failed: it ran c4 against `max_batch_size` 2 and found no shared decode window |

The launcher now caps its warmup concurrency at `max_batch_size`. The serving arm (agent overlay r2, 96 on the CPU) passed the vision check; its 32,774-token depth request prefilled at 1,839 tok/s and decoded at 20.7 tok/s (first request after the boot, see R870).

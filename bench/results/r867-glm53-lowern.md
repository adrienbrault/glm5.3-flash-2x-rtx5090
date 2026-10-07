# R867: 92 experts per layer on the CPU is +2 % at half the batch and boots unreliably; decode after a 32k prompt reads 22.7 tok/s

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r867-glm53-lowern-030858/`. Driver: `scripts/r867-glm53-lowern.sh`. Raw records: `results/2026-10-07-r867-glm53-lowern/` (per arm `c1.jsonl`, `config.yml`, `resolved.json`; `pf-ref/depth.jsonl`; `S/vision.jsonl`; `summary.txt`).

## Configuration and results

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, dynamic placement. c1 score as in `bench/RESULTS.md`, five kinds, 2 runs each.

| arm | change | c1 score (tok/s) | per kind |
|---|---|---|---|
| n92 | 92 on the CPU, chunk 1,024, `max_batch_size` 2, split 31, 31.8 GiB | 49.5 | code 48.8, prose 49.3, chat 49.8, html 48.7, edit 50.9 |
| n88 | 88 on the CPU, same settings | no boot (`Insufficient VRAM`) | |
| d1n96 | MTP depth 1, 96 on the CPU, same settings | no boot (`Insufficient VRAM`) | |
| d1n100 | MTP depth 1, 100 on the CPU, same settings | 47.0 | code 49.8 (acc 0.89), prose 44.6 (0.62), chat 46.5 (0.67), html 42.1 (0.60), edit 52.1 (0.95) |
| ref | MTP off, 96 on the CPU, base settings | 48.4 | code 47.0, prose 48.0, chat 49.0, html 48.6, edit 49.4 |

- 92 on the CPU is 1 to 2 % above 96 and halves the batch. Its second boot (`pf-n92`) failed with the container exiting, and the configuration served at the end answered `White` for the blue vision square. Not used.
- MTP depth 1 at 100 on the CPU is 6 % above `ref` on code and edits and below it on prose and html.
- `pf-ref` (96 on the CPU, base settings): a 32,791-token prompt prefilled at 1,913 tok/s, and the 256 tokens after it decoded at 22.7 tok/s, half the short-context rate. R870 traced this to placement, not context length.

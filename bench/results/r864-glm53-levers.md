# R864: 96 experts per layer on the CPU is +4.8 % at c1; MTP depth 1 helps code and edits only; prompt lookup adds nothing

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r864-glm53-levers-013012/`. Driver: `scripts/r864-glm53-levers.sh`. Raw records: `results/2026-10-07-r864-glm53-levers/` (per arm `c1.jsonl`, `config.yml`, `resolved.json`, `vram-boot.csv`; `S/vision.jsonl`; `summary.txt`, with one line naming a private gateway removed).

## Configuration

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, dynamic placement, 8 CPU threads. c1 score as in `bench/RESULTS.md`; A, A2 and B-n96 ran five kinds (code, prose, chat, html, edit), D and E ran code and edit.

| arm | change | c1 score (tok/s) | per kind |
|---|---|---|---|
| A | MTP off, 104 on the CPU | 46.2 | code 43.8, prose 46.9, chat 46.7, html 46.1, edit 47.3 |
| B-n96 | MTP off, 96 on the CPU | 48.4 | code 47.2, prose 48.7, chat 48.1, html 48.3, edit 49.6 |
| C-n92, C-n88 | MTP off, 92 or 88 on the CPU | no boot (`Insufficient VRAM`) | |
| E | MTP depth 1, 104 on the CPU | 50.3 | code 50.1 (acc 0.85), edit 50.5 (acc 0.95) |
| D | E plus prompt lookup | 49.6 | code 51.0 (acc 0.86), edit 48.1 (acc 0.85) |
| A2 | A again | 46.8 | code 45.2, prose 46.8, chat 47.1, html 47.1, edit 47.8 |

- A and A2 differ by 1.3 %: the boot-to-boot drift of this configuration.
- Moving 8 experts per layer from the CPU to the GPU (104 to 96) raised every kind; card memory at 96: 31,286 and 30,502 MiB in use.
- MTP depth 1 gains 7 to 10 % on code and edits (acceptance 0.85 to 0.95) and nothing on prose or chat (R860).
- The serving arm S booted 96 on the CPU with the agent overlay r2 (`docker/glm-agent-r2/`): vision check passed; a live streamed `write_file` probe of 1,000 lines stopped at 67 argument characters with `finish_reason: stop` (R868 follows it up).

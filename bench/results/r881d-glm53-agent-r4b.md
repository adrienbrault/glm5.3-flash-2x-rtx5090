# R881d: agent overlay r4b does not boot: every chat request raises NameError

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r881d-glm53-agent-r4b-220355/`. Driver: `scripts/r881d-glm53-agent-r4b.sh` (R881 and R881b are earlier attempts of the same comparison: R881 ran the overlay author's own launch, which did not fit in VRAM, and R881b was superseded before it held the GPUs). Raw records: `results/2026-10-07-r881d-glm53-agent-r4b/` (configs, `summary.txt`).

Arm A4B: `tabbyapi:r861-glm-agent-r4b` (`docker/glm-agent-r4b/`), `AGENT=1`, `TAG_TRACE=1`, base configuration, static placement, 96 experts per layer on the CPU, MTP off. The boot failed (`A4B NO BOOT`): r4b's `_encode_prompt` reads `TABBY_GLM_CACHE_VERIFY` through `os.getenv`, and `backends/exllamav3/model.py` does not import `os`, so the launcher's warmup requests failed. The round restored the served configuration (R882c). Revision 4c (`docker/glm-agent-r4c/`) adds the import; R881e measures it.

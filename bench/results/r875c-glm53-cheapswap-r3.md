# R875b and R875c: cheapswap r3 passes its gates; exchange swaps started from the broad hot set are +6.9 % at c1 over static placement

Results directories on the box: `/srv/qwen5090/results/2026-10-07-r875b-glm53-cheapswap-r3-113928/` (gates) and `/srv/qwen5090/results/2026-10-07-r875c-glm53-cheapswap-r3-182934/` (c1 arms). Drivers: `scripts/r875b-glm53-cheapswap-r3.sh`, `scripts/r875c-glm53-cheapswap-r3.sh` (`scripts/r875-glm53-cheapswap-r3.sh` is the first attempt, whose gate hit a transient `CUDA-capable device(s) is/are busy or unavailable` at load). Raw records: `results/2026-10-07-r875b-glm53-cheapswap-r3/` (`T/*.log`, `summary.txt`) and `results/2026-10-07-r875c-glm53-cheapswap-r3/` (per arm `c1.jsonl`, configs, `summary.txt`).

## Gates (R875b)

Image `tabbyapi:cheapswap-r3` (`docker/cheapswap-r3/`). GPU self-test, 96 swaps, on cuda:0 and cuda:1: pass, maximum absolute error 2.3e-4. Worker ring on both cards: pass. Profile initialisation, four combinations of swizzle and fold: pass, 42 initialised layers with all GPU, worker and auxiliary expert bytes matching. The c1 arm of R875b was stopped by the operator after the gates.

## c1 (R875c)

Base configuration, 96 experts per layer on the CPU, MTP off; c1 score as in `bench/RESULTS.md`, five kinds, 2 runs each, one boot per arm in one session. The exchange arms start from the counts in `scripts/split-stats-broad-r869.json`.

| arm | placement | c1 score (tok/s) | code | prose | chat | html | edit |
|---|---|---|---|---|---|---|---|
| SB | static, broad counts | 56.7 | 56.7 | 62.6 | 61.7 | 52.7 | 49.9 |
| XI | exchange, histogram policy, exact cadence every 64 tokens, global budget 64, floor 4, hysteresis 2.0 | 60.6 | 58.3 | 64.3 | 65.4 | 63.8 | 51.5 |
| XIF | exchange, served cadence every 16 tokens, per-layer budget 64, floor 2, hysteresis 1.2 | 58.0 | 53.2 | 61.3 | 61.8 | 60.1 | 53.6 |

XI is 6.9 % above SB, close to R872's XF from the identity start (60.1): the profile start adds little on top of the adaptation. The gain is mostly html (+21 %) and chat (+6 %). No poisoned registry or traceback in either exchange arm. XI is the placement served since R882c.

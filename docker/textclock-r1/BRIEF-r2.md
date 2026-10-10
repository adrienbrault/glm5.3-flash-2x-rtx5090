# Round 2: implement EXL3_MOE_CPU_SWAP_TEXT_CLOCK on the daily image + a GPU ladder unit (implementation task)

Your ANALYSIS.md §4.1 is confirmed on the live server (R967, 2026-10-10): daily config + EXL3_MOE_CPU_SWAP_DEBUG=1 ->
fresh boot c4 76.69 tok/s with 0 exchange sweeps; 10 c1 requests -> 113 sweeps; c4 again 94.24 with 0 sweeps. First
registered split module: model.language_model.layers.45.mlp (MTP). So implement your §5 design exactly:
`EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1` (default 0): `swap_clock_owner(reg)` = first `cpu_component == "text"` module else
reg[0]; used by `_split_swap_tick()` gating AND `moe_exchange.run_sweep()` reset (same owner); validate mode=exchange,
policy=histogram, cadence=exact when enabled (reject otherwise, loudly); startup `[SWAP-OWNER]` line; DEBUG
`[SWAP-CLOCK]` counters. Flag off = byte-identical behaviour.

Deliverables:
1. out/textclock.patch against src/exllamav3 (the daily image's exact source) + out/Dockerfile.textclock (FROM the daily
   image `ARG BASE`, copies the patched Python files only, like ref/Dockerfile.draftchunk) + an import-time assertion that
   the patch landed (e.g. a module attribute checked in the Dockerfile RUN step).
2. CPU tests (no GPU here): registry with MTP-first and text-first orders, flag on/off, the reset-the-same-owner bug
   (counter must not re-fire every call after 64), validation rejects; mutation tests that catch each bug.
   -> out/CPU-TESTS.txt
3. out/r968-glm53-textclock.sh: a GPU unit in the style of ref/r961b-glm53-livemetrics.sh + ref/glm_arms.sh (gpu_lock,
   gateway_drain, build overlay FROM the daily image with --network=none, boot arms with the daily env, FLAN_POWER=0,
   restore the daily at the end). Arms, each a FRESH boot, order A B B A: A = daily image as is; B = overlay +
   TEXT_CLOCK=1 + SWAP_DEBUG=1? (no: DEBUG off in measured arms; one extra short B boot with DEBUG=1 first to log
   [SWAP-OWNER] and count sweeps during c4). Per arm: c4 cold (decode --concurrency 4 --runs 2 --distinct) FIRST, then
   c1 (--phase c1 --runs 2 --kinds code,prose,chat,html,edit), then c4 again, then a greedy fingerprint (glm_arms fp) to
   prove identical outputs between A and B. Report per arm: c4 cold, c1 median, c4 after, fingerprint hashes. Gate in
   the summary: B c4 cold >= A c4 cold + 10 %, B c1 within 2 % of A, fingerprints identical, no FATAL lines.
   Time budget: say the expected wall time (one boot ~2.3 min).
4. out/CHANGES.md, out/last-r2.txt LAST. No network, ssh or commits. If something in the source contradicts the design,
   stop and say exactly what.

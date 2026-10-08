#!/usr/bin/env bash
# R925 (2026-10-08): GPU0 still had 847 MiB free at the promoted 100/104 split (user: "is that wasted vram?"). 98/104 fits
# (587 MiB free, gate 450) but its one R915c arm read lower with low MTP acceptance (1.52 tok/step), likely noise. Paired
# ABAB in one slot: A=100/104 (daily) B=98/104, A, B; each c1 five kinds x2 + server step, distinct c1/c2/c4, vision.
# Promote 98 if B beats A on the c4 mean and c1 mean is not lower by more than 1 %. Built from r915c's helpers.
#   sudo systemd-run --unit=r925-glm53-splitdev-abab --property=RuntimeMaxSec=14400 bash /srv/qwen5090/r925-glm53-splitdev-abab.sh
# R915b helpers and current glm-daily.env are the authority for boot and restoration.
# Build this context before launching the GPU unit; see OPERATOR.md.
set -uo pipefail
set -a
UNIT_START=$(date +%s)
UNIT=r925
PK=/srv/qwen5090/r925
HERE=/srv/qwen5090/r925-tools
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r925-glm53-splitdev-abab-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
GPU_QUEUE_NAME=r925-glm53-splitdev-abab
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
sudo -n docker image inspect "$IMG" >/dev/null || exit 2
gpu_lock || exit 2
SLOT_START=$(date +%s)
DEADLINE=$((SLOT_START + 3300))  # operator: from the lock, not the queue start (R915c queues behind R916); reserve restoration time.
read -r -a DAILY_ARGS <<< "$DENV"
DX=
for arg in "${DAILY_ARGS[@]}"; do
    [[ "$arg" == EXL3_EXTRA=* ]] && DX=${arg#EXL3_EXTRA=}
done
# the daily now carries EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104 itself; drop it so each arm sets exactly one value
# (run 1: B1-n98 booted at 98 but glm53_verify read the daily's 100 -> NO BOOT)
DX=$(tr ';' '\n' <<< "$DX" | grep -v '^EXL3_MOE_CPU_SPLIT_BY_DEVICE=' | paste -sd';' -)
note "GPU lock held; results $R; daily $DENV"
# A fresh bash inherits exported helper definitions/variables, but not the parent's
# cleanup trap. timeout bounds boot/c1spd functions as well as CLI probes.
bounded(){ local cap=$1; shift; local left=$((DEADLINE - $(date +%s)));
    ((left > 0)) || { note "slot test deadline reached"; return 124; }
    ((cap <= left)) || cap=$left
    # arrays do not export: re-declare glm_arms' CLEAN in the child (first run: boot ran `timeout HOME=...`, NO BOOT)
    timeout -k 10 "$cap" bash -c "$(declare -p CLEAN); "'"$@"' splitdev-phase "$@"
}
engine_log(){ sudo -n docker logs glm53 > "$R/$1/engine.log" 2>&1 || true; }
headroom(){
    local tag=$1 stage=${2:-warmup}; local D=$R/$tag; local g0 g1
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$D/vram-$stage.csv"
    cp "$D/vram-$stage.csv" "$D/vram.csv"
    read -r g0 g1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
    note "$tag free MiB: GPU0=$g0 GPU1=$g1 (gate 450/300)"
    [[ "$g0" =~ ^[0-9]+$ && "$g1" =~ ^[0-9]+$ ]] && ((g0 >= 450 && g1 >= 300))
}
updyn(){
    local tag=$1 n0=$2 check=$3; local D=$R/$tag
    mkdir -p "$D"
    sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
    note "$tag boot N=$n0,104 CHECK=$check GPU_SPLIT=31.8,31"
    bounded 180 boot "$D/boot" "$HERE" "${DAILY_ARGS[@]}" GLM_IMG="$IMG" GPU_SPLIT=31.8,31 \
        "EXL3_EXTRA=$DX;EXL3_MOE_CPU_SPLIT_BY_DEVICE=$n0,104;EXL3_MOE_CPU_SPLIT_CHECK=$check" \
        || { engine_log "$tag"; note "$tag NO BOOT"; return 1; }
    engine_log "$tag"
    bounded 90 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1 \
        || { engine_log "$tag"; note "$tag WARMUP FAILED"; return 1; }
    engine_log "$tag"
    headroom "$tag" || { note "$tag REJECT headroom"; return 1; }
    if grep -Eq 'SPLIT-CHECK.*tensor=|scatter gather kernel index out of bounds|device-side assert|previous expert exchange failed' "$D/engine.log"; then
        note "$tag REJECT engine failure"; return 1
    fi
}
measure(){
    local tag=$1; local D=$R/$tag
    # R915b's c1spd is the existing five-kind (code/prose/chat/html/edit) x2 probe.
    bounded 300 c1spd "$tag" || { engine_log "$tag"; note "$tag c1 FAILED/timeout"; return 1; }
    bounded 240 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/dec.jsonl" --phase decode --concurrency 1,2,4 --runs 2 --distinct \
        > "$D/dec.log" 2>&1 || { engine_log "$tag"; note "$tag distinct FAILED/timeout"; return 1; }
    bounded 90 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/vision.jsonl" --phase vision > "$D/vision.log" 2>&1 \
        || { engine_log "$tag"; note "$tag vision FAILED/timeout"; return 1; }
    engine_log "$tag"
    python3 "$HERE/mtp_steps.py" "$D/engine.log" > "$D/server-steps.txt" 2>&1 || true
    headroom "$tag" matrix || return 1
    if grep -Eq 'scatter gather kernel index out of bounds|device-side assert|previous expert exchange failed' "$D/engine.log"; then
        note "$tag REJECT engine failure"; return 1
    fi
    note "$tag full matrix completed"
}
# Export all sourced helper functions so bounded subprocesses can call boot/c1spd.
# No extra EXIT trap: arms_init owns restoration and re-reads glm-daily.env on exit.
export -f $(compgen -A function)
for arm in "A1 100" "B1 98" "A2 100" "B2 98"; do
    set -- $arm
    if (( $(date +%s) >= DEADLINE - 120 )); then note "$1-n$2 UNTESTED: deadline"; continue; fi
    if updyn "$1-n$2" "$2" 0; then measure "$1-n$2" || note "$1-n$2 FAILED/INCOMPLETE"; else note "$1-n$2 NO BOOT/WARMUP"; fi
    D=$R/$1-n$2
    note "$1-n$2 decode distinct: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$D/dec.jsonl" 2>/dev/null | tail -1) | step $(tail -1 "$D/server-steps.txt" 2>/dev/null | cut -c1-120)"
done
note "testing elapsed $(( $(date +%s) - SLOT_START )) seconds; returning to current daily"
exit 0

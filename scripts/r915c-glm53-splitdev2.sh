#!/usr/bin/env bash
# R915b helpers and current glm-daily.env are the authority for boot and restoration.
# Build this context before launching the GPU unit; see OPERATOR.md.
set -uo pipefail
set -a
UNIT_START=$(date +%s)
UNIT=r915c
PK=/srv/qwen5090/r915b
HERE=/srv/qwen5090/r915b-tools
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r915c-splitdev2-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
GPU_QUEUE_NAME=r915c-glm53-splitdev2
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
sudo -n docker image inspect "$IMG" >/dev/null || exit 2
gpu_lock || exit 2
SLOT_START=$(date +%s)
DEADLINE=$((SLOT_START + 1500))  # operator: from the lock, not the queue start (R915c queues behind R916); reserve restoration time.
read -r -a DAILY_ARGS <<< "$DENV"
DX=
for arg in "${DAILY_ARGS[@]}"; do
    [[ "$arg" == EXL3_EXTRA=* ]] && DX=${arg#EXL3_EXTRA=}
done
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
updyn C-n98 98 1 || exit 3
updyn B-n98 98 0 && measure B-n98 || exit 4
# Alternatives are gated on the complete 98 matrix, never on load alone.
# Try both while the slot permits; explicitly report timeouts as untested.
for n0 in 96 100; do
    if (( $(date +%s) >= DEADLINE - 120 )); then
        note "B-n$n0 UNTESTED: reserve restoration window"; continue
    fi
    if updyn "B-n$n0" "$n0" 0; then
        measure "B-n$n0" || note "B-n$n0 FAILED/INCOMPLETE (do not promote)"
    else
        note "B-n$n0 NO BOOT/WARMUP (do not promote)"
    fi
done
note "testing elapsed $(( $(date +%s) - SLOT_START )) seconds; returning to current daily"
exit 0

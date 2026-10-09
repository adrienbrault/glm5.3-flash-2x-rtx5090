#!/usr/bin/env bash
# R929: splitdev2 + ring2/FIX1. No daily configuration is edited by this unit.
# Packet: /srv/qwen5090/r929/ring3 = this entire out/ directory.
# Helpers: R919's glm_arms.sh, launch-glm53.sh, glm53_plan.py, glm53_probe.py,
# glm53_verify.py, mtp_steps.py; glm_arms.sh supplies r860_score.py in HERE.
# arms_init owns daily restoration and GPU-queue cleanup, as in R919.
# K: both-device retained GPU tests; SH: daily 100,104 + ring/shadow, 15-minute workload.
# ABA at CACHE_TOKENS=262144, MAX_SEQ=262144:
# A1 = daily image, 100,104, ring off; B = ring3, first warm/headroom pass among
# 96,100 -> 97,101 -> 98,102 -> 99,103; A2 = same daily image/settings as A1.
# Every measured arm: c1 code/prose/chat/html/edit x2 + server step, distinct
# c1/c2/c4 x2, vision, VRAM after warmup and after measurement. No 524k arm.
# Run on the operator host only:
# sudo systemd-run --unit=r929-glm53-ring-splitdev --property=RuntimeMaxSec=43200 \
#   bash /srv/qwen5090/r929/ring3/r929-glm53-ring-splitdev.sh
set -uo pipefail
export HOME=${HOME:-/root}  # systemd-run gives no HOME; glm_arms.sh boot() reads it under set -u (run 1)
UNIT=r929
PK=/srv/qwen5090/r929
HERE=/srv/qwen5090/r929-tools  # OWN working dir: arms_init removes it.
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3
DAILY_IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
PACKET=$PK/ring3
HELPER=$PACKET/r929_helpers.py
R=/srv/qwen5090/results/$(date +%F)-r929-glm53-ring-splitdev-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r929-glm53-ring-splitdev
. /srv/qwen5090/lib/gpu-queue.sh || exit 2
. "$PK/glm_arms.sh" || exit 2
arms_init || exit 2
cp "$PK/glm53_verify.py" "$HERE/" || { note 'verify overlay FAILED'; exit 2; }
for file in "$PACKET/Dockerfile" "$PACKET/ring3.patch" "$HELPER" \
            "$PACKET/tests/serve_workload.py" "$PACKET/tests/check_shadow.py" \
            "$HERE/glm53_probe.py" "$HERE/r860_score.py" "$HERE/mtp_steps.py"; do
    [[ -f "$file" ]] || { note "packet/helper missing: $file"; exit 2; }
done
# Parsing validates the pinned daily and preserves EXL3_EXTRA as one shell argument.
python3 "$HELPER" extra --daily "$DENV" --pair 100,104 --ring 0 --shadow 0 \
    > "$R/daily-extra.txt" || { note 'daily is not the expected splitdev2 100,104 baseline'; exit 2; }
printf '%s\n' "$DENV" > "$R/daily.env"
gpu_lock || exit 2
note "GPU lock held; results $R; pinned daily $DAILY_IMG (100,104)"
if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
    note "building $IMG"
    timeout -k 30 5400 sudo -n docker build -f "$PACKET/Dockerfile" -t "$IMG" "$PACKET" \
        > "$R/build.log" 2>&1 || { note 'ring3 build FAILED; see build.log'; exit 2; }
fi
sudo -n docker image inspect "$IMG" "$DAILY_IMG" > "$R/images.json" || exit 2
SLOT_START=$SECONDS
printf 'tag\tpair\tring\tshadow\tstage\tfree0_MiB\tfree1_MiB\n' > "$R/arms.tsv"

cap(){ local left=$((WORK_END - SECONDS)); ((left >= 1)) || left=1; ((left < $1)) && echo "$left" || echo "$1"; }
have(){ (( WORK_END - SECONDS >= $1 )); }
aboot(){
    local dir=$1; shift; local time_left=$((WORK_END - SECONDS - 60))
    ((time_left >= 120)) || { note "phase time exhausted before boot $dir"; return 1; }
    ((time_left > 900)) && time_left=900
    boot "$dir" "$HERE" "$@" BOOT_TIMEOUT=$time_left
}
vram(){
    local tag=$1 stage=$2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$R/$tag/vram-$stage.csv" || return 1
    read -r FREE0 FREE1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
    [[ $FREE0 =~ ^[0-9]+$ && $FREE1 =~ ^[0-9]+$ ]] || { note "$tag cannot read both GPU headroom values"; return 1; }
    note "$tag free MiB ($stage): GPU0=$FREE0 GPU1=$FREE1"
}
engine_ok(){
    ! grep -aqE 'RING-SHADOW-MISMATCH|illegal memory access|device-side assert|CUDA error|Traceback \(most recent call last\)|[Oo]ut of memory|OutOfMemory' "$1"
}
STAGE=; BOOTED=0; FREE0=0; FREE1=0
armup(){  # tag image pair ring shadow; no unlabelled pair/pool fallback.
    local tag=$1 image=$2 pair=$3 ring=$4 shadow=$5 extra D=$R/$1
    BOOTED=0; FREE0=0; FREE1=0; STAGE=boot
    mkdir -p "$D"
    extra=$(python3 "$HELPER" extra --daily "$DENV" --pair "$pair" --ring "$ring" --shadow "$shadow") || return 1
    sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
    sleep 3
    note "$tag TRY: image=$image split=$pair ring=$ring shadow=$shadow pool=262144"
    # DENV is the same assignment envelope expanded by R919, not shell-evaluated.
    # OFFLOAD_N=104 is the fallback; splitdev2's explicit per-device pair is authoritative.
    # shellcheck disable=SC2086
    if ! aboot "$D/boot" $DENV GLM_IMG="$image" INDEX_RING="$ring" OFFLOAD_N=104 \
            CACHE_TOKENS=262144 MAX_SEQ=262144 "EXL3_EXTRA=$extra"; then
        sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1 || true
        note "$tag NO BOOT; see $D/boot and engine-boot.log"
        printf '%s\t%s\t%s\t%s\tboot_failed\t0\t0\n' "$tag" "$pair" "$ring" "$shadow" >> "$R/arms.tsv"
        return 1
    fi
    BOOTED=1; STAGE=env
    sudo -n docker inspect glm53 > "$D/container-inspect.json" || return 1
    python3 "$HELPER" env "$D/container-inspect.json" --pair "$pair" --ring "$ring" --shadow "$shadow" \
        --image "$image" --out "$D/effective-env.json" > "$D/envcheck.log" 2>&1 \
        || { note "$tag effective env REJECT; see envcheck.log"; return 1; }
    sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1 || return 1
    STAGE=warmup
    timeout -k 10 "$(cap 600)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1 \
        || { note "$tag WARMUP FAILED; see warmup.log"; return 1; }
    python3 "$HELPER" probes "$D/warmup.jsonl" > "$D/warmup-check.log" 2>&1 \
        || { note "$tag warmup contained failed requests"; return 1; }
    sudo -n docker logs glm53 > "$D/engine-warmup.log" 2>&1 || return 1
    engine_ok "$D/engine-warmup.log" || { note "$tag engine error after warmup"; return 1; }
    vram "$tag" warmup || return 1
    STAGE=up
    printf '%s\t%s\t%s\t%s\twarm\t%s\t%s\n' "$tag" "$pair" "$ring" "$shadow" "$FREE0" "$FREE1" >> "$R/arms.tsv"
}
measure(){
    local tag=$1 D=$R/$1 l0 rc=0 score_rc step_rc
    l0=$(sudo -n docker logs glm53 2>&1 | wc -l)
    timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/c1.jsonl" --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/c1.log" 2>&1 \
        || { note "$tag c1 FAILED rc=$?"; rc=1; }
    sudo -n docker logs glm53 2>&1 | tail -n +"$((l0 + 1))" > "$D/engine-c1.log"
    python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee "$D/c1-score.txt" | tee -a "$R/summary.txt"
    score_rc=${PIPESTATUS[0]}; ((score_rc == 0)) || rc=1
    python3 "$HERE/mtp_steps.py" "$D/engine-c1.log" 10 > "$D/server-step.txt" 2>&1
    step_rc=$?; ((step_rc == 0)) || rc=1
    note "$tag server step rc=$step_rc: $(tail -1 "$D/server-step.txt" | cut -c1-200)"
    timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/dec.jsonl" --phase decode --concurrency 1,2,4 --runs 2 --distinct > "$D/dec.log" 2>&1 \
        || { note "$tag distinct decode FAILED rc=$?"; rc=1; }
    timeout -k 15 "$(cap 300)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
        --out "$D/vision.jsonl" --phase vision > "$D/vision.log" 2>&1 \
        || { note "$tag vision FAILED rc=$?"; rc=1; }
    sudo -n docker logs glm53 > "$D/engine.log" 2>&1 || rc=1
    engine_ok "$D/engine.log" || { note "$tag engine error; measurements invalid"; rc=1; }
    vram "$tag" final || rc=1
    python3 "$HELPER" probes "$D/c1.jsonl" "$D/dec.jsonl" "$D/vision.jsonl" > "$D/probe-check.log" 2>&1 \
        || { note "$tag empty/error probe rows; measurements invalid"; rc=1; }
    printf '%s\n' "$rc" > "$D/measurement.rc"
    note "$tag measurement rc=$rc; c1 score $D/c1-score.txt, distinct $D/dec.jsonl"
    return "$rc"
}

# K: retained kernels, FIX1 graph-history probe and dispatch shadow on both devices.
WORK_END=$((SECONDS + 1500))
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
for _ in $(seq 12); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{busy=1} END{exit busy}' && break
    sleep 5
done
mkdir -p "$R/K"
for g in 0 1; do
    for test in test_kernels test_graph_shadow test_dispatch_shadow; do
        have 30 || { note 'K deadline exhausted; no serving arms'; exit 3; }
        timeout -k 15 "$(cap 1200)" sudo -n docker run --rm --gpus all --ipc=host --workdir /app \
            --env PYTHONPATH=/app --entrypoint python "$IMG" \
            "/opt/index-ring/tests/$test.py" --device "cuda:$g" > "$R/K/$test-cuda$g.log" 2>&1
        rc=$?; note "K $test cuda:$g rc=$rc $(tail -1 "$R/K/$test-cuda$g.log" | cut -c1-160)"
        ((rc == 0)) || { note 'VERDICT R929 kernel FAIL; no serving arms'; exit 3; }
    done
done
note 'VERDICT R929 kernel PASS on both GPUs'

# SH: no alternate split. Boot/warmup is outside the 15-minute traffic budget.
WORK_END=$((SECONDS + 3000))
# run 2: shadow (ring + old indexer side by side) does not fit at 100,104 (Insufficient VRAM); R919 shadowed at N=108.
armup SH "$IMG" 108,112 1 1 || { note 'VERDICT R929 shadow NOT RUN at 108,112; no ABA'; exit 3; }
note 'SH traffic: serve_workload --mode shadow --minutes 15, pool=262144'
timeout -k 30 930 python3 -u "$PACKET/tests/serve_workload.py" --mode shadow --minutes 15 \
    --cache-tokens 262144 --model "$MODEL" --out "$R/SH/traffic.jsonl" > "$R/SH/workload.log" 2>&1
SH_WORK_RC=$?
sleep 30
sudo -n docker logs glm53 > "$R/SH/engine-live.log" 2>&1 || true
sudo -n docker stop -t 90 glm53 > "$R/SH/stop.log" 2>&1
SH_STOP_RC=$?
sudo -n docker logs glm53 > "$R/SH/engine.log" 2>&1 || true
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
python3 "$PACKET/tests/check_shadow.py" "$R/SH/engine.log" --traffic "$R/SH/traffic.jsonl" \
    --out "$R/SH/verdict.json" > "$R/SH/check.log" 2>&1
SH_CHECK_RC=$?
note "SH workload rc=$SH_WORK_RC stop rc=$SH_STOP_RC check rc=$SH_CHECK_RC; verdict $R/SH/verdict.json"
((SH_WORK_RC == 0 && SH_STOP_RC == 0 && SH_CHECK_RC == 0)) \
    || { note 'VERDICT R929 shadow FAIL/incomplete; no ABA'; exit 3; }
note 'VERDICT R929 shadow PASS: zero mismatched bytes and all required served paths/layers'

# A1: unmodified daily, matched pool and context limit.
WORK_END=$((SECONDS + 3600))
armup A1 "$DAILY_IMG" 100,104 0 0 && measure A1 \
    || { note 'VERDICT R929 A1 failed; no valid ABA control'; exit 4; }

# B: ascending per-device search, keeping the same daily GPU split and pool.
LOWBOOT=; LOWPASS=; BTAG=; B_RC=1
WORK_END=$((SECONDS + 7200))
for pair in 96,100 97,101 98,102 99,103; do
    tag=B-${pair/,/-}
    if armup "$tag" "$IMG" "$pair" 1 0; then
        LOWBOOT=${LOWBOOT:-$pair}
        if ((FREE0 >= 450 && FREE1 >= 300)); then
            LOWPASS=$pair; BTAG=$tag
            note "$tag selected: first pair with warm headroom >=450/300 MiB"
            measure "$tag"; B_RC=$?
            if ((FREE0 < 450 || FREE1 < 300)); then
                note "$tag final headroom FAIL ($FREE0/$FREE1 MiB)"; B_RC=1
            fi
            break
        fi
        note "$tag warm headroom REJECT ($FREE0/$FREE1 MiB); trying next listed pair"
    else
        ((BOOTED == 0)) || LOWBOOT=${LOWBOOT:-$pair}
        note "$tag REJECT at $STAGE"
        [[ $STAGE != env ]] || { note 'wrong effective env invalidates the search'; break; }
    fi
    have 1200 || { note 'B search deadline exhausted'; break; }
done
printf 'lowest_boot=%s\nlowest_warm_headroom=%s\nB_tag=%s\nB_rc=%s\n' \
    "${LOWBOOT:-none}" "${LOWPASS:-none}" "${BTAG:-none}" "$B_RC" > "$R/capacity.txt"
note "B capacity: lowest boot=${LOWBOOT:-none}, lowest warm headroom=${LOWPASS:-none}; pool=262144"

# A2 is attempted even if B failed, so the closing daily control is still recorded.
WORK_END=$((SECONDS + 3600))
A2_RC=0
armup A2 "$DAILY_IMG" 100,104 0 0 && measure A2 || A2_RC=$?
if [[ -n $BTAG ]] && ((B_RC == 0 && A2_RC == 0)); then
    python3 "$HELPER" report "$R" "$BTAG" > "$R/aba.json" \
        || { note 'VERDICT R929 incomplete distinct matrix; no valid ABA'; exit 4; }
    note "VERDICT R929 ABA COMPLETE: A1(100,104/off), $BTAG($LOWPASS/on), A2(100,104/off); see aba.json"
else
    note "VERDICT R929 ABA FAILED/incomplete: B=$B_RC A2=$A2_RC; no promotion"; exit 4
fi
note "unit used $(((SECONDS - SLOT_START) / 60)) min after build; daily restoration is arms_init's EXIT cleanup"
note 'No automatic promotion. Compare c1 scores/server steps and B vs both daily controls; exactness is SH verdict.'
exit 0

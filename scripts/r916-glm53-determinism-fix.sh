#!/usr/bin/env bash
# One GPU slot, <=45 minutes INCLUDING a reserved daily restore. Build outside the slot.
set -uo pipefail
: "${HOME:=/root}"   # systemd-run gives no HOME; glm_arms boot passes HOME= (first run: "HOME: unbound variable")
UNIT=r916
PK=${R916_PACKET:-/srv/qwen5090/r916}
HERE=/srv/qwen5090/r916-tools
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_determ4
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=${R916_RESULTS:-/srv/qwen5090/results/$(date +%F)-r916-glm53-determinism-fix-$(date +%H%M%S)}
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r916-glm53-determinism-fix
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
gpu_lock
# Operator (2026-10-08): build here, after the lock, so the native build cannot overlap another unit's measurement;
# the 45-minute slot clock starts after the build. Context = the packet root (Dockerfile COPYs out/...).
if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
  note "building $IMG"
  ( cd "$PK" && timeout -k 30 3000 sudo -n docker build -f out/Dockerfile -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 2; }
  note "built $IMG"
fi
SLOT_END=$((SECONDS+2700))
WORK_END=$((SLOT_END-300))
TOKEN_FILE=$R/control-token.txt
python3 -c 'import secrets;print(secrets.token_hex(32))' > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
CONTROL_TOKEN=$(<"$TOKEN_FILE")
BASE_EXTRA=$(python3 - "$DENV" <<'PY'
import shlex,sys
print(dict(s.split('=',1) for s in shlex.split(sys.argv[1])).get('EXL3_EXTRA',''))
PY
)
# Keep the daily's last per-device override, including in the synthetic GPU probes.
CPU_SPLIT=$(python3 - "$BASE_EXTRA" <<'PY'
import sys
settings=dict(part.split('=',1) for part in sys.argv[1].split(';') if '=' in part)
counts=settings.get('EXL3_MOE_CPU_SPLIT_BY_DEVICE','100,104')
values=[int(v) for v in counts.split(',')]
assert len(values)==2 and all(8<=n<=255 for n in values), 'invalid per-device probe counts'
print(','.join(map(str,values)))
PY
)
BOOT_POLICY=''
LAST_BOOT_EXTRA=''
WINNER=''

note "GPU lock held; results=$R daily=$DENV CPU_SPLIT=$CPU_SPLIT; 45-minute slot, 5-minute restore reserve"
# Override only the helper's unbounded 1500-second boot budget. arms_cleanup also uses this
# function; it retains the helper's lock ownership/queue-aware/latest-daily restore behavior.
boot(){
  local d=$1 tools=$2; shift 2
  local end=$WORK_END
  [[ "$d" == "$R/exit-serve" ]] && end=$SLOT_END
  local cap=$((end-SECONDS-15))
  ((cap>15)) || { note 'NO BOOT: slot deadline reached'; return 1; }
  ((cap>285)) && cap=285
  timeout -k 10 "$cap" "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 MAX_BATCH=4 \
    SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=$((cap-10)) RUN_DIR="$d" "$@" \
    bash "$tools/launch-glm53.sh" > "$d.log" 2>&1
}
remaining(){
  local want=$1 rem=$((WORK_END-SECONDS))
  ((rem>20)) || { note 'VERDICT DEADLINE: experiment stopped with restore time reserved'; exit 3; }
  ((want<rem)) && echo "$want" || echo "$rem"
}
start(){
  local tag=$1 mode=$2 flags=$3 d=$R/$1
  local extra="$BASE_EXTRA;EXL3_DETERM3_CONTROL=1;EXL3_DETERM3_TOKEN=$CONTROL_TOKEN;$flags"
  local config=() draft=()
  if [[ "$mode" == static ]]; then read -r -a config <<< "$STATIC"; draft=(DRAFT=0); fi
  mkdir -p "$d"; remaining 20 >/dev/null
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  note "$tag TRY mode=$mode candidates=$flags"
  # shellcheck disable=SC2086
  boot "$d/boot" "$HERE" $DENV "${config[@]}" "${draft[@]}" GLM_IMG=$IMG TEST_DIR="$R" \
    OFFLOAD_N=104 EXL3_EXTRA="$extra" || { note "VERDICT $tag NO_BOOT"; return 1; }
  LAST_BOOT_EXTRA=$flags
  sudo -n docker inspect glm53 --format '{{json .Config.Env}}' > "$d/container-env.json"
  sudo -n docker logs glm53 > "$d/engine-boot.log" 2>&1
  note "$tag booted; launcher's fixed warmups retained"
}
settings(){ # arm tag flags [trace container path]
  local tag=$1 flags=$2 trace=${3:-}
  timeout -k 5 "$(remaining 120)" python3 -u "$PK/control.py" --token-file "$TOKEN_FILE" \
    --flags "$flags" --trace "$trace" --out "$R/$tag/settings.json" > "$R/$tag/control.log" 2>&1 \
    || { note "VERDICT $tag CONTROL_ERROR"; exit 2; }
}
pair(){ # tag flags [trace path]
  local tag=$1 flags=$2 trace=${3:-} rc verdict
  mkdir -p "$R/$tag"; settings "$tag" "$flags" "$trace"
  timeout -k 10 "$(remaining 600)" python3 -u "$PK/collect.py" collect --fixtures "$PK/fixtures.jsonl" \
    --out "$R/$tag" > "$R/$tag/collect.log" 2>&1 \
    || { note "VERDICT $tag INCOMPLETE_OR_ERROR"; sudo -n docker logs glm53 > "$R/$tag/engine.log" 2>&1; exit 2; }
  # Turning trace off flushes every queued digest before comparisons.
  settings "$tag" "$flags" ''
  verdict=$(python3 "$PK/collect.py" compare "$R/$tag" 2>&1); rc=$?
  python3 "$PK/collect.py" speed "$R/$tag" > "$R/$tag/speed.json" || exit 2
  sudo -n docker logs glm53 > "$R/$tag/engine.log" 2>&1
  note "VERDICT $tag $verdict"
  ((rc<2)) || exit 2
  return "$rc"
}
# Microprobes run in fresh processes on both devices BEFORE loading the large model.
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
for dev in 0 1; do
  timeout -k 10 "$(remaining 180)" sudo -n docker run --rm --gpus all --ipc=host \
    -v "$PK":/r916:ro --entrypoint /usr/bin/env "$IMG" EXL3_DETERMINISM_CHECK=1 \
    EXL3_MOE_CPU_SPLIT_BY_DEVICE="$CPU_SPLIT" \
    python /r916/fused_probe.py --device "$dev" > "$R/fused-$dev.log" 2>&1 \
    || { note "VERDICT kernels-$dev FAIL"; exit 2; }
done
timeout -k 10 "$(remaining 180)" sudo -n docker run --rm --gpus all --ipc=host \
  -v "$PK":/r916:ro --entrypoint /usr/bin/env "$IMG" EXL3_DETERMINISTIC=1 PYTHONHASHSEED=0 \
  EXL3_MOE_CPU_SPLIT_BY_DEVICE="$CPU_SPLIT" \
  python /r916/gpu_tests.py > "$R/gpu-tests.log" 2>&1 || { note 'VERDICT changed-kernels FAIL'; exit 2; }
note 'VERDICT kernels PASS: producer/contracts/hash/penalties/gather on both GPUs, 100 timing repeats'

start S0 static '' || exit 2
pair T0 '' '/test-results/T0/trace' || true
python3 "$PK/first_divergence.py" "$R/T0/trace" "$R/T0/trace" --left-repeat 1 --right-repeat 2 --json \
  > "$R/first-divergence.jsonl" 2> "$R/first-divergence.err"
trace_rc=$?
((trace_rc<2)) || { note 'VERDICT localizer INCOMPLETE'; exit 2; }
note "VERDICT localizer rc=$trace_rc details=$R/first-divergence.jsonl"
# Trace changes scheduling, so ALWAYS repeat baseline without it before trying candidates.
CANDIDATES=(
  ''
  'EXL3_ORDERED_MOE=1'
  'EXL3_FIXED_STREAM_T=8'
  'EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8'
  'EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8;EXL3_NO_PREFIX_REUSE=1'
  'EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8;EXL3_NO_PREFIX_REUSE=1;EXL3_ORDERED_HANDOFF=1'
  'EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8;EXL3_NO_PREFIX_REUSE=1;EXL3_ORDERED_HANDOFF=1;EXL3_EAGER_DECODE=1'
)
STATIC_ARM=''
WINNER_INDEX=-1
for i in "${!CANDIDATES[@]}"; do
  flags=${CANDIDATES[$i]}
  if pair "C$i" "$flags"; then WINNER=$flags; STATIC_ARM=C$i; WINNER_INDEX=$i; break; fi
done
if [[ -z "$STATIC_ARM" ]]; then
  BOOT_POLICY='EXL3_DETERMINISTIC=1'
  start F0 static "$BOOT_POLICY" || exit 2
  pair F0 '' || { note 'VERDICT static NO_REPRODUCIBLE_CANDIDATE'; exit 1; }
  WINNER=''; STATIC_ARM=F0
fi
verify_static(){
  local repeat_tag=$1 verdict rc
  start "$repeat_tag" static "$BOOT_POLICY;$WINNER" || exit 2
  pair "$repeat_tag" "$WINNER" || return 1
  verdict=$(python3 "$PK/collect.py" compare "$R/$STATIC_ARM" "$R/$repeat_tag" 2>&1); rc=$?
  note "VERDICT static-cross-boot-$repeat_tag $verdict"
  ((rc<2)) || exit 2
  return "$rc"
}
if ! verify_static S1; then
  # Continue in this second process with the remaining independent/cumulative trials.
  FOUND=0
  if [[ -z "$BOOT_POLICY" ]]; then
    for ((j=WINNER_INDEX+1;j<${#CANDIDATES[@]};j++)); do
      flags=${CANDIDATES[$j]}
      if pair "retry-C$j" "$flags"; then
        WINNER=$flags; STATIC_ARM=retry-C$j
        if verify_static "retry-S$j"; then FOUND=1; break; fi
      fi
    done
  fi
  if ((FOUND==0)); then
    [[ -z "$BOOT_POLICY" ]] || { note 'VERDICT static-policy-cross-boot DIFFERS'; exit 1; }
    BOOT_POLICY='EXL3_DETERMINISTIC=1'; WINNER=''; STATIC_ARM=F0retry
    start F0retry static "$BOOT_POLICY" || exit 2
    pair F0retry '' || { note 'VERDICT static-policy DIFFERS'; exit 1; }
    verify_static F1retry || { note 'VERDICT static-policy-cross-boot DIFFERS'; exit 1; }
  fi
fi
# Daily baseline uses the base image/config in its own boot; no candidate, MTP and exchange live.
DAILY_IMG=$(python3 - "$DENV" <<'PY'
import shlex,sys
print(dict(s.split('=',1) for s in shlex.split(sys.argv[1]))['GLM_IMG'])
PY
)
mkdir -p "$R/Daily"
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
# shellcheck disable=SC2086
boot "$R/Daily/boot" "$HERE" $DENV TEST_DIR= TEST_PYTHONPATH= || { note 'VERDICT daily-baseline NO_BOOT'; exit 2; }
timeout -k 10 "$(remaining 180)" python3 -u "$PK/collect.py" collect --fixtures "$PK/fixtures.jsonl" \
  --out "$R/Daily" > "$R/Daily/collect.log" 2>&1 || { note 'VERDICT daily-baseline INCOMPLETE'; exit 2; }
python3 "$PK/collect.py" speed "$R/Daily" > "$R/Daily/speed.json" || exit 2
sudo -n docker logs glm53 > "$R/Daily/engine.log" 2>&1
note "VERDICT daily-baseline measured image=$DAILY_IMG MTP+exchange c1 five kinds x2"
# Full policy suppresses exchange itself. Otherwise, test the winning set with actual dynamic exchange first.
start D0 daily "$BOOT_POLICY;$WINNER" || exit 2
DAILY_ARM=D0
if ! pair D0 "$WINNER"; then
  [[ -z "$BOOT_POLICY" ]] || { note 'VERDICT daily NO_REPRODUCIBLE_CANDIDATE'; exit 1; }
  # Freeze from a FRESH boot's initial placement, never the history-dependent map after Daily/D0.
  WINNER="${WINNER:+$WINNER;}EXL3_FREEZE_EXCHANGE=1"
  start Dfreeze daily "$WINNER" || exit 2
  if pair Dfreeze "$WINNER"; then
    DAILY_ARM=Dfreeze
  else
    # MTP verify/draft shapes can expose a seam absent from the plain/static arm.
    DAILY_ARM=''
    for f in EXL3_ORDERED_MOE=1 EXL3_FIXED_STREAM_T=8 EXL3_NO_PREFIX_REUSE=1 EXL3_ORDERED_HANDOFF=1 EXL3_EAGER_DECODE=1; do
      [[ ";$WINNER;" == *";$f;"* ]] && continue
      WINNER="$WINNER;$f"; tag=Daily-${f%%=*}
      if pair "$tag" "$WINNER"; then DAILY_ARM=$tag; break; fi
    done
    if [[ -z "$DAILY_ARM" ]]; then
      BOOT_POLICY='EXL3_DETERMINISTIC=1'; WINNER=''; DAILY_ARM=Daily-policy
      start "$DAILY_ARM" daily "$BOOT_POLICY" || exit 2
      pair "$DAILY_ARM" '' || { note 'VERDICT daily-policy DIFFERS'; exit 1; }
    fi
  fi
fi
start D1 daily "$BOOT_POLICY;$WINNER" || exit 2
pair D1 "$WINNER" || { note 'VERDICT daily-second-boot DIFFERS'; exit 1; }
verdict=$(python3 "$PK/collect.py" compare "$R/$DAILY_ARM" "$R/D1" 2>&1); rc=$?
note "VERDICT daily-cross-boot $verdict"
((rc==0)) || exit 1
python3 "$PK/collect.py" speed "$R/$DAILY_ARM" --baseline "$R/Daily" | tee "$R/daily-cost.json" || exit 2
printf '%s\n' "$BOOT_POLICY;$WINNER" > "$R/winner.env"
python3 - "$R/daily-cost.json" "$R/winner.env" <<'PY' | tee -a "$R/summary.txt"
import json,sys
cost=json.load(open(sys.argv[1]))['cost_percent']
print(f'VERDICT PASS daily c1 reproducible within/across boots; measured cost={cost:.2f}%; flags='+open(sys.argv[2]).read().strip())
print('PROMOTION_REVIEW '+('eligible cost<=1%; repeated speed confirmation still needed' if cost<=1 else 'keep opt-in; cost>1%'))
PY
note "slot elapsed=$((2700-(SLOT_END-SECONDS)))s; cleanup restores latest daily if no other unit queued"
exit 0

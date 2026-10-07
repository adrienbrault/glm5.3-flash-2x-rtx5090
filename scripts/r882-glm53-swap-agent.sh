#!/usr/bin/env bash
# R882 (2026-10-07): serve r3 swaps with the agent overlay. R875c (same run): static SB 56.7 c1, XI (exchange from the static
# hot set, histogram policy, exact cadence /64, global 64, floor 4, hyst 2.0) 60.6 (+6.9 %), XIF 58.0. cheapswap-r3 has no
# agent overlay, so tabbyapi:cheapswap-r3-agent-r2 stacks the r2 overlay (TabbyAPI /app) on it
# (patches/tabbyapi/glm-agent-r2-on-cheapswap-r3; disjoint trees).
#   build  the combo image
#   XA     XI env on the combo image + AGENT=1: c1 five kinds x2, measure c1/c2/c4 + 8k/32k prefill, vision, toolcheck simple
#   S      serve XA if c1 >= 59.0, vision PASS and simple toolcheck passes; else static + r2 overlay (today's daily)
#   sudo systemd-run --unit=r882-glm53-swap-agent --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r882-glm53-swap-agent.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r882-tools
R=/srv/qwen5090/results/$(date +%F)-r882-glm53-swap-agent-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r875] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"  # exit relaunch
export GPU_QUEUE_NAME=r882-glm53-swap-agent
. /srv/qwen5090/lib/gpu-queue.sh
BEST_ENV=""
cleanup(){
  local rc=$?
  trap - EXIT INT TERM HUP
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
    note "no GLM serving at exit: relaunching $BASE $BEST_ENV"
    # shellcheck disable=SC2086
    "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 MAX_BATCH=4 \
      SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$R/exit-serve" DRAFT=0 $THREADS $BASE $BEST_ENV \
      bash "$HERE/launch-glm53.sh" > "$R/exit-serve.log" 2>&1 || note "exit relaunch FAILED (exit-serve.log)"
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  echo "exit=$rc" > "$R/last.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'note interrupted; exit 130' INT
trap 'note terminated; exit 143' TERM HUP
gpu_lock
note "GPU lock held; results $R; base $BASE"
sudo -n docker logs glm53 > "$R/glm-before.log" 2>&1 || true

run_arm(){  # tag, probe kinds, then KEY=VALUE launcher env
  local tag=$1 kinds=$2; shift 2
  local D=$R/$tag rc=0
  mkdir -p "$D"
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  # shellcheck disable=SC2086
  timeout -k 20 1500 "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 \
    CHUNK=2048 MAX_BATCH=4 SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$D" $THREADS $PIN "$@" \
    bash "$HERE/launch-glm53.sh" > "$D/boot.log" 2>&1 || rc=$?
  if ((rc != 0)); then note "$tag NO BOOT rc=$rc ($(grep -ahoE 'Insufficient VRAM|out of memory|ABORT[^;]*' "$D"/*.log | head -1))"; return 1; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$D/vram-up.csv"; free -b > "$D/free-up.txt"
  [[ -z "$kinds" ]] && return 0
  timeout -k 15 7200 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/c1.jsonl" --phase c1 --runs 2 --kinds "$kinds" > "$D/c1.log" 2>&1 || rc=$?
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee -a "$R/summary.txt"
  ((rc == 0)) || note "$tag PROBE FAILED rc=$rc: $(grep -a 'Error' "$D/c1.log" | tail -1 | cut -c1-200)"
  return "$rc"
}
teardown(){ sudo -n docker logs glm53 > "$R/$1/engine.log" 2>&1 || true; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; }
ALL=code,prose,chat,html,edit
STATS_FILE=$HERE/split-stats-broad-r869.json
STATIC="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$STATS_FILE"
XBASE="OFFLOAD_N=96 GLM_IMG=tabbyapi:cheapswap-r3 PINNED_ARENA=1 SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_INIT_STATS=$STATS_FILE"
MODEL_DIR=/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp
REVIEW=/srv/qwen5090/cheapswap-r3-build
# shellcheck disable=SC2086
arm(){ local tag=$1; shift; run_arm "$tag" "$ALL" DRAFT=0 "$@"; local rc=$?; teardown "$tag"; return $rc; }
mkdir -p "$R/T"; TOK=1
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
# R875b (2026-10-07): R875's init-sw1-fo0 gate died at model load with "CUDA-capable device(s) is/are busy or unavailable"
# right after the previous probe container exited (infrastructure, not the patch; init-sw1-fo1 passed all 42 layers).
# Settle 15 s before each gate and retry once on that exact error. SB2 dropped (two same-day SB controls: 57.6, 57.7).
selftest(){ local tag=$1; shift; local try rc
  for try in 1 2; do
    sleep 15
    timeout -k 15 1800 sudo -n docker run --rm --gpus all --ipc=host "$@" > "$R/T/$tag.log" 2>&1
    rc=$?
    ((rc != 0)) && grep -q 'busy or unavailable' "$R/T/$tag.log" && { note "selftest $tag try $try: GPU busy/unavailable, retrying"; cp "$R/T/$tag.log" "$R/T/$tag.try$try.log"; continue; }
    break
  done
  note "selftest $tag rc=$rc: $(tail -1 "$R/T/$tag.log" | cut -c1-160)"; ((rc == 0)) || TOK=0; }
BUILDCTX=/srv/qwen5090/r882-build
( sudo -n docker build -f "$BUILDCTX/glm-agent-r2-on-cheapswap-r3/Dockerfile" -t tabbyapi:cheapswap-r3-agent-r2 "$BUILDCTX/glm-agent-r2" ) > "$R/build.log" 2>&1
BUILD_RC=$?; note "build cheapswap-r3-agent-r2 rc=$BUILD_RC: $(tail -1 "$R/build.log" | cut -c1-120)"
XA_ENV="${XBASE/GLM_IMG=tabbyapi:cheapswap-r3/GLM_IMG=tabbyapi:cheapswap-r3-agent-r2} SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0 AGENT=1"
OK=0
# shellcheck disable=SC2086
if ((BUILD_RC == 0)) && run_arm XA "$ALL" DRAFT=0 $XA_ENV; then
  timeout -k 15 5400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/XA/measure.jsonl" \
    --phase measure --runs 3 > "$R/XA/measure.log" 2>&1 || note "XA measure rc=$?"
  python3 - "$R/XA/measure.jsonl" <<'PY2' | tee -a "$R/summary.txt"
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get('phase') == 'decode-summary':
        for s in r['summaries']:
            print(f"XA c{s['c']}: aggregate {s['ss_agg_tps_median']:.1f} tok/s, per stream {s['ss_per_stream_tps_median']:.1f}")
    elif 'prefill' in str(r.get('phase')):
        print('XA prefill', r.get('tag'), 'engine_prefill_tps', r.get('engine_prefill_tps'))
PY2
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/XA/vision.jsonl" --phase vision \
    > "$R/XA/vision.log" 2>&1 && note "XA vision PASS" || note "XA vision FAILED: $(grep -a 'vision check failed' "$R/XA/vision.log" | tail -1 | cut -c1-200)"
  timeout -k 15 1800 python3 -u "$HERE/glm53_toolcheck_r3.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/XA/toolcheck" \
    --repeats 2 --modes stream --temperatures zero default --cases simple > "$R/XA/toolcheck.log" 2>&1
  note "XA toolcheck simple rc=$?"
  C1=$(grep -aoE '^XA c1 score [0-9.]+' "$R/summary.txt" | awk '{print $4}')
  grep -aq '^XA vision PASS' "$R/summary.txt" && grep -aq '^XA toolcheck simple rc=0' "$R/summary.txt" \
    && python3 -c "import sys; sys.exit(0 if float('${C1:-0}') >= 59.0 else 1)" && OK=1
  note "XA gate: c1 ${C1:-?}, OK=$OK"
fi
teardown XA
if ((OK)); then
  # shellcheck disable=SC2086
  run_arm S "" DRAFT=0 $XA_ENV && note "SERVING on :8029: r3 swaps (XI) from the static hot set + agent overlay r2, N=96, MTP off, vision on, 256k ($XA_ENV)"
else
  # shellcheck disable=SC2086
  run_arm S "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1 && note "SERVING on :8029: static N=96 + agent overlay r2 (XA not promoted)"
fi
exit 0

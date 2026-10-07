#!/usr/bin/env bash
# R883 (2026-10-08): MTP depth 1/2 on the served r3-swap config (XA). R864 measured MTP depth 1 before the swaps:
# +7-10 % on code/edit (acceptance 0.85-0.95), nothing on prose/chat, because the verify row routes to other experts and
# doubles the CPU expert work. Swaps cut CPU work per token (c1 46 -> 59.9), so a verify row now costs less.
# MTP needs VRAM: depth 1 booted at N=104 default chunk/batch (R864), at N=100 only with CHUNK=1024 MAX_BATCH=2 (R867).
#   C      XA (N=96, no draft), c1 five kinds x2: same-session control
#   M1     XA + DRAFT_N=1 at N=104, c1 x2 + measure (c1/c2/c4, prefill)  [fallback N=100 CHUNK=1024 MAX_BATCH=2]
#   M2     XA + DRAFT_N=2 at N=112, c1 x2 + measure                       [fallback N=112 CHUNK=1024 MAX_BATCH=2]
#   S      serve the best MTP arm if c1 >= 1.08 x C, c4 aggregate >= 104 (0.95 x R882b 109.9), vision + simple toolcheck;
#          else XA (R882c's daily)
#   sudo systemd-run --unit=r883-glm53-swap-mtp --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r883-glm53-swap-mtp.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r883-tools
R=/srv/qwen5090/results/$(date +%F)-r883-glm53-swap-mtp-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r883] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"  # exit relaunch
export GPU_QUEUE_NAME=r883-glm53-swap-mtp
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
XBASE="OFFLOAD_N=96 GLM_IMG=tabbyapi:cheapswap-r3-agent-r2 PINNED_ARENA=1 SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_INIT_STATS=$STATS_FILE"
MODEL_DIR=/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp
REVIEW=/srv/qwen5090/cheapswap-r3-build
# shellcheck disable=SC2086
arm(){ local tag=$1; shift; run_arm "$tag" "$ALL" DRAFT=0 "$@"; local rc=$?; teardown "$tag"; return $rc; }
XA_ENV="GLM_IMG=tabbyapi:cheapswap-r3-agent-r2 PINNED_ARENA=1 SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_INIT_STATS=$STATS_FILE SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0 AGENT=1"
BASE="OFFLOAD_N=96 $XA_ENV"
SM="CHUNK=1024 MAX_BATCH=2 GPU_SPLIT=31,31.8"
score(){ grep -aoE "^$1 c1 score [0-9.]+" "$R/summary.txt" | tail -1 | awk '{print $4}'; }
measure(){  # tag
  timeout -k 15 3600 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/$1/measure.jsonl" \
    --phase measure --runs 1 > "$R/$1/measure.log" 2>&1 || note "$1 measure rc=$?"
  python3 - "$R/$1/measure.jsonl" "$1" <<'PY2' | tee -a "$R/summary.txt"
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get('phase') == 'decode-summary':
        for s in r['summaries']:
            print(f"{sys.argv[2]} c{s['c']}: aggregate {s['ss_agg_tps_median']:.1f} tok/s, per stream {s['ss_per_stream_tps_median']:.1f}")
    elif 'prefill' in str(r.get('phase')):
        print(sys.argv[2], 'prefill', r.get('tag'), 'engine_prefill_tps', r.get('engine_prefill_tps'))
PY2
}
c4(){ grep -aoE "^$1 c4: aggregate [0-9.]+" "$R/summary.txt" | tail -1 | awk '{print $4}'; }
mtp(){  # tag, then env; tries default chunk/batch, then the small-batch fallback
  local tag=$1; shift
  if run_arm "$tag" "$ALL" "$@"; then measure "$tag"; teardown "$tag"; return 0; fi
  teardown "$tag"
  if run_arm "${tag}s" "$ALL" "$@" $SM; then measure "${tag}s"; teardown "${tag}s"; return 0; fi
  teardown "${tag}s"; return 1
}
# shellcheck disable=SC2086
run_arm C "$ALL" DRAFT=0 $BASE; teardown C
CC=$(score C); CC=${CC:-59.9}
# shellcheck disable=SC2086
mtp M1 DRAFT=1 DRAFT_N=1 OFFLOAD_N=104 $XA_ENV
# shellcheck disable=SC2086
mtp M2 DRAFT=1 DRAFT_N=2 OFFLOAD_N=112 $XA_ENV
WIN=""; WS=0
for t in M1 M1s M2 M2s; do
  s=$(score $t); c=$(c4 $t); [[ -z "$s" || -z "$c" ]] && continue
  if python3 -c "import sys; s,c,cc,ws=map(float,sys.argv[1:]); sys.exit(0 if s>=1.08*cc and c>=104 and s>ws else 1)" "$s" "$c" "$CC" "$WS"; then WIN=$t; WS=$s; fi
done
note "control C c1 $CC; winner: ${WIN:-none} ${WS}"
SERVE="DRAFT=0 $BASE"
if [[ -n "$WIN" ]]; then
  SERVE=$(grep -a "^$WIN TRY: " "$R/summary.txt" | tail -1 | sed "s/^$WIN TRY: //")
  # shellcheck disable=SC2086
  if run_arm S "" $SERVE; then
    python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/vision.jsonl" --phase vision > "$R/S/vision.log" 2>&1; V=$?
    timeout -k 15 1800 python3 -u "$HERE/glm53_toolcheck_r3.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/toolcheck" \
      --repeats 2 --modes stream --temperatures zero default --cases simple > "$R/S/toolcheck.log" 2>&1; T=$?
    note "S vision rc=$V toolcheck simple rc=$T"
    if ((V == 0 && T == 0)); then note "SERVING on :8029: MTP $WIN ($SERVE)"; exit 0; fi
  fi
  teardown S; SERVE="DRAFT=0 $BASE"
fi
# shellcheck disable=SC2086
run_arm S2 "" $SERVE && note "SERVING on :8029: r3 swaps + agent r2 ($SERVE)" || teardown S2
exit 0

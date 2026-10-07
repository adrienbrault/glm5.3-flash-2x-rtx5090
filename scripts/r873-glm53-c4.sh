#!/usr/bin/env bash
# R873 (2026-10-07): the config about to be served (R869 static broad placement N=96, MTP off, r861 agent overlay, vision on,
# 256k), measured with glm53_probe.py --phase measure: steady-state decode at c1/c2/c4 (per-stream and aggregate, 3 runs
# each) and cold prefill at 8k/32k. The last c2/c4 numbers (60.1/68.6 aggregate) are from R858/R859 on N=104 dynamic.
# Leaves that config serving.
#   sudo systemd-run --unit=r873-glm53-c4 --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r873-glm53-c4.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r873-tools
R=/srv/qwen5090/results/$(date +%F)-r873-glm53-c4-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r873] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"  # exit relaunch
export GPU_QUEUE_NAME=r873-glm53-c4
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
STATIC="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"
# shellcheck disable=SC2086
if run_arm M "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  timeout -k 15 5400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/M/measure.jsonl" \
    --phase measure --runs 3 > "$R/M/measure.log" 2>&1 || note "M probe rc=$?"
  python3 - "$R/M/measure.jsonl" <<'PY2' | tee -a "$R/summary.txt"
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get('phase') == 'decode-summary':
        for s in r['summaries']:
            print(f"c{s['c']}: aggregate {s['ss_agg_tps_median']:.1f} tok/s, per stream {s['ss_per_stream_tps_median']:.1f}")
    elif 'prefill' in str(r.get('phase')):
        print('prefill', {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != 'phase'})
PY2
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/M/vision.jsonl" --phase vision \
    > "$R/M/vision.log" 2>&1 && note "vision check PASS" || note "vision check FAILED"
  note "SERVING on :8029: static broad placement N=96, MTP off, agent overlay, vision on, 256k"
else
  teardown M
fi
exit 0

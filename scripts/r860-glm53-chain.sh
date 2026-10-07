#!/usr/bin/env bash
# R860 (2026-10-07, user: "wait for my DSH pagoda session to complete before you take back GPU. once my session is
# done, do not stop experimenting/iterating"): GLM-5.3-Flash c1 chain on flan, one boot per arm, 256k cache.
#   0  wait until GLM on :8029 has had no request start and no open gateway connection for IDLE_MIN minutes
#   1  MTP: depth 1 / 2 / 3, depth 3 + dynamic_draft, MTP off — c1 decode code/prose/chat (1,024) + html (2,048) x2
#   2  diagnostics: host->device bandwidth probe; predict-trace image (route trace + next-layer prediction + CPU-MoE
#      profilers, DRAFT=0) — c1 x1 + 32k depth; graceful stop so the trace writer drains
#   3  CPU knobs on the phase-1 winner: 16 threads, 7 threads, pinned arena — c1 x2
#   4  serve the fastest measured arm on :8029 (KEEP_GLM=1) for DeepSeek Harness
# Score = mean over the four kinds of the per-kind median c1 decode rate.
set -uo pipefail
: "${HOME:=/root}"  # systemd-run units start without HOME; the chain forwards it to the launchers
HERE=/srv/qwen5090/r858
P2=/srv/qwen5090/r859-prefetch-r2
IDLE_MIN=${IDLE_MIN:-10}
R=/srv/qwen5090/results/$(date +%F)-r860-glm53-chain-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r860] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
LIVE=/srv/qwen5090/launch-flashnext.sh
PARENT_ID=sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"

# ---- 0: wait for the user's session to finish (GLM serving DeepSeek Harness through Olla) ----
glm_busy(){
  local active
  active=$(curl -s -m 5 127.0.0.1:40114/internal/status/endpoints | python3 -c 'import json,sys
d=json.load(sys.stdin); print(sum(e.get("active_connections",0) for e in d["endpoints"] if e["name"]=="flan-glm53"))' 2>/dev/null || echo 1)
  [[ "$active" != 0 ]] && return 0
  sudo -n docker logs --since "${IDLE_MIN}m" glm53 2>&1 | grep -aqE '#[0-9]+ (chat/)?completions( \(stream\))?: [0-9,]+ prompt tokens' && return 0
  return 1
}
note "waiting for GLM to be idle for ${IDLE_MIN} min (no request start, no open gateway connection)"
while sudo -n docker ps --format '{{.Names}}' | grep -qx glm53 && glm_busy; do sleep 60; done
note "GLM idle; taking the GPUs"

export GPU_QUEUE_NAME=r860-glm53-chain
. /srv/qwen5090/lib/gpu-queue.sh
cleanup(){
  local rc=$?
  trap - EXIT INT TERM HUP
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
    note "no GLM serving at exit: relaunching the R859 base config"
    "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split OFFLOAD_N=104 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 \
      SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 DRAFT=1 CPU_THREADS=8 BOOT_TIMEOUT=1200 RUN_DIR="$R/exit-serve" \
      bash "$HERE/launch-glm53.sh" > "$R/exit-serve.log" 2>&1 || note "exit relaunch FAILED (see exit-serve.log)"
  fi
  rm -f "$GPU_QUEUE_MARK"
  echo "exit=$rc" > "$R/last.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'note interrupted; exit 130' INT
trap 'note terminated; exit 143' TERM HUP
gpu_lock
note "GPU lock held; results $R"
sudo -n docker logs glm53 > "$R/glm-before.log" 2>&1 || true

run_arm(){  # tag, launcher, probe args, then KEY=VALUE launcher env
  local tag=$1 launcher=$2 probe=$3; shift 3
  local D=$R/$tag rc=0
  mkdir -p "$D"
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  timeout -k 20 1500 "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split OFFLOAD_N=104 CACHE_TOKENS=262144 MAX_SEQ=262144 \
    CHUNK=2048 SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$D" "$@" \
    bash "$launcher" > "$D/boot.log" 2>&1 || rc=$?
  if ((rc != 0)); then note "$tag NO BOOT rc=$rc ($(grep -ahoE 'Insufficient VRAM|out of memory|NO BOOT[^;]*|ABORT[^;]*' "$D"/*.log | head -1))"; return 1; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$D/vram-up.csv"; free -b > "$D/free-up.txt"
  # shellcheck disable=SC2086
  timeout -k 15 7200 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/c1.jsonl" --phase c1 $probe > "$D/c1.log" 2>&1 || rc=$?
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee -a "$R/summary.txt"
  if ((rc != 0)); then note "$tag PROBE FAILED rc=$rc: $(grep -a 'Error' "$D/c1.log" | tail -1)"; fi
  return "$rc"
}
teardown(){  # tag [graceful]
  local D=$R/$1
  if [[ "${2:-}" == graceful ]]; then sudo -n docker stop -t 120 glm53 >/dev/null 2>&1 || true; fi
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1 || true
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
}
L=$HERE/launch-glm53.sh
DEC="--runs 2 --kinds code,prose,chat,html"

# ---- 1: MTP ----
declare -A ENV_OF=(
  [d1]="DRAFT=1 DRAFT_N=1 CPU_THREADS=8"
  [d2]="DRAFT=1 DRAFT_N=2 CPU_THREADS=8"
  [d3]="DRAFT=1 DRAFT_N=3 CPU_THREADS=8"
  [d3dyn]="DRAFT=1 DRAFT_N=3 DYN_DRAFT=1 CPU_THREADS=8"
  [off]="DRAFT=0 CPU_THREADS=8"
)
for tag in d1 d2 d3 d3dyn off; do
  # shellcheck disable=SC2086
  run_arm "$tag" "$L" "$DEC" ${ENV_OF[$tag]}; teardown "$tag"
done
W1=$(python3 "$HERE/r860_score.py" --best "$R" d1 d2 d3 d3dyn off)
note "phase 1 winner: ${W1:-none}"
[[ -n "$W1" ]] || W1=d1

# ---- 2: diagnostics ----
mkdir -p "$R/h2d"
timeout -k 10 600 sudo -n docker run --rm --gpus all -v "$P2/out/calib":/calib:ro --entrypoint python3 \
  tabbyapi:r828-prompt-lookup-r3 /calib/probe_h2d.py > "$R/h2d/h2d.jsonl" 2> "$R/h2d/h2d.err" \
  && note "h2d probe: $(tail -2 "$R/h2d/h2d.jsonl" | tr '\n' ' ' | cut -c1-300)" || note "h2d probe FAILED (h2d.err)"
if sudo -n docker image inspect tabbyapi:r859-predict-trace-r2 >/dev/null 2>&1 && [[ -f "$HERE/launch-glm53-predict-r2.sh" ]]; then
  run_arm predict "$HERE/launch-glm53-predict-r2.sh" "--runs 1 --kinds code,prose,chat,html --depths 32768" \
    DRAFT=0 CPU_THREADS=8 GLM_IMG=tabbyapi:r859-predict-trace-r2 PROFILE=1 PREDICT=1 CPU_SWAP=0 \
    ROUTE_TRACE_DIR="$R/predict/route-traces"
  teardown predict graceful
  note "predict trace files: $(find "$R/predict/route-traces" -type f 2>/dev/null | wc -l)"
else
  note "predict arm SKIPPED: image or launcher missing"
fi

# ---- 3: CPU knobs on the phase-1 winner ----
BASEENV=${ENV_OF[$W1]}
BASEENV=${BASEENV/CPU_THREADS=8/}
# shellcheck disable=SC2086
run_arm t16 "$L" "$DEC" $BASEENV CPU_THREADS=16; teardown t16
# shellcheck disable=SC2086
run_arm t7 "$L" "$DEC" $BASEENV CPU_THREADS=7; teardown t7
# shellcheck disable=SC2086
run_arm pinned "$L" "$DEC" $BASEENV CPU_THREADS=8 PINNED_ARENA=1; teardown pinned
ENV_OF[t16]="$BASEENV CPU_THREADS=16"; ENV_OF[t7]="$BASEENV CPU_THREADS=7"; ENV_OF[pinned]="$BASEENV CPU_THREADS=8 PINNED_ARENA=1"
BEST=$(python3 "$HERE/r860_score.py" --best "$R" "$W1" t16 t7 pinned)
note "overall winner: ${BEST:-$W1} (${ENV_OF[${BEST:-$W1}]})"

# ---- 4: serve it ----
# shellcheck disable=SC2086
run_arm SERVE "$L" "--runs 1 --kinds code" ${ENV_OF[${BEST:-$W1}]} || { teardown SERVE; note 'SERVE relaunch failed'; exit 4; }
note "SERVING on :8029: ${BEST:-$W1} (${ENV_OF[${BEST:-$W1}]}), 256k context"
exit 0

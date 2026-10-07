#!/usr/bin/env bash
# R876 (2026-10-07): GPU/CPU decode timeline for GLM-5.3 on the served static config (codex decode-profile-r1, torch
# profiler; no nsys in the image). The cost model says ~14 ms/token of GPU-side time remains even with zero CPU picks
# (~70 tok/s ceiling); this attributes it: attention/indexer vs KDA vs shared expert vs routed GPU experts vs handoff
# waits vs launch gaps, per device. Standalone loads (no server): base image c1 (no ranges, overhead control), overlay
# c1 with ranges, c1 with stream-wait events + native handoff aggregates, c4, 32k c1. Leaves static + agent overlay serving.
#   sudo systemd-run --unit=r876-glm53-timeline --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r876-glm53-timeline.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r876-tools
TOOLS=/srv/qwen5090/decode-profile-r1/out
R=/srv/qwen5090/results/$(date +%F)-r876-glm53-timeline-$(date +%H%M%S); mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
note(){ printf '%s [r876] %s\n' "$(date -Iseconds)" "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SERVE=(PACK=A OFFLOAD_MODE=split OFFLOAD_N=96 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 MAX_BATCH=4 SYSMEM_RC_MB=1024
       GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 DRAFT=0 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json
       GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1)
MODEL=/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp
STATS=$HERE/split-stats-broad-r869.json
TUNE=/srv/qwen5090/.exl3cache-r858-glm53
export GPU_QUEUE_NAME=r876-glm53-timeline
. /srv/qwen5090/lib/gpu-queue.sh
cleanup(){
  local rc=$?; trap - EXIT INT TERM HUP
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
    "${CLEAN[@]}" HOME="$HOME" "${SERVE[@]}" RUN_DIR="$R/serve" bash "$HERE/launch-glm53.sh" > "$R/serve.log" 2>&1 \
      && note "SERVING on :8029: static N=96 + agent overlay" || note "serve FAILED (serve.log)"
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; echo "exit=$rc" > "$R/last.txt"; exit "$rc"
}
trap cleanup EXIT
trap 'note terminated; exit 143' TERM HUP INT
gpu_lock
note "GPU lock held; results $R"
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$R/vram-before.csv"
mkdir -p "$TUNE"
timeout -k 30 1800 sudo -n docker build -t tabbyapi:r828-decode-profile-r1 "$TOOLS/overlay" > "$R/build.log" 2>&1 \
  || { note "overlay build FAILED: $(tail -2 "$R/build.log" | tr '\n' ' ' | cut -c1-200)"; exit 0; }
COMMON=(--rm --gpus all --ipc=host --shm-size=16g --restart no -v "$MODEL:/model:ro" -v "$STATS:/app/split-stats.json:ro"
        -v "$TOOLS:/tools:ro" -v "$TUNE:/exl3-cache" -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache
        -w /app --entrypoint /usr/bin/env)
view(){  # tag image env... -- profile args
  local tag=$1 img=$2; shift 2; local envs=(); while [[ $1 != -- ]]; do envs+=("$1"); shift; done; shift
  mkdir -p "$R/$tag"; sudo -n chmod 0777 "$R/$tag"
  timeout -k 30 2400 sudo -n docker run "${COMMON[@]}" -v "$R/$tag:/results" "$img" "${envs[@]}" \
    python3 -u /tools/profile_decode.py "$@" > "$R/$tag/run.log" 2>&1
  local rc=$?; note "$tag rc=$rc: $(grep -aiE 'wall_ms|tok/s|Error' "$R/$tag/run.log" | tail -2 | tr '\n' ' ' | cut -c1-220)"
  if [[ " $* " == *" --handoff-prof "* ]]; then
    python3 "$TOOLS/profile_decode.py" --analyze "$R/$tag/decode.trace.json.gz" --out "$R/$tag" --rows "${ROWS:-1}" \
      --handoff-log "$R/$tag/run.log" > "$R/$tag/analyze.log" 2>&1 || note "$tag analyze rc=$?"
  fi
}
view base-c1 tabbyapi:r828-prompt-lookup-r3 -- --steps 32 --ctx 2048 --rows 1
view graphs-c1 tabbyapi:r828-decode-profile-r1 EXL3_DECODE_PROFILE_RANGES=1 -- --steps 32 --ctx 2048 --rows 1
view waits-c1 tabbyapi:r828-decode-profile-r1 EXL3_DECODE_PROFILE_RANGES=1 EXL3_DECODE_PROFILE_WAIT_EVENTS=1 -- --steps 32 --ctx 2048 --rows 1 --handoff-prof
view graphs-c4 tabbyapi:r828-decode-profile-r1 EXL3_DECODE_PROFILE_RANGES=1 -- --steps 32 --ctx 2048 --rows 4
view graphs-32k-c1 tabbyapi:r828-decode-profile-r1 EXL3_DECODE_PROFILE_RANGES=1 -- --steps 32 --ctx 32768 --rows 1
find "$R" -maxdepth 2 -name '*.md' -o -maxdepth 2 -name '*.csv' | head -40 > "$R/outputs.txt"
exit 0

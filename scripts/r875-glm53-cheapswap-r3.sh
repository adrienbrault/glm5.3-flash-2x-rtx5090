#!/usr/bin/env bash
# R875 (2026-10-07): cheapswap-r3 = r2's exchange + initialization from the static counts file (+ an optional score
# policy). r3's replay: static-initialized exchange with r2's B-safe histogram predicts ~+7 % over static alone; the
# score policy loses to it. Review: no Critical/High. Gates first, all in tabbyapi:cheapswap-r3:
#   r2 fixtures (gpu fixture cuda:0/cuda:1, real-worker ring) and probe_static_initialization.py on the real model
#   (swizzle 0/1 x folded 0/1: every layer's GPU/worker bytes, maps and streamed prefill vs the checkpoint).
# Any gate failure skips the exchange arms. Then c1 x5 kinds x2:
#   SB   static (r828 image, control)       XI  r3 static-init + B-safe (exact/64, global 64, floor 4, hyst 2)
#   XIF  r3 static-init + B-fast (served/16, layer 64, floor 2, hyst 1.2)       SB2  static again
# Ends serving static + the agent overlay (exchange is not served from here: that is a later, explicit promotion).
#   sudo systemd-run --unit=r875-glm53-cheapswap-r3 --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r875-glm53-cheapswap-r3.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r875-tools
R=/srv/qwen5090/results/$(date +%F)-r875-glm53-cheapswap-r3-$(date +%H%M%S)
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
export GPU_QUEUE_NAME=r875-glm53-cheapswap-r3
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
selftest(){ local tag=$1; shift
  timeout -k 15 1800 sudo -n docker run --rm --gpus all --ipc=host "$@" > "$R/T/$tag.log" 2>&1
  local rc=$?; note "selftest $tag rc=$rc: $(tail -1 "$R/T/$tag.log" | cut -c1-160)"; ((rc == 0)) || TOK=0; }
selftest gpu0 --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_selftest.py --device cuda:0 --swizzle 1
selftest gpu1 --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_selftest.py --device cuda:1 --swizzle 1
selftest ring --entrypoint python tabbyapi:cheapswap-r3 /opt/cheapswap/gpu_ring_selftest.py
LK='{"use_per_device":[31,31],"max_chunk_size":2048,"max_batch_size":4}'
for sw in 1 0; do for fo in 1 0; do
  selftest "init-sw$sw-fo$fo" --entrypoint python -v "$MODEL_DIR:/model:ro" -v "$STATS_FILE:/app/split-stats.json:ro" \
    -v "$REVIEW:/review:ro" -e EXL3_MOE_CPU_SWIZZLE=$sw -e EXL3_MOE_RECON_FOLDED=$fo -e EXL3_MOE_PINNED_ARENA=1 \
    tabbyapi:cheapswap-r3 /review/probe_static_initialization.py --model /model --stats /app/split-stats.json \
    --load-kwargs "$LK" --rows 128
  ((TOK)) || break 2
done; done
arm SB $STATIC
if ((TOK)); then
  # shellcheck disable=SC2086
  arm XI $XBASE SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0
  # shellcheck disable=SC2086
  arm XIF $XBASE SWAP_CADENCE=served SWAP_INTERVAL=16 SWAP_MAX=64 SWAP_SCOPE=layer SWAP_FLOOR=2 SWAP_HYST=1.2
  for t in XI XIF; do [[ -f "$R/$t/engine.log" ]] && note "$t engine: $(grep -aiE 'poison|Traceback' "$R/$t/engine.log" | head -1 | cut -c1-160)"; done
else
  note "gates failed: exchange arms skipped"
fi
arm SB2 $STATIC
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  note "SERVING on :8029: static broad placement N=96, MTP off, agent overlay, vision on, 256k"
else
  teardown S
fi
exit 0

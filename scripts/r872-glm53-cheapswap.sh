#!/usr/bin/env bash
# R872 (2026-10-07): cheapswap-r2 (checkpoint-free GPU<->worker expert exchange, codex + independent review, 4 High fixed)
# against R869's static placement, plus a static-placement depth ladder.
#   T   self-tests in tabbyapi:cheapswap-r2 (cpu_native, gpu fixture cuda:0/cuda:1, real-worker ring, DMA probe);
#       any failure skips the exchange arms
#   SB  static broad placement (control, r828 image)
#   XS  exchange, B-safe: exact cadence every 64 calls, global budget 64, floor 4, hyst 2.0
#   XF  exchange, B-fast: served cadence every 16, per-layer budget 64, floor 2, hyst 1.2
#   SB2 control again
#   D   static, PROFILE=1: depth 4k, 4k again, 32k, 128k (R870 on dynamic placement: 21.7-25.8 tok/s at 4-8k, 43-49 at
#       16-128k, i.e. not a context effect; this checks static is flat and the first requests after boot are not slow)
#   S   serve static + the r861 agent overlay, vision check
#   sudo systemd-run --unit=r872-glm53-cheapswap --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r872-glm53-cheapswap.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r872-tools
R=/srv/qwen5090/results/$(date +%F)-r872-glm53-cheapswap-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r872] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"  # exit relaunch
export GPU_QUEUE_NAME=r872-glm53-cheapswap
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
STATIC="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"
XBASE="OFFLOAD_N=96 GLM_IMG=tabbyapi:cheapswap-r2 PINNED_ARENA=1 SWAP_MODE=exchange"
# shellcheck disable=SC2086
arm(){ local tag=$1; shift; run_arm "$tag" "$ALL" DRAFT=0 "$@"; local rc=$?; teardown "$tag"; return $rc; }
mkdir -p "$R/T"; TOK=1
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
selftest(){ local tag=$1; shift
  timeout -k 15 900 sudo -n docker run --rm --gpus all --ipc=host "$@" > "$R/T/$tag.log" 2>&1
  local rc=$?; note "selftest $tag rc=$rc: $(tail -1 "$R/T/$tag.log" | cut -c1-160)"; ((rc == 0)) || TOK=0; }
selftest cpu_native --entrypoint python tabbyapi:cheapswap-r2 /opt/cheapswap/cpu_native_selftest.py
selftest gpu0 --entrypoint python tabbyapi:cheapswap-r2 /opt/cheapswap/gpu_selftest.py --device cuda:0 --swizzle 1
selftest gpu0_nosw --entrypoint python tabbyapi:cheapswap-r2 /opt/cheapswap/gpu_selftest.py --device cuda:0 --swizzle 0
selftest gpu1 --entrypoint python tabbyapi:cheapswap-r2 /opt/cheapswap/gpu_selftest.py --device cuda:1 --swizzle 1
selftest ring --entrypoint python tabbyapi:cheapswap-r2 /opt/cheapswap/gpu_ring_selftest.py
selftest dma --entrypoint python tabbyapi:cheapswap-r2 /opt/cheapswap/probe_dma.py --mb 6.33 --iters 256
arm SB $STATIC
if ((TOK)); then
  # shellcheck disable=SC2086
  arm XS $XBASE SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0
  # shellcheck disable=SC2086
  arm XF $XBASE SWAP_CADENCE=served SWAP_INTERVAL=16 SWAP_MAX=64 SWAP_SCOPE=layer SWAP_FLOOR=2 SWAP_HYST=1.2
  for t in XS XF; do [[ -f "$R/$t/engine.log" ]] && note "$t engine: $(grep -acE 'exchange|swap' "$R/$t/engine.log") swap/exchange log lines; $(grep -aiE 'poison|Traceback|error' "$R/$t/engine.log" | head -1 | cut -c1-160)"; done
else
  note "self-tests failed: exchange arms skipped"
fi
arm SB2 $STATIC
# depth ladder on static
D=$R/D; mkdir -p "$D"
# shellcheck disable=SC2086
if run_arm D "" DRAFT=0 $STATIC PROFILE=1; then
  timeout -k 15 5400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/depth.jsonl" \
    --phase c1 --runs 1 --kinds "" --depths 4096,4096,32768,131072 > "$D/depth.log" 2>&1 || note "D probe rc=$?"
  python3 - "$D/depth.jsonl" <<'PY2' | tee -a "$R/summary.txt"
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get('phase') == 'depth':
        print(f"D {r['prompt_tokens']//1024}k: prefill {r['engine_prefill_tps']:.0f} t/s, decode {r['decode_tps']:.1f} t/s")
PY2
  teardown D; grep -a "moe_cpu prof" "$R/D/engine.log" | tail -3 > "$R/D/prof-tail.txt"
fi
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/vision.jsonl" --phase vision \
    > "$R/S/vision.log" 2>&1 && note "vision check PASS" || note "vision check FAILED: $(grep -a Error "$R/S/vision.log" | tail -1 | cut -c1-200)"
  note "SERVING on :8029: static broad placement N=96, MTP off, agent overlay, vision on, 256k"
else
  teardown S
fi
exit 0

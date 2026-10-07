#!/usr/bin/env bash
# R877 (2026-10-07): agent overlay r3 (think tags kept literal in GLM job stops / parser / history) vs r2, on static N=96,
# plus the sturdier vision check (256 px, red/blue/green + left/right split). R871 S showed the hard tool call still
# failing on r2 (stream stopped right before a literal <think>), and R868's r2 "pass" substituted the tags.
# Think ids (154841/154842) are not in the model's EOS list (154820/154827/154829); r3's TABBY_GLM_TAG_TRACE=1 names the
# stop source and triggering id. Toolcheck: hard / think-only / close-only, 2 repeats, streamed, temperature 0 and
# default; r3 also runs the 1,000-line case once. Ends serving the better overlay (trace off).
#   sudo systemd-run --unit=r877-glm53-agent-r3 --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r877-glm53-agent-r3.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r877-tools
R=/srv/qwen5090/results/$(date +%F)-r877-glm53-agent-r3-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r877] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"  # exit relaunch
export GPU_QUEUE_NAME=r877-glm53-agent-r3
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
check(){  # tag, extra toolcheck args...
  local tag=$1; shift
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/$tag/vision.jsonl" --phase vision \
    > "$R/$tag/vision.log" 2>&1 && note "$tag vision PASS" || note "$tag vision FAILED: $(grep -a 'vision check failed' "$R/$tag/vision.log" | tail -1 | cut -c1-240)"
  timeout -k 15 5400 python3 -u "$HERE/glm53_toolcheck_r3.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/$tag/toolcheck" \
    --repeats 2 --modes stream --temperatures zero default --cases hard think-only close-only "$@" > "$R/$tag/toolcheck.log" 2>&1
  note "$tag toolcheck rc=$?: $(grep -aiE 'pass|fail|summary' "$R/$tag/toolcheck.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
}
# shellcheck disable=SC2086
if run_arm A2 "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then check A2; fi
teardown A2
# shellcheck disable=SC2086
if run_arm A3 "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r3 AGENT=1 TAG_TRACE=1; then
  check A3
  timeout -k 15 3600 python3 -u "$HERE/glm53_toolcheck_r3.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/A3/toolcheck-long" \
    --repeats 1 --modes stream --temperatures zero --cases 1000-line > "$R/A3/toolcheck-long.log" 2>&1
  note "A3 1000-line rc=$?: $(tail -2 "$R/A3/toolcheck-long.log" | tr '\n' ' ' | cut -c1-300)"
fi
teardown A3
grep -a 'GLM-TAG-R3' "$R/A3/engine.log" 2>/dev/null | grep -aiE 'eos|stop|trigger' | tail -20 > "$R/A3/tag-trace-tail.txt"
note "A3 tag-trace lines: $(grep -ac 'GLM-TAG-R3' "$R/A3/engine.log" 2>/dev/null)"
# serve: r3 (trace off) unless its toolcheck errored out
IMG=tabbyapi:r861-glm-agent-r3
grep -aq 'A3 toolcheck rc=0' "$R/summary.txt" || IMG=tabbyapi:r861-glm-agent-r2
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $STATIC GLM_IMG=$IMG AGENT=1; then
  note "SERVING on :8029: static N=96, MTP off, agent overlay $IMG, vision on, 256k"
else
  teardown S
fi
exit 0

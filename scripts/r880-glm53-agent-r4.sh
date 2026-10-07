#!/usr/bin/env bash
# R880 (2026-10-07): agent overlay r4 vs r2 (served). R877: r3 raised "Tokenizer encoded a literal GLM tag as a control
# ID" on every prompt quoting a tag (nothing generated); r2 lost the turn at temperature 0 (stop after ~26 reasoning
# tokens at a literal </fake>) and its default-temperature write_file bodies were not byte-exact. r4 (codex
# round, docker/glm-agent-r4) encodes literal tags in message
# text as ordinary pieces, carries real reasoning control ids separately, and keeps tool-argument bytes exact.
# Checker: glm53_toolcheck_r4.py (UTF-8 exact-body equality, LF instruction), the same on both arms:
# hard / think-only / close-only / 1000-line, streamed, temperature 0 and default, 2 repeats. Plus the 256 px vision
# check, r2's original-prompt plain probe (eos_reason / stop_str), r4's in-container literal tokenization probe.
# Serves r4 (trace off) only if its whole matrix and vision pass; otherwise r2.
#   sudo systemd-run --unit=r880-glm53-agent-r4 --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r880-glm53-agent-r4.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r880-tools
PK=/srv/qwen5090/glm-agent-r4-build   # copy of flan/patches/tabbyapi/glm-agent-r4
R=/srv/qwen5090/results/$(date +%F)-r880-glm53-agent-r4-$(date +%H%M%S)
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
export GPU_QUEUE_NAME=r880-glm53-agent-r4
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
( cd "$PK/r4" && python3 -B tests/check_hashes.py SHA256SUMS . && bash build.sh ) > "$R/build-r4.log" 2>&1
BUILD_RC=$?; note "build r4 rc=$BUILD_RC: $(tail -1 "$R/build-r4.log" | cut -c1-120)"
((BUILD_RC == 0)) || { note "r4 build FAILED: A2 only, keep r2"; }
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
check(){  # tag
  local tag=$1
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/$tag/vision.jsonl" --phase vision \
    > "$R/$tag/vision.log" 2>&1 && note "$tag vision PASS" || note "$tag vision FAILED: $(grep -a 'vision check failed' "$R/$tag/vision.log" | tail -1 | cut -c1-240)"
  timeout -k 15 9000 python3 -u "$HERE/glm53_toolcheck_r4.py" --url http://127.0.0.1:8029/v1 --model "$MODEL" --out "$R/$tag/toolcheck" \
    --cases hard think-only close-only 1000-line --modes stream --temperatures zero default --repeats 2 \
    --max-tokens 12000 --long-max-tokens 40000 > "$R/$tag/toolcheck.log" 2>&1
  local rc=$?
  note "$tag toolcheck rc=$rc: $(grep -a '^RESULT' "$R/$tag/toolcheck.log" | python3 -c 'import json,sys
rs=[json.loads(l[7:]) for l in sys.stdin]; print(sum(r.get("passed") is True for r in rs), "of", len(rs), "passed; failed:", " ".join(r["tag"] for r in rs if not r.get("passed")))' | cut -c1-400)"
}
# shellcheck disable=SC2086
if run_arm A2 "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  check A2
  timeout -k 15 3600 python3 -u "$HERE/glm53_toolcheck_r3.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/A2/original-plain-probe" \
    --cases hard think-only close-only --modes plain --temperatures zero --repeats 2 > "$R/A2/original-plain-probe.log" 2>&1
  note "A2 original-prompt plain probe rc=$?: $(grep -a '^RESULT' "$R/A2/original-plain-probe.log" | python3 -c 'import json,sys
for l in sys.stdin:
    r=json.loads(l[7:]); print(r["tag"], r.get("finish"), "eos_reason", r.get("eos_reason"), "stop_str", repr(r.get("stop_str")))' | tr '\n' ';' | cut -c1-500)"
fi
teardown A2
# shellcheck disable=SC2086
if ((BUILD_RC == 0)) && run_arm A4 "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r4 AGENT=1 TAG_TRACE=1; then
  check A4
  sudo -n docker cp "$PK/r4/tests/probe_literals.py" glm53:/tmp/glm-probe-literals.py \
    && sudo -n docker exec glm53 python3 /tmp/glm-probe-literals.py "/models/$MODEL" --app /app --out /tmp/glm-literals.json > "$R/A4/literal-probe.log" 2>&1 \
    && sudo -n docker cp glm53:/tmp/glm-literals.json "$R/A4/literal-probe.json"
  note "A4 literal probe rc=$?: $(tail -2 "$R/A4/literal-probe.log" | tr '\n' ' ' | cut -c1-240)"
fi
teardown A4
IMG=tabbyapi:r861-glm-agent-r2
grep -aq '^A4 toolcheck rc=0' "$R/summary.txt" && grep -aq '^A4 vision PASS' "$R/summary.txt" && IMG=tabbyapi:r861-glm-agent-r4
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $STATIC GLM_IMG=$IMG AGENT=1; then
  note "SERVING on :8029: static N=96, MTP off, agent overlay $IMG, vision on, 256k"
else
  teardown S
fi
exit 0

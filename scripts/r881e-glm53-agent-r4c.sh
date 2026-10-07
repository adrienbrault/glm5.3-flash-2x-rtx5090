#!/usr/bin/env bash
# R881e (2026-10-08): R881d with r4c = r4b + the missing `import os` (r4b died on every request: NameError in
# _encode_prompt, TABBY_GLM_CACHE_VERIFY check). Builds r4c first.
# R881d (2026-10-08): FAST r4b-only check (user: "how could this take 1h30?"). R881b ran r4 AND r4b serially (~3 h of
# GPU; r4 was already measured in R880). Here: r4b only, the 16 cases + 2 more 1000-line zero repeats spread over 4
# concurrent checker processes (MAX_BATCH 4: ~110 tok/s aggregate vs ~57 serial), ~30 min.
# Earlier R881b header: agent overlay r4b vs r4 through OUR launcher (R881 ran codex's operator-run.sh, whose own docker run
# config died at load: "Insufficient VRAM in split for model and cache", and its readiness loop would have waited 30 min).
# r4b (patches/tabbyapi/glm-agent-r4b): TABBY_GLM_LITERAL_ENCODING=runs (image default), TABBY_GLM_CACHE_VERIFY=0.
# Arms A4 (r4) and A4B (r4b), both TAG_TRACE=1, static N=96, MTP off: vision, 16-case r4 checker matrix, and a 3-repeat
# 1000-line run at both temperatures (cache-reuse probe). No promotion: ends serving the current daily (R882b's choice).
#   sudo systemd-run --unit=r881e-glm53-agent-r4c --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r881e-glm53-agent-r4c.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r881e-tools
PK=/srv/qwen5090/glm-agent-r4-build   # copy of flan/patches/tabbyapi/glm-agent-r4
R=/srv/qwen5090/results/$(date +%F)-r881e-glm53-agent-r4c-$(date +%H%M%S)
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
export GPU_QUEUE_NAME=r881e-glm53-agent-r4c
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
( sudo -n docker build -t tabbyapi:r861-glm-agent-r4c /srv/qwen5090/glm-agent-r4c-build ) > "$R/build.log" 2>&1
BUILD_RC=$?; note "build r4c rc=$BUILD_RC"; ((BUILD_RC == 0)) || exit 1
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
repeat(){  # tag
  timeout -k 15 9000 python3 -u "$HERE/glm53_toolcheck_r4.py" --url http://127.0.0.1:8029/v1 --model "$MODEL" --out "$R/$1/cache-repeat" \
    --cases 1000-line --modes stream --temperatures zero default --repeats 3 --max-tokens 12000 --long-max-tokens 40000 \
    > "$R/$1/cache-repeat.log" 2>&1
  note "$1 cache-repeat rc=$?: $(grep -a '^RESULT' "$R/$1/cache-repeat.log" | python3 -c 'import json,sys
rs=[json.loads(l[7:]) for l in sys.stdin]; print(sum(r.get("passed") is True for r in rs), "of", len(rs), "passed; failed:", " ".join(r["tag"] for r in rs if not r.get("passed")))' | cut -c1-300)"
}
tc(){  # out-subdir case temps repeats
  timeout -k 15 5400 python3 -u "$HERE/glm53_toolcheck_r4.py" --url http://127.0.0.1:8029/v1 --model "$MODEL" --out "$R/A4B/$1" \
    --cases "$2" --modes stream --temperatures $3 --repeats "$4" --max-tokens 12000 --long-max-tokens 40000 > "$R/A4B/$1.log" 2>&1
}
# shellcheck disable=SC2086
if run_arm A4B "" DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r4c AGENT=1 TAG_TRACE=1; then
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/A4B/vision.jsonl" --phase vision \
    > "$R/A4B/vision.log" 2>&1 && note "A4B vision PASS" || note "A4B vision FAILED: $(grep -a 'vision check failed' "$R/A4B/vision.log" | tail -1 | cut -c1-200)"
  ( tc p1-long-zero 1000-line zero 1; tc p1-hard-zero hard zero 2; tc p1-hard-default hard default 1 ) &
  ( tc p2-long-zero 1000-line zero 1; tc p2-hard-default hard default 1; tc p2-think-zero think-only zero 2 ) &
  ( tc p3-long-zero 1000-line zero 1; tc p3-think-default think-only default 2; tc p3-close-zero close-only zero 1 ) &
  ( tc p4-long-default 1000-line default 1; tc p4-close-zero close-only zero 1; tc p4-close-default close-only default 2 ) &
  wait
  cat "$R"/A4B/p*.log | grep -a '^RESULT' > "$R/A4B/all-results.txt"
  note "A4B r4b: $(python3 -c 'import json,sys
rs=[json.loads(l[7:]) for l in open(sys.argv[1])]
bad=[r["tag"]+":"+str((((r.get("calls") or [{}])[0]).get("byte_diff") or {}).get("offset_zero_based")) for r in rs if not r.get("passed")]
print(sum(r.get("passed") is True for r in rs), "of", len(rs), "passed; lost turns", sum(1 for r in rs if r.get("finish")!="tool_calls"), "; failed:", " ".join(bad))' "$R/A4B/all-results.txt" | cut -c1-400)"
fi
teardown A4B
DAILY="DRAFT=0 $STATIC GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1"
# the newest swap-family daily (R882c's, or R883's MTP promotion if it ran first)
L=""; for f in $(ls -t /srv/qwen5090/results/*r88[23]*-glm53-swap-*/summary.txt 2>/dev/null); do L=$(grep -h '^SERVING on :8029: ' "$f" | tail -1); [[ -n "$L" ]] && break; done
[[ -n "$L" ]] && DAILY="DRAFT=0 $(sed -E 's/.*\((.*)\)$/\1/' <<<"$L")"
note "restoring daily: $DAILY"
# shellcheck disable=SC2086
run_arm S "" $DAILY && note "SERVING on :8029: $DAILY" || teardown S
exit 0

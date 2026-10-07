#!/usr/bin/env bash
# R884 (2026-10-08): serve the daily with KEEP_THINKING=1 (templates/glm53-keep-thinking.jinja: clear_thinking defaults to
# false, so earlier turns keep their reasoning and a new user message no longer rewrites the previous tool loop; found by
# the TensorFold survey, user: "lets default to clear_thinking=false"). Waits for R883/R881e to finish so their restores
# can't undo it. Boots the newest daily env with the new launcher, then glm53_keepthink_probe.py (keep vs clear on the
# same boot: cached tokens of the follow-up request), vision, simple toolcheck. Keeps serving if keep caches >= 90 % of
# the first request and checks pass; else serves the same env with KEEP_THINKING=0.
#   sudo systemd-run --unit=r884-glm53-keepthink --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r884-glm53-keepthink.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r884-tools
R=/srv/qwen5090/results/$(date +%F)-r884-glm53-keepthink-$(date +%H%M%S)
while systemctl is-active --quiet r883-glm53-swap-mtp || systemctl is-active --quiet r881e-glm53-agent-r4c; do sleep 30; done
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r884] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"  # exit relaunch
export GPU_QUEUE_NAME=r884-glm53-keepthink
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
L=""; for f in $(ls -t /srv/qwen5090/results/*r88[1-3]*/summary.txt 2>/dev/null); do L=$(grep -h '^SERVING on :8029: ' "$f" | tail -1); [[ -n "$L" ]] && break; done
if [[ "$L" == *"("*")" ]]; then DENV=$(sed -E 's/.*\((.*)\)$/\1/' <<<"$L"); else DENV=${L#SERVING on :8029: }; fi
[[ -n "$DENV" ]] || DENV="OFFLOAD_N=96 GLM_IMG=tabbyapi:cheapswap-r3-agent-r2 PINNED_ARENA=1 SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_INIT_STATS=$HERE/split-stats-broad-r869.json SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0 AGENT=1"
DENV=$(sed -E 's#/srv/qwen5090/r[0-9a-z]+-tools/#'"$HERE"'/#g; s/(^| )KEEP_THINKING=[01]//g' <<<"$DENV")
[[ "$DENV" == *DRAFT=* ]] || DENV="DRAFT=0 $DENV"
BASE="$DENV KEEP_THINKING=0"   # exit relaunch fallback
note "daily env: $DENV"
# shellcheck disable=SC2086
if run_arm K "" $DENV KEEP_THINKING=1; then
  grep -q 'prompt_template: glm53-keep-thinking' "$R/K/config.yml" && note "K config has prompt_template glm53-keep-thinking" || note "K config MISSING prompt_template"
  timeout -k 15 1800 python3 -u "$HERE/glm53_keepthink_probe.py" --model "$MODEL" --out "$R/K/keepthink.json" > "$R/K/keepthink.log" 2>&1
  note "K probe rc=$?:"; sed 's/^/  /' "$R/K/keepthink.log" | tail -4 | tee -a "$R/summary.txt"
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/K/vision.jsonl" --phase vision > "$R/K/vision.log" 2>&1; V=$?
  timeout -k 15 1800 python3 -u "$HERE/glm53_toolcheck_r3.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/K/toolcheck" \
    --repeats 2 --modes stream --temperatures zero default --cases simple > "$R/K/toolcheck.log" 2>&1; T=$?
  OK=$(python3 -c 'import json,sys
rows={r["mode"]:r for r in json.load(open(sys.argv[1]))}
ok=all((rows[m]["B"]["cached_tokens"] or 0) >= 0.9*(rows[m]["A"]["prompt_tokens"] or 1e9) for m in ("keep","keep2"))
print(1 if ok else 0)' "$R/K/keepthink.json" 2>/dev/null || echo 0)
  note "K vision rc=$V toolcheck simple rc=$T keep-cache-ok=$OK"
  if ((V == 0 && T == 0 && OK == 1)); then note "SERVING on :8029: keep-thinking template ($DENV KEEP_THINKING=1)"; exit 0; fi
fi
teardown K
# shellcheck disable=SC2086
run_arm S "" $DENV KEEP_THINKING=0 && note "SERVING on :8029: stock template ($DENV KEEP_THINKING=0)" || teardown S
exit 0

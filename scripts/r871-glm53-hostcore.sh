#!/usr/bin/env bash
# R871 (2026-10-07): reserve a host core for the TabbyAPI host process (emulating exllamav3 dev fa822cf, upstream +15 %
# decode without MTP on a 7900X+4090), on top of R869's static placement (split-stats-broad-r869.json, out-of-sample
# 57.5 tok/s vs ~48.7 dynamic). All arms: N=96 MTP off, static placement, 256k, five kinds x2.
#   t8    8 worker threads, host unconfined (control)
#   t7h   7 worker threads, host confined to CPUs 7,15 (HOST_CONFINE=1)
#   t7    7 worker threads, host unconfined
#   t8s   8 worker threads, host confined to the SMT siblings 8-15
#   t8b   control again
#   S     serve the best with the r861 agent overlay (AGENT=1; R868: it fixes tool calls lost on a literal </think>)
#   sudo systemd-run --unit=r871-glm53-hostcore --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r871-glm53-hostcore.sh
# (shared harness from R869 follows)
# R869 (2026-10-07): does static expert placement generalize? R866's static counts were fitted on the probe prompts
# themselves: 63.2 tok/s in-sample, while the held-out "edit" kind gained nothing. Here the counts come ONLY from a
# broad workload that shares no text with the probe (glm53_workload.py: 12 agent tasks with tools + 20 plain prompts),
# and all five probe kinds are scored out of sample.
#   trace  route-trace image, N=96 dynamic, MTP off: run the workload, graceful stop, build split-stats-broad.json
#   SB     static placement from split-stats-broad.json, N=96           five kinds x2
#   A      dynamic placement (served default), N=96                       five kinds x2
#   S      serve SB if it beats A by > 3 %, else A (plain image); vision check
#   sudo systemd-run --unit=r869-glm53-hotset --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r869-glm53-hotset.sh
# (shared harness from R866 follows)
# R866 (2026-10-07): CPU-split expert placement on the GPU. R860b: the served dynamic placement leaves the CPU share of
# expert picks at 0.347 (uniform 0.361); its sweep floor (8x mean) and global budget (64) stop it on 288 experts.
# sim_placement.py: interval 64 / floor 4 -> 0.288 at 186 swaps per 1k steps; a static counts file cross-validates at
# ~0.29. Same N as R865's best (default 96), MTP off, 256k, vision on, five kinds x2. The counts file is fitted on the
# R860b probe prompts (code/prose/chat/html), so those kinds are in-sample; "edit" is the held-out kind.
#
#   A    dynamic, engine defaults (served)
#   B    dynamic, SWAP_INTERVAL=64 SWAP_FLOOR=4
#   C    static, SPLIT_STATS=split-stats-r860b.json
#   A2   dynamic defaults again (drift)
#   S    serve the winner on the plain image (the r861 overlay failed its live tool-call probe in R864); vision check
#
#   sudo systemd-run --unit=r866-glm53-placement --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r866-glm53-placement.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r871-tools
R=/srv/qwen5090/results/$(date +%F)-r871-glm53-hostcore-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r871] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=""; PIN=""
STATS=$HERE/split-stats-r860b.json
# R865's winner ("best: <tag> (<env>)"), else N=96 on the default split.
BASE="OFFLOAD_N=96 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json"
export GPU_QUEUE_NAME=r871-glm53-hostcore
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
declare -A ENV_OF
# shellcheck disable=SC2086
arm(){ local tag=$1; shift; if run_arm "$tag" "$ALL" DRAFT=0 $BASE "$@"; then ENV_OF[$tag]="$*"; teardown "$tag"; return 0; fi; teardown "$tag"; return 1; }
arm t8 CPU_THREADS=8
arm t7h CPU_THREADS=7 HOST_CONFINE=1
arm t7 CPU_THREADS=7
arm t8s CPU_THREADS=8 HOST_CONFINE=1
arm t8b CPU_THREADS=8
BEST=$(python3 "$HERE/r860_score.py" --best "$R" "${!ENV_OF[@]}")
BEST_ENV=${ENV_OF[${BEST:-t8}]:-CPU_THREADS=8}
note "best: ${BEST:-none} ($BEST_ENV)"
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $BASE $BEST_ENV GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/vision.jsonl" --phase vision \
    > "$R/S/vision.log" 2>&1 && note "vision check PASS" || note "vision check FAILED: $(grep -a Error "$R/S/vision.log" | tail -1 | cut -c1-200)"
  timeout -k 15 1800 python3 -u "$HERE/glm53_toolcheck.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/toolcheck" > "$R/S/toolcheck.log" 2>&1
  grep -a '^RESULT' "$R/S/toolcheck.log" | python3 -c 'import json,sys
for l in sys.stdin:
    r=json.loads(l[7:]); print("toolcheck", r["tag"], r["finish"], "args", r["args_len"], "json", r["args_json_ok"])' | tee -a "$R/summary.txt"
  note "SERVING on :8029: MTP off, $BASE $BEST_ENV, agent overlay, vision on, 256k"
else
  teardown S
fi
exit 0

#!/usr/bin/env bash
# R864 (2026-10-07, user: "optimize the shit out of this", "rapid iteration", "Need vision"): GLM-5.3-Flash c1 levers
# after R860, one boot per arm, 256k cache, vision on. Runs the repo's flan/r858 tooling from /srv/qwen5090/r858-next
# so R860/R863, which execute /srv/qwen5090/r858, never see a file change under them.
#
# R860 found: MTP depth 1 = 45.7, MTP off = 46.2 (the verify row reads ~2x CPU experts), depth 2/3 no boot at N=104.
# MTP off frees ~2.35 GB of VRAM, about 8 experts per layer (42 x 6.33 MB each), so N (experts per layer computed on
# the CPU) can drop. Fewer CPU experts per token is the lever; depth 2/3 are skipped (3 rows for <= 2.2 tokens).
#
#   A   control: MTP off, N=104, threads from R860's winner        code/prose/chat/html/edit x2
#   B   MTP off, N ladder 96 -> 100 (first that boots)              x2
#   C   MTP off, N ladder 88 -> 92, only if B booted at 96          x2
#   D   MTP depth 1 + prompt lookup, N=104, edit + code             x2   (vs E, same boot shape without lookup)
#   E   MTP depth 1, N=104, edit + code                              x2
#   A2  control again (boot-to-boot drift)                          x2
#   S   serve the best of A/B/C/A2 with the r861 agent overlay (AGENT=1), then the vision check and the live
#       tool-streaming probe, direct and through Olla.
#
#   sudo systemd-run --unit=r864-glm53-levers --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r864-glm53-levers.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r858-next
R=/srv/qwen5090/results/$(date +%F)-r864-glm53-levers-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r864] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
AGENT_PKG=/srv/qwen5090/r861-glm-agent-r2
cp -R "$HERE" "$R/build"
# Threads from R860's phase-3 winner ("overall winner: <tag> (<env>)"); default 8.
THREADS=$(grep -h "overall winner" /srv/qwen5090/results/*r860-glm53-chain*/summary.txt 2>/dev/null | tail -1 | grep -oE 'CPU_THREADS=[0-9]+' | tail -1)
THREADS=${THREADS:-CPU_THREADS=8}
PIN=$(grep -h "overall winner" /srv/qwen5090/results/*r860-glm53-chain*/summary.txt 2>/dev/null | tail -1 | grep -oE 'PINNED_ARENA=1' || true)

export GPU_QUEUE_NAME=r864-glm53-levers
. /srv/qwen5090/lib/gpu-queue.sh
BEST_ENV=""
cleanup(){
  local rc=$?
  trap - EXIT INT TERM HUP
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
    note "no GLM serving at exit: relaunching ${BEST_ENV:-the R860 base} (no overlay)"
    # shellcheck disable=SC2086
    "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split OFFLOAD_N=104 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 \
      SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$R/exit-serve" DRAFT=0 $THREADS $PIN ${BEST_ENV} \
      bash /srv/qwen5090/r858/launch-glm53.sh > "$R/exit-serve.log" 2>&1 || note "exit relaunch FAILED (exit-serve.log)"
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  echo "exit=$rc" > "$R/last.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'note interrupted; exit 130' INT
trap 'note terminated; exit 143' TERM HUP
gpu_lock
note "GPU lock held; results $R; base $THREADS $PIN"
sudo -n docker logs glm53 > "$R/glm-before.log" 2>&1 || true

run_arm(){  # tag, probe kinds, then KEY=VALUE launcher env
  local tag=$1 kinds=$2; shift 2
  local D=$R/$tag rc=0
  mkdir -p "$D"
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  # shellcheck disable=SC2086
  timeout -k 20 1500 "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 \
    CHUNK=2048 SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$D" $THREADS $PIN "$@" \
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

run_arm A "$ALL" DRAFT=0 OFFLOAD_N=104; teardown A; ENV_OF[A]="OFFLOAD_N=104"
NB=""
for n in 96 100; do
  if run_arm "B-n$n" "$ALL" DRAFT=0 OFFLOAD_N=$n; then NB=$n; teardown "B-n$n"; ENV_OF[B-n$n]="OFFLOAD_N=$n"; break; fi
  teardown "B-n$n"
done
if [[ "$NB" == 96 ]]; then
  for n in 88 92; do
    if run_arm "C-n$n" "$ALL" DRAFT=0 OFFLOAD_N=$n; then teardown "C-n$n"; ENV_OF[C-n$n]="OFFLOAD_N=$n"; break; fi
    teardown "C-n$n"
  done
fi
run_arm D edit,code DRAFT=1 DRAFT_N=1 PLOOKUP=1 OFFLOAD_N=104; teardown D
run_arm E edit,code DRAFT=1 DRAFT_N=1 OFFLOAD_N=104; teardown E
run_arm A2 "$ALL" DRAFT=0 OFFLOAD_N=104; teardown A2; ENV_OF[A2]="OFFLOAD_N=104"

BEST=$(python3 "$HERE/r860_score.py" --best "$R" "${!ENV_OF[@]}")
BEST_ENV=${ENV_OF[${BEST:-A}]:-OFFLOAD_N=104}
note "best MTP-off arm: ${BEST:-none} ($BEST_ENV)"

# ---- S: serve with the agent overlay; vision + live tool-streaming checks ----
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $BEST_ENV GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/vision.jsonl" --phase vision \
    > "$R/S/vision.log" 2>&1 && note "vision check PASS" || note "vision check FAILED: $(grep -a Error "$R/S/vision.log" | tail -1 | cut -c1-200)"
  for via in direct olla; do
    url=http://127.0.0.1:8029; [[ $via == olla ]] && url=http://127.0.0.1:40114/olla/proxy
    TABBY_BASE_URL=$url TABBY_MODEL=$MODEL timeout -k 10 1800 python3 -u "$AGENT_PKG/tests/probe_live.py" --expect-live \
      --request-out "$R/S/live-$via.request.json" --trace-out "$R/S/live-$via.trace.json" > "$R/S/live-$via.log" 2>&1 \
      && note "live tool-streaming probe ($via) PASS" || note "live tool-streaming probe ($via) FAILED: $(tail -2 "$R/S/live-$via.log" | tr '\n' ' ' | cut -c1-200)"
  done
  note "SERVING on :8029: MTP off, $BEST_ENV, agent overlay, vision on, 256k"
else
  teardown S
  note "S (agent overlay) did not boot; cleanup relaunches the plain image"
fi
exit 0

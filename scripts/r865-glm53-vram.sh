#!/usr/bin/env bash
# R865 (2026-10-07): free VRAM for fewer CPU experts. R864: N=96 MTP off = 48.4 tok/s (+4.8 % over N=104); N=92 and
# N=88 did not fit at GPU_SPLIT 31,31, while card 1 kept ~2 GiB unused. Context stays 256k and vision stays on
# (user). Levers: an uneven split (31,31.8), then a smaller prefill chunk (1024) and batch (2).
#
#   ctl  N=96, split 31,31 (R864 B-n96 again: drift)                       x2 five kinds
#   s92  N=92, split 31,31.8                                                x2
#   c92  N=92, split 31,31.8, CHUNK=1024, MAX_BATCH=2   (if s92 fails)      x2
#   x88  N=88 with whichever of s92/c92 booted (+ chunk/batch)              x2
#   pf   32k prefill + decode depth probe on the winner, because CHUNK moves prefill
#   S    serve the winner with the r861 agent overlay; vision check
#
#   sudo systemd-run --unit=r865-glm53-vram --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r865-glm53-vram.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r865-tools
R=/srv/qwen5090/results/$(date +%F)-r865-glm53-vram-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r865] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
THREADS=CPU_THREADS=8; PIN=""
export GPU_QUEUE_NAME=r865-glm53-vram
. /srv/qwen5090/lib/gpu-queue.sh
BEST_ENV="OFFLOAD_N=96"
cleanup(){
  local rc=$?
  trap - EXIT INT TERM HUP
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
    note "no GLM serving at exit: relaunching $BEST_ENV (no overlay)"
    # shellcheck disable=SC2086
    "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 \
      SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$R/exit-serve" DRAFT=0 $THREADS $BEST_ENV \
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
note "GPU lock held; results $R"
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
arm(){ local tag=$1; shift; if run_arm "$tag" "$ALL" DRAFT=0 "$@"; then ENV_OF[$tag]="$*"; teardown "$tag"; return 0; fi; teardown "$tag"; return 1; }

arm ctl OFFLOAD_N=96
X=""
if arm s92 OFFLOAD_N=92 GPU_SPLIT=31,31.8; then X="GPU_SPLIT=31,31.8"
elif arm c92 OFFLOAD_N=92 GPU_SPLIT=31,31.8 CHUNK=1024 MAX_BATCH=2; then X="GPU_SPLIT=31,31.8 CHUNK=1024 MAX_BATCH=2"; fi
if [[ -n "$X" ]]; then
  # shellcheck disable=SC2086
  arm x88 OFFLOAD_N=88 $X || { [[ "$X" == GPU_SPLIT=31,31.8 ]] && arm x88c OFFLOAD_N=88 GPU_SPLIT=31,31.8 CHUNK=1024 MAX_BATCH=2; }
fi
BEST=$(python3 "$HERE/r860_score.py" --best "$R" "${!ENV_OF[@]}")
BEST_ENV=${ENV_OF[${BEST:-ctl}]:-OFFLOAD_N=96}
note "best: ${BEST:-none} ($BEST_ENV)"

# ---- pf + S: serve the winner with the agent overlay; prefill depth check; vision check ----
# shellcheck disable=SC2086
if run_arm S "" DRAFT=0 $BEST_ENV GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1; then
  timeout -k 15 1800 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/depth.jsonl" \
    --phase c1 --runs 1 --kinds "" --depths 32768 > "$R/S/depth.log" 2>&1
  python3 "$HERE/r860_score.py" "$R/S/depth.jsonl" S-depth | tee -a "$R/summary.txt"
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/S/vision.jsonl" --phase vision \
    > "$R/S/vision.log" 2>&1 && note "vision check PASS" || note "vision check FAILED: $(grep -a Error "$R/S/vision.log" | tail -1 | cut -c1-200)"
  note "SERVING on :8029: MTP off, $BEST_ENV, agent overlay, vision on, 256k"
else
  teardown S
  note "S (agent overlay) did not boot; cleanup relaunches the plain image"
fi
exit 0

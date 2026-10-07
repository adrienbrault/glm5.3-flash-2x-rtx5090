#!/usr/bin/env bash
# R860b (2026-10-07): R860's predict/profile arm, rerun with dynamic placement ON. R860 ran it with CPU_SWAP=0, which
# disables dynamic expert placement; the launcher's verify step then found no dynamic split registrations and
# rejected the boot (predict NO BOOT, 0 trace files). The served config uses dynamic placement, so the trace and
# the CPU/handoff profilers belong there anyway: MTP off, N=104, 256k, 8 threads, c1 x1 four kinds + 32k depth,
# graceful stop so the trace writer drains. Feeds build_hotset.py, prefetch-r2 calibrate/replay, and the
# CPU-bytes-per-token analysis.
#   sudo systemd-run --unit=r860b-glm53-trace --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r860b-glm53-trace.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r858
R=/srv/qwen5090/results/$(date +%F)-r860b-glm53-trace-$(date +%H%M%S); mkdir -p "$R/predict"
exec > >(tee -a "$R/transcript.log") 2>&1
note(){ printf '%s [r860b] %s\n' "$(date -Iseconds)" "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
export GPU_QUEUE_NAME=r860b-glm53-trace
. /srv/qwen5090/lib/gpu-queue.sh
cleanup(){
  local rc=$?; trap - EXIT INT TERM HUP
  sudo -n docker stop -t 120 glm53 >/dev/null 2>&1 || true
  sudo -n docker logs glm53 > "$R/predict/engine.log" 2>&1 || true
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  note "trace files: $(find "$R/predict/route-traces" -type f 2>/dev/null | wc -l), $(du -sh "$R/predict/route-traces" 2>/dev/null | cut -f1)"
  note "restoring the base GLM config"
  "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split OFFLOAD_N=104 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 \
    SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 DRAFT=0 CPU_THREADS=8 BOOT_TIMEOUT=1200 RUN_DIR="$R/restore" \
    bash "$HERE/launch-glm53.sh" > "$R/restore.log" 2>&1 || note "restore FAILED"
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; echo "exit=$rc" > "$R/last.txt"; exit "$rc"
}
trap cleanup EXIT
trap 'note terminated; exit 143' TERM HUP INT
gpu_lock
note "GPU lock held; results $R"
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
"${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split OFFLOAD_N=104 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 \
  SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$R/predict" DRAFT=0 CPU_THREADS=8 \
  GLM_IMG=tabbyapi:r859-predict-trace-r2 PROFILE=1 PREDICT=1 CPU_SWAP=1 ROUTE_TRACE_DIR="$R/predict/route-traces" \
  bash "$HERE/launch-glm53-predict-r2.sh" > "$R/predict/boot.log" 2>&1 || { note "NO BOOT: $(tail -2 "$R/predict/boot.log" | tr '\n' ' ' | cut -c1-240)"; exit 1; }
timeout -k 15 3600 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model glm53-flash-exl3-2.05bpw-turboderp \
  --out "$R/predict/c1.jsonl" --phase c1 --runs 1 --kinds code,prose,chat,html --depths 32768 > "$R/predict/c1.log" 2>&1
python3 "$HERE/r860_score.py" "$R/predict/c1.jsonl" predict | tee -a "$R/summary.txt"
exit 0

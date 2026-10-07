#!/usr/bin/env bash
# R868 (2026-10-07): why did the r861 agent overlay's live probe end a write_file call at 67 argument characters with
# finish_reason "stop" (R864)? Boot tabbyapi:r861-glm-agent-r2 at N=96 MTP off twice, overlay flags OFF (AGENT=0: the
# patch is installed but every switch is off, which should be byte-identical to the parent image) and ON (AGENT=1),
# and run glm53_toolcheck.py (simple streamed, simple non-streamed, hard streamed with literal tags) on each. Raw SSE
# is kept. The comparison separates model behavior from overlay behavior. Ends serving the plain image.
#   sudo systemd-run --unit=r868-glm53-agent-debug --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r868-glm53-agent-debug.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r868-tools
R=/srv/qwen5090/results/$(date +%F)-r868-glm53-agent-debug-$(date +%H%M%S); mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
note(){ printf '%s [r868] %s\n' "$(date -Iseconds)" "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
COMMON=(PACK=A OFFLOAD_MODE=split OFFLOAD_N=96 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 SYSMEM_RC_MB=1024
        GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 DRAFT=0 CPU_THREADS=8)
export GPU_QUEUE_NAME=r868-glm53-agent-debug
. /srv/qwen5090/lib/gpu-queue.sh
cleanup(){
  local rc=$?; trap - EXIT INT TERM HUP
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  "${CLEAN[@]}" HOME="$HOME" "${COMMON[@]}" RUN_DIR="$R/serve" bash "$HERE/launch-glm53.sh" > "$R/serve.log" 2>&1 \
    && note "SERVING on :8029: N=96 MTP off, plain image" || note "plain serve FAILED (serve.log)"
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; echo "exit=$rc" > "$R/last.txt"; exit "$rc"
}
trap cleanup EXIT
trap 'note terminated; exit 143' TERM HUP INT
gpu_lock
note "GPU lock held; results $R"
for agent in 0 1; do
  D=$R/agent$agent; mkdir -p "$D"
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  if "${CLEAN[@]}" HOME="$HOME" "${COMMON[@]}" RUN_DIR="$D" GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=$agent \
       bash "$HERE/launch-glm53.sh" > "$D/boot.log" 2>&1; then
    timeout -k 15 2400 python3 -u "$HERE/glm53_toolcheck.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D" \
      > "$D/toolcheck.log" 2>&1
    grep -a '^RESULT' "$D/toolcheck.log" | python3 -c '
import json,sys
for l in sys.stdin:
    r=json.loads(l[7:]); print(f"agent'$agent' {r[\"tag\"]}: finish={r[\"finish\"]} tool={r[\"tool\"]} args={r[\"args_len\"]} json_ok={r[\"args_json_ok\"]} body={r[\"body_len\"]} content={r[\"content_len\"]} reasoning={r[\"reasoning_len\"]} tokens={r[\"completion_tokens\"]}")' | tee -a "$R/summary.txt"
    sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  else
    note "agent$agent NO BOOT: $(tail -2 "$D/boot.log" | tr '\n' ' ' | cut -c1-200)"
  fi
done
exit 0

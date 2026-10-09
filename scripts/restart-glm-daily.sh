#!/usr/bin/env bash
# Relaunch the GLM-5.3-Flash daily on :8029 from /srv/qwen5090/glm-daily.env with the installed tools (2026-10-09).
# Same launch environment as glm_arms.sh boot(); the launcher takes the GPU-exclusive lock and replaces the container.
# Run after install-glm-daily.sh:  ssh flan 'bash -s' < flan/r858/restart-glm-daily.sh
set -euo pipefail
export HOME=${HOME:-/root}
T=/srv/qwen5090/glm-daily-tools
DAILY=$(cat /srv/qwen5090/glm-daily.env)
D=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-glm-daily-restart
mkdir -p "$(dirname "$D")"
# shellcheck disable=SC2086
timeout -k 20 1500 env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME="$HOME" \
  PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 MAX_BATCH=4 SYSMEM_RC_MB=1024 \
  GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$D" $DAILY bash "$T/launch-glm53.sh" > "$D.log" 2>&1
echo "restarted: $D"
tail -3 "$D.log"

#!/usr/bin/env bash
# Derived from LIVE config/preflight/run/health/model guards and clocks; GLM policy is separate.
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMG=${GLM_IMG:-tabbyapi:r828-prompt-lookup-r3}
# R859 diagnostic knobs: PROFILE=1 sets EXL3_MOE_CPU_PROF=1; ROUTE_TRACE_DIR (host dir, needs the route-trace image)
# mounts at /app/route-traces and sets EXL3_ROUTE_TRACE; PINNED_ARENA=1 sets EXL3_MOE_PINNED_ARENA=1.
case "$IMG" in tabbyapi:r828-prompt-lookup-r3|tabbyapi:r859-route-trace-r1) ;; *) echo "ABORT: GLM_IMG $IMG not allowed" >&2; exit 3;; esac
PROFILE=${PROFILE:-0}; PINNED_ARENA=${PINNED_ARENA:-0}; ROUTE_TRACE_DIR=${ROUTE_TRACE_DIR:-}
case "$PROFILE$PINNED_ARENA" in 00|01|10|11) ;; *) echo 'ABORT: PROFILE/PINNED_ARENA must be 0 or 1' >&2; exit 3;; esac
NAME=glm53
PORT_FIXED=8029
DRY_RUN=${DRY_RUN:-0}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
case "$DRY_RUN" in 0|1) ;; *) echo 'ABORT: DRY_RUN must be 0 or 1' >&2; exit 3;; esac
[[ "$BOOT_TIMEOUT" =~ ^[0-9]+$ ]] && ((BOOT_TIMEOUT >= 30 && BOOT_TIMEOUT <= 3600)) || { echo 'ABORT: BOOT_TIMEOUT must be 30..3600 seconds' >&2; exit 3; }
# Validate before creating files or changing any live state.
RESOLVED=$(python3 "$HERE/glm53_plan.py" check)
PACK_NAME=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["pack"])' "$RESOLVED")
CPU_THREADS=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["threads"])' "$RESOLVED")
DRAFT_RESOLVED=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["draft"])' "$RESOLVED")
RUN_DIR=${RUN_DIR:-/srv/qwen5090/results/glm53-manual-$(date +%Y%m%d-%H%M%S)}
if [[ "$DRY_RUN" == 1 ]]; then
  RUN_DIR=${RUN_DIR_DRY:-$HERE/dry-run/launcher}
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
CFG=$RUN_DIR/config.yml
LOG=$RUN_DIR/launch.log
log(){ printf '%s [glm53] %s\n' "$(date -Iseconds 2>/dev/null || date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
python3 "$HERE/glm53_plan.py" config > "$CFG"
printf '%s\n' "$RESOLVED" > "$RUN_DIR/resolved.json"
cp "$HERE/FLAG-DECISIONS.txt" "$RUN_DIR/FLAG-DECISIONS.txt"
log "config: $RESOLVED; fixed container $NAME port $PORT_FIXED image $IMG"
if [[ "$DRY_RUN" == 1 ]]; then
  log 'CPU DRY RUN: generated config and flag decisions; no docker, GPU, clocks, lock or service changes'
  exit 0
fi
for cmd in sudo docker nvidia-smi curl flock timeout; do command -v "$cmd" >/dev/null || { log "ABORT: missing $cmd"; exit 3; }; done
# Standalone launch takes the same lock; inherited fd9 avoids deadlocking the audition.
[[ "$(readlink /proc/self/fd/9 2>/dev/null || true)" == /srv/qwen5090/gpu-exclusive.lock ]] || exec 9>/srv/qwen5090/gpu-exclusive.lock
flock 9
CKPT=/storage/data/models/$PACK_NAME
python3 "$HERE/glm53_plan.py" inspect "$CKPT" > "$RUN_DIR/pack.json"
python3 "$HERE/glm53_plan.py" template "$CKPT" > "$RUN_DIR/template.json"
sudo -n docker image inspect "$IMG" > "$RUN_DIR/image.json"
# Environment inherited from the Docker image is also sanitized, not merely host env.
UNSET_ENV=()
while IFS= read -r key; do UNSET_ENV+=(-u "$key"); done < <(python3 - "$RUN_DIR/image.json" <<'PY'
import json,sys
for kv in json.load(open(sys.argv[1]))[0]['Config'].get('Env') or []:
    k=kv.partition('=')[0]
    if k.startswith('EXL3_') or k == 'TABBY_OUTPUT_CHUNK_TOKENS': print(k)
PY
)
SANITIZED=("${UNSET_ENV[@]}" "EXL3_MOE_CPU_THREADS=$CPU_THREADS" "EXL3_MOE_PINNED_ARENA=$PINNED_ARENA" PYTHONUNBUFFERED=1)
[[ "$PROFILE" == 1 ]] && SANITIZED+=(EXL3_MOE_CPU_PROF=1)
TRACE_MOUNT=()
if [[ -n "$ROUTE_TRACE_DIR" ]]; then
  [[ "$IMG" == tabbyapi:r859-route-trace-r1 ]] || { log 'ABORT: ROUTE_TRACE_DIR needs GLM_IMG=tabbyapi:r859-route-trace-r1'; exit 3; }
  mkdir -p "$ROUTE_TRACE_DIR"; chmod 0777 "$ROUTE_TRACE_DIR"
  TRACE_MOUNT=(-v "$ROUTE_TRACE_DIR":/app/route-traces); SANITIZED+=(EXL3_ROUTE_TRACE=/app/route-traces/run)
fi
# LIVE's same config.load path, in the exact running image, before destroying a container.
timeout --kill-after=15s 90s sudo -n docker run --rm --name glm53-preflight \
  -v "$CFG":/app/config.yml:ro -w /app --entrypoint /usr/bin/env "$IMG" \
  "${SANITIZED[@]}" python3 -c 'from common.tabby_config import config; config.load({})' \
  > "$RUN_DIR/preflight.log" 2>&1 || { cat "$RUN_DIR/preflight.log"; log 'ABORT: image schema preflight failed'; exit 3; }
# A standalone launch will never take the :8022 daily down behind the operator's back.
if sudo -n docker ps --format '{{.Names}}' | grep -xE 'flashnext|vllm-27b' > "$RUN_DIR/conflicts.txt"; then
  log 'ABORT: daily still running; use r858-glm53-audition.sh to stop it under the queue lock'; exit 3
fi
sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
idle=0
for ((i=0; i<24; i++)); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$RUN_DIR/gpu-idle.csv"
  if awk 'BEGIN {ok=1} $1>1024 {ok=0} END {exit !ok}' "$RUN_DIR/gpu-idle.csv"; then idle=1; break; fi
  sleep 5
done
[[ "$idle" == 1 ]] || { log 'ABORT: GPUs still busy after 120s'; exit 3; }
[[ "$(wc -l < "$RUN_DIR/gpu-idle.csv" | tr -d ' ')" == 2 ]] || { log 'ABORT: expected exactly two GPUs'; exit 3; }
# Same policies as LIVE, with readback failures fatal instead of swallowed.
bash /srv/qwen5090/daily-power.sh stock > "$RUN_DIR/power-policy.log" 2>&1
sudo -n python3 - <<'PY' | tee "$RUN_DIR/clocks.txt"
import pynvml as N
N.nvmlInit()
assert N.nvmlDeviceGetCount() == 2
for i in range(2):
    h=N.nvmlDeviceGetHandleByIndex(i)
    N.nvmlDeviceSetMemClkVfOffset(h,4500)
    N.nvmlDeviceSetGpcClkVfOffset(h,0)
    mem=N.nvmlDeviceGetMemClkVfOffset(h); core=N.nvmlDeviceGetGpcClkVfOffset(h)
    power=N.nvmlDeviceGetPowerManagementLimit(h)//1000
    print(i, 'memory offset',mem,'core offset',core,'power W',power,flush=True)
    assert (mem,core,power)==(4500,0,(600,575)[i]), 'clock/power readback differs from LIVE'
PY
TUNEDIR=/srv/qwen5090/.exl3cache-r858-glm53
mkdir -p "$TUNEDIR"
STARTED=0
LOG_PID=
failed(){ local rc=$?; trap - EXIT
  if [[ -n "$LOG_PID" ]]; then kill "$LOG_PID" 2>/dev/null || true; wait "$LOG_PID" 2>/dev/null || true; fi
  if ((rc != 0)); then
    log "FAILED rc=$rc; archiving state and tearing down glm53"
    sudo -n docker inspect "$NAME" > "$RUN_DIR/failed-inspect.json" 2>&1 || true
    sudo -n docker logs "$NAME" > "$RUN_DIR/docker-final.log" 2>&1 || true
    [[ "$STARTED" == 0 ]] || sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap failed EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
STARTED=1
sudo -n docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=16g --restart no \
  -v "$TUNEDIR":/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  -p 0.0.0.0:8029:8029 -v "$CKPT":/models/"$PACK_NAME":ro -v "$CFG":/app/config.yml:ro "${TRACE_MOUNT[@]}" \
  -w /app --entrypoint /usr/bin/env "$IMG" "${SANITIZED[@]}" \
  python3 main.py --host 0.0.0.0 --port 8029 --disable-auth true | tee "$RUN_DIR/container-id.txt"
python3 -u "$HERE/glm53_follow.py" "$RUN_DIR/docker-stream.log" & LOG_PID=$!
end=$((SECONDS + BOOT_TIMEOUT))
up=0
while ((SECONDS < end)); do
  state=$(sudo -n docker inspect -f '{{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' "$NAME")
  [[ "$state" == 'running false 0' ]] || { log "NO BOOT: $state"; exit 1; }
  sudo -n docker logs "$NAME" > "$RUN_DIR/docker-current.log" 2>&1
  if grep -aiE 'out of memory|OutOfMemory|Insufficient VRAM|Traceback|CPU offload skipped|CPU split.*skipped' "$RUN_DIR/docker-current.log"; then
    log 'NO BOOT: engine error or offload eligibility failure'; exit 1
  fi
  if curl -fsS --max-time 3 http://127.0.0.1:8029/health >/dev/null 2>&1; then up=1; break; fi
  sleep 2
done
[[ "$up" == 1 ]] || { log "NO BOOT: timeout ${BOOT_TIMEOUT}s"; exit 1; }
curl -fsS --max-time 8 http://127.0.0.1:8029/v1/model > "$RUN_DIR/api-model.json"
python3 - "$RUN_DIR/api-model.json" "$PACK_NAME" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['id']==sys.argv[2], 'wrong served model'
PY
sudo -n docker logs "$NAME" > "$RUN_DIR/docker-current.log" 2>&1
if [[ "$DRAFT_RESOLVED" == 1 ]]; then
  grep -aF 'Using main model MTP component for drafting' "$RUN_DIR/docker-current.log" >/dev/null || { log 'NO BOOT: requested MTP component not loaded'; exit 1; }
fi
# Check actual registration counts, not only that config requested offload.
python3 "$HERE/glm53_verify.py" offload "$RUN_DIR/resolved.json" "$RUN_DIR/docker-current.log"
python3 "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$PACK_NAME" --out "$RUN_DIR/warmup.jsonl" --phase warmup
sudo -n docker inspect "$NAME" > "$RUN_DIR/served-inspect.json"
free -b | tee "$RUN_DIR/free-boot.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,power.limit,clocks.mem,clocks.gr --format=csv | tee "$RUN_DIR/vram-boot.csv"
log 'UP: http://flan:8029/v1 (FASTWARM omitted: Flash-Next calibration; generic warmup completed)'

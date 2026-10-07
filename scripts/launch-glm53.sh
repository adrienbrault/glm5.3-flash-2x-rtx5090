#!/usr/bin/env bash
# Derived from LIVE config/preflight/run/health/model guards and clocks; GLM policy is separate.
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMG=${GLM_IMG:-tabbyapi:r828-prompt-lookup-r3}
# R859 diagnostic knobs: PROFILE=1 sets EXL3_MOE_CPU_PROF=1; ROUTE_TRACE_DIR (host dir, needs the route-trace image)
# mounts at /app/route-traces and sets EXL3_ROUTE_TRACE; PINNED_ARENA=1 sets EXL3_MOE_PINNED_ARENA=1.
case "$IMG" in tabbyapi:r828-prompt-lookup-r3|tabbyapi:r859-route-trace-r1|tabbyapi:r861-glm-agent-r2|tabbyapi:r862-gpucache-r2|tabbyapi:cheapswap-r2|tabbyapi:cheapswap-r3|tabbyapi:r861-glm-agent-r3|tabbyapi:r861-glm-agent-r4|tabbyapi:cheapswap-r3-agent-r2|tabbyapi:r861-glm-agent-r4b|tabbyapi:r861-glm-agent-r4c) ;; *) echo "ABORT: GLM_IMG $IMG not allowed" >&2; exit 3;; esac
PROFILE=${PROFILE:-0}; PINNED_ARENA=${PINNED_ARENA:-0}; ROUTE_TRACE_DIR=${ROUTE_TRACE_DIR:-}
# R864: AGENT=1 turns on the r861 agent overlay (live tool-call streaming, SSE keepalive, GLM tool fixes; grammar
# forcing stays off until probe_grammar.py passes on the real tokenizer). Needs GLM_IMG=tabbyapi:r861-glm-agent-r2.
AGENT=${AGENT:-0}
# R871: HOST_CONFINE=1 emulates exllamav3 dev fa822cf ("CPU MoE: reserve a host core", upstream +15 % decode without MTP on
# a 7900X+4090): after boot, the TabbyAPI host process is confined to the cores the pinned CPU-MoE worker does not use
# (CPU_THREADS < 8: cores CPU_THREADS..7 and their SMT siblings; CPU_THREADS=8: the SMT siblings 8..15).
HOST_CONFINE=${HOST_CONFINE:-0}
# R872: SWAP_MODE=exchange turns on cheapswap-r2's checkpoint-free GPU<->worker exchange (needs GLM_IMG=tabbyapi:cheapswap-r2,
# PINNED_ARENA=1, dynamic placement). SWAP_CADENCE (exact|served), SWAP_MAX, SWAP_SCOPE (global|layer), SWAP_HYST pass through.
# R875: with cheapswap-r3, SWAP_INIT_STATS=<host json> starts exchange mode from a static counts file (mounted like
# PLACEMENT=static's SPLIT_STATS) and SWAP_POLICY (histogram|score) picks the selection policy.
SWAP_MODE=${SWAP_MODE:-}; SWAP_CADENCE=${SWAP_CADENCE:-}; SWAP_MAX=${SWAP_MAX:-}; SWAP_SCOPE=${SWAP_SCOPE:-}; SWAP_HYST=${SWAP_HYST:-}
SWAP_INIT_STATS=${SWAP_INIT_STATS:-}; SWAP_POLICY=${SWAP_POLICY:-}
case "$SWAP_MODE" in '') ;; exchange) [[ "$IMG" =~ ^tabbyapi:cheapswap-r[23](-agent-r2)?$ && "${PINNED_ARENA:-0}" == 1 ]] || { echo 'ABORT: SWAP_MODE=exchange needs GLM_IMG=tabbyapi:cheapswap-r2|r3 and PINNED_ARENA=1' >&2; exit 3; };; *) echo 'ABORT: SWAP_MODE must be empty or exchange' >&2; exit 3;; esac
if [[ -n "$SWAP_INIT_STATS$SWAP_POLICY" ]]; then
  [[ "$IMG" =~ ^tabbyapi:cheapswap-r3(-agent-r2)?$ && -n "$SWAP_MODE" ]] || { echo 'ABORT: SWAP_INIT_STATS/SWAP_POLICY need GLM_IMG=tabbyapi:cheapswap-r3 and SWAP_MODE=exchange' >&2; exit 3; }
  [[ -z "$SWAP_INIT_STATS" || -f "$SWAP_INIT_STATS" ]] || { echo 'ABORT: SWAP_INIT_STATS file missing' >&2; exit 3; }
  [[ "$SWAP_POLICY" =~ ^(|histogram|score)$ ]] || { echo 'ABORT: SWAP_POLICY must be histogram or score' >&2; exit 3; }
fi
[[ "$SWAP_CADENCE" =~ ^(|exact|served)$ && "$SWAP_MAX" =~ ^[0-9]*$ && "$SWAP_SCOPE" =~ ^(|global|layer)$ && "$SWAP_HYST" =~ ^([0-9]+(\.[0-9]+)?)?$ ]] || { echo 'ABORT: bad SWAP_* value' >&2; exit 3; }
case "$HOST_CONFINE" in 0|1) ;; *) echo 'ABORT: HOST_CONFINE must be 0 or 1' >&2; exit 3;; esac
case "$AGENT" in 0) ;; 1) [[ "$IMG" =~ ^tabbyapi:(r861-glm-agent-r[234][bc]?|cheapswap-r3-agent-r2)$ ]] || { echo 'ABORT: AGENT=1 needs GLM_IMG=tabbyapi:r861-glm-agent-r2|r3|r4' >&2; exit 3; };; *) echo 'ABORT: AGENT must be 0 or 1' >&2; exit 3;; esac
# R877: TAG_TRACE=1 (agent r3/r4 images only) logs GLM stop sources, the EOS trigger id and parser state ([GLM-TAG-R3]).
TAG_TRACE=${TAG_TRACE:-0}
case "$TAG_TRACE" in 0) ;; 1) [[ "$IMG" =~ ^tabbyapi:r861-glm-agent-r[34][bc]?$ ]] || { echo 'ABORT: TAG_TRACE=1 needs GLM_IMG=tabbyapi:r861-glm-agent-r3|r4' >&2; exit 3; };; *) echo 'ABORT: TAG_TRACE must be 0 or 1' >&2; exit 3;; esac
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
PLOOKUP=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("plookup",0))' "$RESOLVED")
read -r PLACEMENT SWAP_INTERVAL SWAP_FLOOR < <(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);print(d.get("placement","dynamic"),d.get("swap_interval",0),d.get("swap_floor",0))' "$RESOLVED")
SPLIT_STATS=${SPLIT_STATS:-}
if [[ "$PLACEMENT" == static ]]; then
  [[ -f "$SPLIT_STATS" ]] || { echo 'ABORT: PLACEMENT=static needs SPLIT_STATS=<host json file>' >&2; exit 3; }
elif [[ -n "$SPLIT_STATS" ]]; then echo 'ABORT: SPLIT_STATS needs PLACEMENT=static' >&2; exit 3; fi
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
[[ "$PLOOKUP" == 1 ]] && SANITIZED+=(EXL3_PROMPT_LOOKUP=1)
STATS_MOUNT=()
if [[ "$PLACEMENT" == static ]]; then
  STATS_MOUNT=(-v "$SPLIT_STATS":/app/split-stats.json:ro); SANITIZED+=(EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT_STATS=/app/split-stats.json)
fi
((SWAP_INTERVAL > 0)) && SANITIZED+=("EXL3_MOE_CPU_SWAP_INTERVAL=$SWAP_INTERVAL")
((SWAP_FLOOR > 0)) && SANITIZED+=("EXL3_MOE_CPU_SWAP_FLOOR=$SWAP_FLOOR")
if [[ -n "$SWAP_MODE" ]]; then
  [[ "$PLACEMENT" == dynamic ]] || { echo 'ABORT: SWAP_MODE=exchange needs dynamic placement' >&2; exit 3; }
  SANITIZED+=(EXL3_MOE_CPU_SWAP=1 "EXL3_MOE_CPU_SWAP_MODE=$SWAP_MODE")
  [[ -n "$SWAP_CADENCE" ]] && SANITIZED+=("EXL3_MOE_CPU_SWAP_CADENCE=$SWAP_CADENCE")
  [[ -n "$SWAP_MAX" ]] && SANITIZED+=("EXL3_MOE_CPU_SWAP_MAX=$SWAP_MAX")
  [[ -n "$SWAP_SCOPE" ]] && SANITIZED+=("EXL3_MOE_CPU_SWAP_BUDGET_SCOPE=$SWAP_SCOPE")
  [[ -n "$SWAP_HYST" ]] && SANITIZED+=("EXL3_MOE_CPU_SWAP_HYST=$SWAP_HYST")
  [[ -n "$SWAP_POLICY" ]] && SANITIZED+=("EXL3_MOE_CPU_SWAP_POLICY=$SWAP_POLICY")
  if [[ -n "$SWAP_INIT_STATS" ]]; then
    STATS_MOUNT=(-v "$SWAP_INIT_STATS":/app/split-stats.json:ro); SANITIZED+=(EXL3_MOE_CPU_SPLIT_STATS=/app/split-stats.json)
  fi
fi
[[ "$AGENT" == 1 ]] && SANITIZED+=(TABBY_STREAM_TOOLCALLS=1 TABBY_SSE_KEEPALIVE_S=5 TABBY_GLM_TOOL_FIXES=1 TABBY_GLM_FORCING=0)
[[ "$TAG_TRACE" == 1 ]] && SANITIZED+=(TABBY_GLM_TAG_TRACE=1)
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
TEMPLATE_MOUNT=()
if [[ "$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("keep_thinking",0))' "$RESOLVED")" == 1 ]]; then
  [[ -f "$HERE/templates/glm53-keep-thinking.jinja" ]] || { log "ABORT: KEEP_THINKING=1 but $HERE/templates/glm53-keep-thinking.jinja missing"; exit 3; }
  TEMPLATE_MOUNT=(-v "$HERE/templates/glm53-keep-thinking.jinja":/app/templates/glm53-keep-thinking.jinja:ro)
fi
STARTED=1
sudo -n docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=16g --restart no \
  -v "$TUNEDIR":/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  -p 0.0.0.0:8029:8029 -v "$CKPT":/models/"$PACK_NAME":ro -v "$CFG":/app/config.yml:ro "${TRACE_MOUNT[@]}" "${STATS_MOUNT[@]}" "${TEMPLATE_MOUNT[@]}" \
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
MAX_BATCH_RESOLVED=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["max_batch"])' "$RESOLVED")
python3 "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$PACK_NAME" --out "$RUN_DIR/warmup.jsonl" --phase warmup --max-batch "$MAX_BATCH_RESOLVED"
sudo -n docker inspect "$NAME" > "$RUN_DIR/served-inspect.json"
free -b | tee "$RUN_DIR/free-boot.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,power.limit,clocks.mem,clocks.gr --format=csv | tee "$RUN_DIR/vram-boot.csv"
if [[ "$HOST_CONFINE" == 1 ]]; then
  if ((CPU_THREADS < 8)); then CONF="$CPU_THREADS-7,$((CPU_THREADS + 8))-15"; else CONF="8-15"; fi
  HOST_PID=$(sudo -n docker top "$NAME" -eo pid,args | awk '/python3 main.py/{print $1; exit}')
  [[ -n "$HOST_PID" ]] || { log 'ABORT: HOST_CONFINE could not find the TabbyAPI host process'; exit 1; }
  sudo -n taskset -a -cp "$CONF" "$HOST_PID" > "$RUN_DIR/host-confine.txt" 2>&1 || { log 'ABORT: taskset failed'; exit 1; }
  log "HOST_CONFINE: host pid $HOST_PID (all threads) -> CPUs $CONF"
fi
log 'UP: http://flan:8029/v1 (FASTWARM omitted: Flash-Next calibration; generic warmup completed)'

#!/usr/bin/env bash
# GPU-exclusive flan unit. CPU dry run is portable and performs no live actions.
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${DRY_RUN:-0}" == 1 ]]; then exec python3 "$HERE/glm53_dry_run.py"; fi
KEEP_GLM=${KEEP_GLM:-0}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
MEASURE_TIMEOUT=${MEASURE_TIMEOUT:-7200}
SANITY_TIMEOUT=${SANITY_TIMEOUT:-1200}
RUNS=${RUNS:-3}
CACHE_TOKENS=${CACHE_TOKENS:-65536}
GPU_SPLIT=${GPU_SPLIT:-31,31}
CACHE_MODE=${CACHE_MODE:-8,8}
DRAFT_CHOICE=${DRAFT:-auto}
CPU_THREADS=${CPU_THREADS:-8}
case "$KEEP_GLM" in 0|1) ;; *) echo 'KEEP_GLM must be 0 or 1' >&2; exit 3;; esac
case "$DRAFT_CHOICE" in auto|0|1) ;; *) echo 'DRAFT must be auto, 0, or 1' >&2; exit 3;; esac
for value in "$BOOT_TIMEOUT" "$MEASURE_TIMEOUT" "$SANITY_TIMEOUT" "$RUNS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'timeouts/RUNS must be positive integers' >&2; exit 3; }
done
(( RUNS <= 10 && BOOT_TIMEOUT >= 30 && BOOT_TIMEOUT <= 3600 && SANITY_TIMEOUT <= 3600 && MEASURE_TIMEOUT <= 43200 )) || { echo 'timeouts/RUNS outside supported limits' >&2; exit 3; }
for cmd in sudo docker curl python3 timeout flock nvidia-smi free; do command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 3; }; done
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
"${CLEAN[@]}" PACK=A DRAFT=1 CACHE_TOKENS="$CACHE_TOKENS" GPU_SPLIT="$GPU_SPLIT" CACHE_MODE="$CACHE_MODE" CPU_THREADS="$CPU_THREADS" python3 "$HERE/glm53_plan.py" check >/dev/null
R=/srv/qwen5090/results/$(date +%F)-r858-glm53-$(date +%H%M%S)-$$
mkdir -p "$R"
# Global transcript plus individual phase logs, line-buffered probes and JSONL files.
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r858] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
LIVE=/srv/qwen5090/launch-flashnext.sh
[[ -f "$LIVE" && -f /srv/qwen5090/daily-power.sh ]] || { note 'ABORT: LIVE launcher or power helper missing'; exit 3; }
sudo -n docker image inspect tabbyapi:r828-prompt-lookup-r3 > "$R/audition-image.json"
cp "$LIVE" "$R/live-before.sh"
cp -R "$HERE" "$R/build"
export GPU_QUEUE_NAME=r858-glm53-$$
if [[ -f /srv/qwen5090/lib/gpu-queue.sh ]]; then
  . /srv/qwen5090/lib/gpu-queue.sh
else
  . "$HERE/lib/gpu-queue.sh"
fi
# Replace library EXIT trap but preserve marker removal after our restoration.
DAILY_STOPPED=0
KEEP_ARMED=0
MON_PID=
DOCKER_PID=
CURRENT=
kill_monitors(){
  for pid in "$MON_PID" "$DOCKER_PID"; do
    [[ -z "$pid" ]] || { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
  done
  MON_PID=; DOCKER_PID=
}
restore(){
  sudo -n docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
  note 'RESTORE: /srv/qwen5090/launch-flashnext.sh (LIVE Flash-Next)'
  timeout --kill-after=20s 1200s "${CLEAN[@]}" bash "$LIVE" 2>&1 | tee "$R/restore-launch.log" || return 1
  for ((j=0;j<90;j++)); do
    if curl -fsS --max-time 5 http://127.0.0.1:8022/health >/dev/null 2>&1; then
      sudo -n docker inspect flashnext > "$R/restored-inspect.json" || return 1
      python3 "$HERE/glm53_verify.py" daily "$R/restored-inspect.json" || return 1
      curl -fsS --max-time 8 http://127.0.0.1:8022/v1/model > "$R/restored-model.json" || return 1
      python3 - "$R/restored-model.json" <<'PY' || return 1
import json,sys
assert json.load(open(sys.argv[1]))['id']=='qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab', 'wrong restored daily model'
PY
      note 'RESTORED: Flash-Next :8022, full dfaed2cb image pin, no TABBY_OUTPUT_CHUNK_TOKENS'
      return 0
    fi
    sleep 2
  done
  return 1
}
cleanup(){
  local rc=$?
  trap - EXIT INT TERM HUP
  kill_monitors
  sudo -n docker logs glm53 > "$R/glm-final.log" 2>&1 || true
  sudo -n docker inspect glm53 > "$R/glm-final-inspect.json" 2>&1 || true
  if [[ "$DAILY_STOPPED" == 1 ]]; then
    if [[ "$KEEP_ARMED" == 1 ]] && curl -fsS --max-time 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
      note "KEEP_GLM=1: best validated config remains at :8029; daily not restored; results $R"
    else
      restore || { note "RESTORE FAILED: run bash $HERE/r858-restore-daily.sh; see restore-launch.log"; rc=5; }
    fi
  fi
  rm -f "$GPU_QUEUE_MARK"
  printf 'exit=%s\nresults=%s\nkeep_armed=%s\n' "$rc" "$R" "$KEEP_ARMED" > "$R/last.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'note "interrupted"; exit 130' INT
trap 'note "terminated"; exit 143' TERM HUP
gpu_lock
note "GPU lock held; results $R"
sudo -n docker inspect flashnext > "$R/daily-entry-inspect.json"
# 2026-10-07: :8022 may still serve the R855 B1 candidate (tabbyapi:r855-outchunk-r1, overlay 6883efc6...) left
# from the 10-06 run; accept exactly that image at ENTRY only. The exit restore still pins dfaed2cb + no knob.
R855_B1=sha256:6883efc656db2c5cac8cffd3a160924d85aa5711b4bce44cc1cb552dd7059d7f
if [[ "$(sudo -n docker inspect -f '{{.Image}}' flashnext)" == "$R855_B1" ]]; then
  note "entry: :8022 serves the R855 B1 candidate image (accepted at entry; restore pins the daily)"
else
  python3 "$HERE/glm53_verify.py" daily "$R/daily-entry-inspect.json"
fi
curl -fsS --max-time 8 http://127.0.0.1:8022/v1/model > "$R/daily-entry-model.json"
python3 - "$R/daily-entry-model.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['id']=='qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab'
PY
# Confirm A is complete before disrupting the daily. B is inspected after A (may be downloading).
python3 "$HERE/glm53_plan.py" inspect /storage/data/models/glm53-flash-exl3-2.05bpw-turboderp > "$R/pack-A.json"
python3 "$HERE/glm53_plan.py" template /storage/data/models/glm53-flash-exl3-2.05bpw-turboderp > "$R/template-A.json"
# GLM-5.3 template has no thinking-off switch (always opens <think>); the probe then runs effort low/high arms.
if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("always_think") else 1)' "$R/template-A.json"; then
  export R858_ALWAYS_THINK=1; note "template: always-think, sanity arms = reasoning_effort low/high"
fi
DAILY_STOPPED=1
sudo -n docker logs flashnext > "$R/daily-entry.log" 2>&1
sudo -n docker rm -f flashnext
sudo -n docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
free -b | tee "$R/free-after-stop.txt"

available(){ awk '/^MemAvailable:/ {printf "%.0f\n", $2*1024}' /proc/meminfo; }

attempt(){
  local pack=$1 tag=$2 mode=$3 n=$4 d=$5 peak=$6
  CURRENT=$R/$tag
  mkdir -p "$CURRENT"
  sudo -n docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
  # Let process teardown release resident host pages; then recheck actual admission.
  sleep 3
  if (( $(available) < peak )); then note "$tag SKIP: host load estimate $peak > current MemAvailable $(available)"; return 1; fi
  note "$tag TRY: $pack $mode N=$n MTP=$d cache=$CACHE_TOKENS @ $CACHE_MODE split=$GPU_SPLIT"
  python3 -u "$HERE/glm53_resources.py" "$CURRENT" & MON_PID=$!
  local rc=0
  timeout --kill-after=20s "$((BOOT_TIMEOUT + 480))s" "${CLEAN[@]}" PACK="$pack" \
    OFFLOAD_MODE="$mode" OFFLOAD_N="$n" DRAFT="$d" CACHE_TOKENS="$CACHE_TOKENS" CACHE_MODE="$CACHE_MODE" \
    GPU_SPLIT="$GPU_SPLIT" CPU_THREADS="$CPU_THREADS" BOOT_TIMEOUT="$BOOT_TIMEOUT" RUN_DIR="$CURRENT" \
    bash "$HERE/launch-glm53.sh" 2>&1 | tee "$CURRENT/boot.log" || rc=$?
  if ((rc == 0)); then
    python3 -u "$HERE/glm53_follow.py" "$CURRENT/engine-measure.log" & DOCKER_PID=$!
    local model
    model=$(python3 - "$CURRENT/resolved.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['pack'])
PY
)
    timeout --kill-after=15s "${SANITY_TIMEOUT}s" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 \
      --model "$model" --out "$CURRENT/sanity.jsonl" --phase sanity 2>&1 | tee "$CURRENT/sanity.log" || rc=$?
    if ((rc == 0)); then
      timeout --kill-after=15s "${MEASURE_TIMEOUT}s" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 \
        --model "$model" --out "$CURRENT/measure.jsonl" --phase measure --runs "$RUNS" 2>&1 | tee "$CURRENT/measure.log" || rc=$?
    fi
  fi
  if [[ -n "$MON_PID" ]] && ! kill -0 "$MON_PID" 2>/dev/null; then rc=1; note "$tag FAILED: resource monitor exited"; fi
  kill_monitors
  sudo -n docker logs glm53 > "$CURRENT/engine-final.log" 2>&1 || true
  sudo -n docker inspect glm53 > "$CURRENT/inspect-final.json" 2>&1 || true
  free -b > "$CURRENT/free-final.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$CURRENT/vram-final.csv"
  [[ ! -f "$CURRENT/host-guard-failed" ]] || rc=1
  if ((rc == 0)); then
    python3 - "$CURRENT/inspect-final.json" <<'PY' || rc=$?
import json,sys
d=json.load(open(sys.argv[1]))[0]
assert d['State']['Running'] and not d['State'].get('OOMKilled') and d['RestartCount']==0
PY
  fi
  printf '%s\n' "$rc" > "$CURRENT/exit-code.txt"
  if ((rc != 0)); then
    note "$tag FAILED rc=$rc (boot/sanity/measurement/resource logs archived)"
    sudo -n docker rm -f glm53 glm53-preflight >/dev/null 2>&1 || true
    return 1
  fi
  python3 - "$pack" "$tag" "$CURRENT" "$R/winners.jsonl" <<'PY'
import json,sys
pack,tag,path,out=sys.argv[1:]
rows=[json.loads(x) for x in open(path+'/measure.jsonl')]
summ=next(r for r in rows if r['phase']=='decode-summary')
score=next(s['ss_agg_tps_median'] for s in summ['summaries'] if s['c']==1)
config=json.load(open(path+'/resolved.json'))
with open(out,'a') as f:f.write(json.dumps(dict(pack=pack,tag=tag,path=path,score_c1=score,config=config))+'\n')
print('WINNER',pack,tag,'c1',score,flush=True)
PY
  local score_rc=$?
  ((score_rc == 0)) || { note "$tag FAILED: unable to produce measured score"; return 1; }
  note "$tag PASS: sanity thinking off/on, forced decode c1/c2/c4, 8k/32k prefill"
}

for pack in A B; do
  if [[ "$pack" == A ]]; then dir=/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp
  else dir=/storage/data/models/glm53-flash-exl3-2.25bpw-r0b0tlab; fi
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  sleep 3
  if ! python3 "$HERE/glm53_plan.py" inspect "$dir" > "$R/pack-$pack.json" 2> "$R/pack-$pack-check.log"; then
    note "$pack SKIPPED: absent/incomplete/ineligible pack; see pack-$pack-check.log"
    [[ "$pack" == B ]] && continue
    exit 3
  fi
  if ! python3 "$HERE/glm53_plan.py" template "$dir" > "$R/template-$pack.json" 2> "$R/template-$pack-check.log"; then
    note "$pack SKIPPED: thinking template not established; exact probe in FIT-ANALYSIS.md"; continue
  fi
  # PLAN_SPLIT (default GPU_SPLIT): planner budget only. 2026-10-07 first run: the estimate missed real VRAM by
  # several GB (split-80 and layers-12 both OOM at module 45/50), so plan tighter while loading at GPU_SPLIT.
  python3 "$HERE/glm53_plan.py" ladder "$dir" --available "$(available)" --cache "$CACHE_TOKENS" \
    --split "${PLAN_SPLIT:-$GPU_SPLIT}" --draft "$DRAFT_CHOICE" --out "$R/ladder-$pack.json"
  python3 - "$R/ladder-$pack.json" > "$R/ladder-$pack.tsv" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for i,c in enumerate(d['candidates']):
    print(i,c['mode'],c['n'],c['draft'],c['host_peak'],sep='\t')
PY
  cat "$R/ladder-$pack.json"
  if [[ ! -s "$R/ladder-$pack.tsv" ]]; then note "$pack NO-FIT: no overlap of host and GPU admission estimates"; continue; fi
  while IFS=$'\t' read -r rung mode n draft peak; do
    tag=$pack-r$rung-$mode-$n-mtp$draft
    if attempt "$pack" "$tag" "$mode" "$n" "$draft" "$peak"; then break; fi
  done < "$R/ladder-$pack.tsv"
done
[[ -s "$R/winners.jsonl" ]] || { note 'NO SANE MEASURED CONFIG: restoring daily'; exit 1; }
python3 - "$R/winners.jsonl" "$R/best.json" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1])]
best=max(rows,key=lambda r:(r['score_c1'],r['pack']=='A'))
json.dump(best,open(sys.argv[2],'w'),indent=2)
print('BEST',json.dumps(best),flush=True)
PY
if [[ "$KEEP_GLM" == 1 ]]; then
  # Reboot measured best (A may have won but was removed to audition B). Validate again.
  python3 - "$R/best.json" > "$R/best.tsv" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['config']
print(s['pack'],s['mode'],s['n'],s['draft'],sep='\t')
PY
  IFS=$'\t' read -r best_pack best_mode best_n best_draft < "$R/best.tsv"
  CURRENT=$R/keep-best; mkdir -p "$CURRENT"
  timeout --kill-after=20s "$((BOOT_TIMEOUT + 480))s" "${CLEAN[@]}" PACK="$best_pack" \
    OFFLOAD_MODE="$best_mode" OFFLOAD_N="$best_n" DRAFT="$best_draft" CACHE_TOKENS="$CACHE_TOKENS" \
    GPU_SPLIT="$GPU_SPLIT" CACHE_MODE="$CACHE_MODE" CPU_THREADS="$CPU_THREADS" \
    BOOT_TIMEOUT="$BOOT_TIMEOUT" RUN_DIR="$CURRENT" bash "$HERE/launch-glm53.sh" 2>&1 | tee "$CURRENT/reboot.log"
  timeout --kill-after=15s "${SANITY_TIMEOUT}s" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 \
    --model "$best_pack" --out "$CURRENT/sanity.jsonl" --phase sanity 2>&1 | tee "$CURRENT/sanity.log"
  KEEP_ARMED=1
  note "SERVING BEST at http://flan:8029/v1; KEEP_GLM=1; holding GPU lock while unit runs; best.json is the review record"
  python3 -u "$HERE/glm53_resources.py" "$CURRENT" & MON_PID=$!
  python3 -u "$HERE/glm53_follow.py" "$CURRENT/engine-serving.log" & DOCKER_PID=$!
  # Unit stays alive to retain exclusive ownership. Stop releases lock and leaves healthy GLM.
  sudo -n docker wait glm53
  KEEP_ARMED=0
  note 'GLM stopped: restore daily on exit'
else
  note 'AUDITION COMPLETE: best.json records highest measured c1 throughput; restore daily on exit'
fi

#!/usr/bin/env bash
# R859 (2026-10-07, user: "let's focus on c1" + "full kv context eg 250k"): GLM-5.3-Flash c1 profile on flan.
# Pack A (turboderp 2.05 bpw), split CPU-MoE offload, 256k cache, chunk 2048, sysmem_recurrent_cache 1024 MB (R858's 0
# crashed the first prefill stash). One boot per arm, c1 only:
#   A   base: N=104 MTP on, 8 threads — c1 decode code/prose/chat x2, cold prefill + 256-token decode at 8k/32k/128k/240k
#       (falls back to N=112 if the 256k cache does not fit; every later arm then uses the N that booted)
#   B   MTP off — c1 decode x2
#   C   16 CPU threads (SMT) — c1 decode x2
#   D   7 CPU threads (one core left to the host process; upstream #453) — c1 decode x2
#   E   EXL3_MOE_PINNED_ARENA=1 — c1 decode x1 + 32k depth
#   F   diagnostics: route-trace image (EXL3_ROUTE_TRACE) + EXL3_MOE_CPU_PROF=1 — c1 decode x1 + 32k depth; graceful stop
# End: relaunch the fastest of A-E on :8029 and leave it serving (KEEP_GLM=1, default), else restore the Flash-Next
# daily through the LIVE launcher with the dfaed2cb image pin.
set -uo pipefail
HERE=/srv/qwen5090/r858
KEEP_GLM=${KEEP_GLM:-1}
R=/srv/qwen5090/results/$(date +%F)-r859-glm53-c1-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r859] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
LIVE=/srv/qwen5090/launch-flashnext.sh
PARENT_ID=sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
cp -R "$HERE" "$R/build"
cp "$LIVE" "$R/live-before.sh"
export GPU_QUEUE_NAME=r859-glm53-c1
. /srv/qwen5090/lib/gpu-queue.sh
DAILY_STOPPED=0
restore(){
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  note 'RESTORE: LIVE launch-flashnext.sh'
  timeout -k 30 1200 "${CLEAN[@]}" HOME="$HOME" bash "$LIVE" > "$R/restore.log" 2>&1 || { note 'RESTORE: launcher rc!=0'; return 1; }
  for ((j=0;j<120;j++)); do curl -fsS -m 5 http://127.0.0.1:8022/health >/dev/null 2>&1 && break; sleep 2; done
  [[ "$(sudo -n docker inspect -f '{{.Image}}' flashnext 2>/dev/null)" == "$PARENT_ID" ]] || { note 'RESTORE: flashnext is not the pinned daily image'; return 1; }
  ! sudo -n docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext | grep -q '^TABBY_OUTPUT_CHUNK_TOKENS=' || { note 'RESTORE: output knob set'; return 1; }
  note 'RESTORED: Flash-Next :8022 on the pinned image'
}
cleanup(){
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ "$DAILY_STOPPED" == 1 ]]; then
    if [[ "$KEEP_GLM" == 1 ]] && curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
      note "KEEP_GLM=1: GLM stays on :8029, daily not restored; results $R"
    else
      restore || rc=5
    fi
  fi
  rm -f "$GPU_QUEUE_MARK"
  echo "exit=$rc" > "$R/last.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'note interrupted; exit 130' INT
trap 'note terminated; exit 143' TERM HUP
gpu_lock
note "GPU lock held; results $R"
sudo -n docker inspect -f '{{.Image}}' flashnext > "$R/daily-entry-image.txt" 2>&1 || true
sudo -n docker logs flashnext > "$R/daily-entry.log" 2>&1 || true
DAILY_STOPPED=1
sudo -n docker rm -f flashnext glm53 glm53-preflight >/dev/null 2>&1 || true

N=104
run_arm(){  # tag, probe args, then KEY=VALUE launcher env
  local tag=$1 probe=$2; shift 2
  local D=$R/$tag rc=0
  mkdir -p "$D"
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: N=$N $*"
  timeout -k 20 1500 "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split OFFLOAD_N="$N" CACHE_TOKENS=262144 MAX_SEQ=262144 \
    CHUNK=2048 SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$D" "$@" \
    bash "$HERE/launch-glm53.sh" > "$D/boot.log" 2>&1 || rc=$?
  if ((rc != 0)); then note "$tag NO BOOT rc=$rc ($(grep -ahoE 'Insufficient VRAM|out of memory|NO BOOT[^;]*' "$D"/*.log | head -1))"; return 1; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$D/vram-up.csv"; free -b > "$D/free-up.txt"
  # shellcheck disable=SC2086
  timeout -k 15 7200 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/c1.jsonl" --phase c1 $probe > "$D/c1.log" 2>&1 || rc=$?
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$D/vram-end.csv"; free -b > "$D/free-end.txt"
  python3 - "$D/c1.jsonl" "$tag" <<'PY' | tee -a "$R/summary.txt"
import json,sys,statistics as st
rows=[json.loads(x) for x in open(sys.argv[1])] if __import__('os').path.exists(sys.argv[1]) else []
dec=[r for r in rows if r.get('phase')=='c1-decode']
by={}
for r in dec: by.setdefault(r['kind'],[]).append(r)
parts=[f"{k} {st.median(x['tps'] for x in v):.1f} t/s (acc {st.median([x['accept_rate'] for x in v if x['accept_rate'] is not None] or [0]):.2f})" for k,v in by.items()]
deps=[f"{r['target']//1024}k: pf {r['engine_prefill_tps']:.0f} t/s, dec {r['decode_tps']:.1f}" for r in rows if r.get('phase')=='depth']
print(f"{sys.argv[2]} c1: " + '; '.join(parts) + (' | ' + '; '.join(deps) if deps else ''))
PY
  if ((rc != 0)); then note "$tag PROBE FAILED rc=$rc: $(grep -a 'Error' "$D/c1.log" | tail -1)"; fi
  return "$rc"
}
teardown(){  # tag [graceful]
  local D=$R/$1
  if [[ "${2:-}" == graceful ]]; then sudo -n docker stop -t 120 glm53 >/dev/null 2>&1 || true; fi
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1 || true
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
}

DEC2="--runs 2 --kinds code,prose,chat"
if ! run_arm A "$DEC2 --depths 8192,32768,131072,245760" DRAFT=1 CPU_THREADS=8; then
  teardown A
  N=112
  run_arm A112 "$DEC2 --depths 8192,32768,131072,245760" DRAFT=1 CPU_THREADS=8 || { teardown A112; note 'ABORT: no 256k boot at N=104 or 112'; exit 4; }
  BASE=A112; teardown A112
else
  BASE=A; teardown A
fi
run_arm B "$DEC2" DRAFT=0 CPU_THREADS=8; teardown B
run_arm C "$DEC2" DRAFT=1 CPU_THREADS=16; teardown C
run_arm D "$DEC2" DRAFT=1 CPU_THREADS=7; teardown D
run_arm E "--runs 1 --kinds code,prose,chat --depths 32768" DRAFT=1 CPU_THREADS=8 PINNED_ARENA=1; teardown E
run_arm F "--runs 1 --kinds code,prose,chat --depths 32768" DRAFT=1 CPU_THREADS=8 GLM_IMG=tabbyapi:r859-route-trace-r1 \
  PROFILE=1 ROUTE_TRACE_DIR="$R/F/route-traces"; teardown F graceful
ls "$R/F/route-traces" 2>/dev/null | wc -l | xargs -I{} note "F route-trace files: {}"

# Pick the fastest decode arm (mean over kinds of the per-kind median c1 rate) among A/A112..E and relaunch it.
BEST=$(python3 - "$R" "$BASE" <<'PY'
import json,sys,os,statistics as st
R,base=sys.argv[1:]
best=None
for tag in [base,'B','C','D','E']:
    p=f'{R}/{tag}/c1.jsonl'
    if not os.path.exists(p): continue
    dec=[json.loads(x) for x in open(p)]; dec=[r for r in dec if r.get('phase')=='c1-decode']
    if not dec: continue
    by={}
    for r in dec: by.setdefault(r['kind'],[]).append(r['tps'])
    score=st.mean(st.median(v) for v in by.values())
    if best is None or score>best[1]: best=(tag,score)
print(best[0] if best else '')
PY
)
note "best c1 arm: ${BEST:-none}"
if [[ "$KEEP_GLM" == 1 && -n "$BEST" ]]; then
  declare -A ENV_OF=([A]="DRAFT=1 CPU_THREADS=8" [A112]="DRAFT=1 CPU_THREADS=8" [B]="DRAFT=0 CPU_THREADS=8"
    [C]="DRAFT=1 CPU_THREADS=16" [D]="DRAFT=1 CPU_THREADS=7" [E]="DRAFT=1 CPU_THREADS=8 PINNED_ARENA=1")
  # shellcheck disable=SC2086
  run_arm "SERVE-$BEST" "--runs 1 --kinds code" ${ENV_OF[$BEST]} || { teardown "SERVE-$BEST"; note 'serving relaunch failed'; }
  sudo -n docker logs glm53 > "$R/SERVE-$BEST/engine-boot.log" 2>&1 || true
  note "SERVING: $BEST config on :8029 (256k context)"
fi
exit 0

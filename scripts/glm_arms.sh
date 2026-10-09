# glm_arms.sh: shared helpers for GLM codex-round GPU units (2026-10-08, from R902). Source after setting:
#   UNIT (e.g. r903), R (results dir), HERE (tools dir), PK (repo copy with launch-glm53.sh, glm53_plan.py,
#   glm53_probe.py, mtp_steps.py), IMG (overlay image), MODEL. The caller sources /srv/qwen5090/lib/gpu-queue.sh first
#   (registration BEFORE any wait, so a queued unit counts as live), then this file, then calls arms_init.
# Provides: note, boot, up <tag> "<EXL3_EXTRA flags>" env..., fp, c1spd, c4spd, compare <ref> <tags...>, and an EXIT
# cleanup that skips the daily restore while another unit is live in the GPU queue (the last unit restores).
log(){ printf '%s [%s] %s\n' "$(date -Iseconds)" "$UNIT" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
STATIC_BASE='PLACEMENT=static SWAP_MODE= SWAP_POLICY= SWAP_INIT_STATS= SWAP_CADENCE= SWAP_INTERVAL=0 SWAP_MAX= SWAP_SCOPE= SWAP_FLOOR=0 SWAP_HYST='
serving_line(){
  local f L
  for f in $(ls -t /srv/qwen5090/results/*r89[0-9]*/summary.txt /srv/qwen5090/results/*r9[0-9][0-9]*/summary.txt \
                  /srv/qwen5090/results/*r885d*/summary.txt 2>/dev/null); do
    L=$(grep -h '^SERVING on :8029: ' "$f" | tail -1); [[ -n "$L" ]] && { echo "$L"; return; }
  done
}
boot(){  # dir tools env...
  local d=$1 tools=$2; shift 2
  # shellcheck disable=SC2086
  timeout -k 20 1500 "${CLEAN[@]}" HOME="$HOME" PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 MAX_BATCH=4 \
    SYSMEM_RC_MB=1024 GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 RUN_DIR="$d" "$@" bash "$tools/launch-glm53.sh" > "$d.log" 2>&1
}
# Only the unit that took the GPU lock may touch the glm53 container on exit: a unit stopped while still queued, or one
# exiting early (e.g. R909 after a failed self-test), would otherwise rm -f another unit's booting arm (2026-10-08).
GLM_ARMS_LOCKED=0
eval "$(declare -f gpu_lock | sed '1s/^gpu_lock/_gq_gpu_lock/')"
gpu_lock(){ _gq_gpu_lock "$@" && GLM_ARMS_LOCKED=1; }
arms_cleanup(){
  local rc=$? others
  trap - EXIT INT TERM HUP
  if [[ "$GLM_ARMS_LOCKED" != 1 ]]; then
    rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; echo "exit=$rc (never held the GPU lock)" > "$R/last.txt"; exit "$rc"
  fi
  sudo -n docker logs glm53 > "$R/last-engine.log" 2>&1 || true
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1 || [[ "$(sudo -n docker inspect glm53 --format '{{.Config.Image}}' 2>/dev/null)" == "$IMG" ]]; then
    sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
    others=$(gpu_queue_others)
    if [[ -n "$others" ]]; then
      note "daily restore skipped: next unit queued ($others)"
    else
      # Re-read the daily at restore time: a promotion while this unit ran must not be undone (2026-10-08).
      local DAILY=$DENV DT=$DTOOLS
      [[ -s "$GLM_DAILY_FILE" ]] && { DAILY=$(<"$GLM_DAILY_FILE"); DT=$GLM_DAILY_TOOLS; }
      note "restoring the daily: $DAILY"
      # shellcheck disable=SC2086
      boot "$R/exit-serve" "$DT" $DAILY && note "SERVING on :8029: restored ($DAILY)" || note "exit relaunch FAILED"
    fi
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  echo "exit=$rc" > "$R/last.txt"
  exit "$rc"
}
GLM_DAILY_FILE=/srv/qwen5090/glm-daily.env      # the daily (repo flan/r858/glm-daily.env, install-glm-daily.sh)
GLM_DAILY_TOOLS=/srv/qwen5090/glm-daily-tools
arms_init(){  # resolve the daily (= experiment base), overlay the tools, install cleanup
  if [[ -s "$GLM_DAILY_FILE" ]]; then
    DENV=$(<"$GLM_DAILY_FILE"); DTOOLS=$GLM_DAILY_TOOLS
  else
    LS=$(serving_line); DENV=$(sed -E 's/.*\((.*)\)$/\1/' <<<"$LS")
    [[ "$LS" == *"("*")" && -n "$DENV" ]] || { note "no SERVING line found; stop"; exit 2; }
    DTOOLS=$(grep -oE '/srv/qwen5090/r[0-9a-z]+-tools' <<<"$DENV" | head -1); DTOOLS=${DTOOLS:-/srv/qwen5090/r885c-tools}
  fi
  # (R891-era rule "INDEX_RING=1 -> /srv/qwen5090/r891-tools" removed 2026-10-09: the daily is a ring image since R929c and
  #  glm-daily-tools is current; r891-tools has a glm53_verify that rejects per-device CPU splits.)
  STATIC="$STATIC_BASE SPLIT_STATS=$DTOOLS/split-stats-broad-r869.json"
  trap arms_cleanup EXIT
  trap 'note interrupted; exit 130' INT
  trap 'note terminated; exit 143' TERM HUP
  rm -rf "$HERE"; cp -a "$DTOOLS" "$HERE"
  cp "$PK/launch-glm53.sh" "$PK/glm53_plan.py" "$PK/glm53_probe.py" "$PK/mtp_steps.py" "$HERE/" && bash -n "$HERE/launch-glm53.sh" || { note "tools overlay FAILED"; exit 2; }
}
up(){  # tag, EXL3_EXTRA flags, env...  (static placement always; env after it wins)
  local tag=$1 extra=$2; shift 2; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $* EXL3_EXTRA=[$extra]"
  # shellcheck disable=SC2086
  boot "$D/boot" "$HERE" $DENV $STATIC GLM_IMG=$IMG INDEX_RING=0 CACHE_TOKENS=262144 "$@" EXL3_EXTRA="$extra" \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|ABORT[^;]*|Error[^;]{0,120}' "$D"/boot*.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  sudo -n docker inspect glm53 --format '{{json .Config.Env}}' | tr ',' '\n' | grep -E 'EXL3_' > "$D/container-env.txt"
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  note "$tag up: workers $(grep -aoE 'CPU MoE worker started: [0-9]+ layers' "$D/engine-boot.log" | sort | uniq -c | tr '\n' ';')"
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1
}
fp(){  # tag: greedy fingerprint, five kinds, fixed salt
  timeout -k 15 1800 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/$1/fp.jsonl" \
    --phase c1 --runs 1 --kinds code,prose,chat,html,edit --salt "${UNIT}fp" > "$R/$1/fp.log" 2>&1 || note "$1 fp rc=$?"
}
c1spd(){  # tag
  local D=$R/$1
  timeout -k 15 3600 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/c1.jsonl" \
    --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/c1.log" 2>&1 || note "$1 c1 rc=$?"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$1" | tee -a "$R/summary.txt"
}
c4spd(){  # tag
  local D=$R/$1
  timeout -k 15 2400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/dec.jsonl" \
    --phase decode --concurrency 4 --runs 2 --distinct > "$D/dec.log" 2>&1 || note "$1 decode rc=$?"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  note "$1 c4 distinct: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$D/dec.jsonl" 2>/dev/null | tail -1)"
}
compare(){  # ref tags...: greedy fingerprint identity per kind
  python3 - "$R" "$@" <<'PY' | tee -a "$R/summary.txt"
import json, sys, pathlib
R = pathlib.Path(sys.argv[1]); ref, tags = sys.argv[2], sys.argv[3:]
def text(tag, kind):
    p = R / tag / f'c1-{kind}-r0.events.jsonl'
    if not p.exists(): return None
    s = ''
    for l in open(p):
        try: e = json.loads(l)['event']
        except Exception: continue
        for ch in e.get('choices') or []:
            dl = ch.get('delta') or {}
            s += (dl.get('reasoning_content') or '') + (dl.get('content') or '')
    return s
for tag in tags:
    out, same = [], True
    for kind in ('code', 'prose', 'chat', 'html', 'edit'):
        a, b = text(ref, kind), text(tag, kind)
        if a is None or b is None: out.append(f'{kind}:missing'); same = False; continue
        i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
        ok = i is None and len(a) == len(b); same &= ok
        out.append(f'{kind}:' + ('identical' if ok else f'diverges@{i if i is not None else min(len(a), len(b))}/{len(a)}'))
    print(f'fingerprint {tag} vs {ref}: {"IDENTICAL" if same else "DIFFERS"} ' + ' '.join(out))
PY
}

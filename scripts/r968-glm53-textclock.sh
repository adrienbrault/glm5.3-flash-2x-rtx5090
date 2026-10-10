#!/usr/bin/env bash
# R968: daily vs text-owned clock; diagnostic B, then FRESH boots A B B A.
# Install the complete out/ directory at /srv/qwen5090/r968 (preserve textclock/).
# In this repository: this script and textclock_summary.py are in scripts/, glm_arms.sh in scripts/,
# Dockerfile.textclock is docker/textclock-r1/Dockerfile and textclock/ is docker/textclock-r1/textclock/.
# sudo -n systemd-run --unit=r968-glm53-textclock-$(date +%H%M%S) --uid="$(id -un)" \
#   -p RuntimeMaxSec=43200 -p TimeoutStopSec=600 -p KillMode=mixed \
#   --setenv=HOME="$HOME" /usr/bin/bash /srv/qwen5090/r968/r968-glm53-textclock.sh
# Expected wall time: 50-70 min excluding queue/drain; six boots incl. restore ~13.8 min
# (each ~2.3 min), workloads ~35-50 min at c4 75-95 and c1 30-50 tok/s, build ~1-3 min.
# No promotion. Always restore the current daily at exit, including failure/interruption.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UNIT=r968
MODEL=glm53-flash-exl3-2.05bpw-turboderp
EXPECTED_BASE=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1
IMG=${EXPECTED_BASE}_textclock1
R=/srv/qwen5090/results/$(date -u +%F)-r968-glm53-textclock-$(date -u +%H%M%S)
mkdir -p "$R"
HERE=$R/tools
PK=/srv/qwen5090/glm-daily-tools
export FLAN_POWER=0 GPU_QUEUE_NAME=r968-glm53-textclock
source /srv/qwen5090/lib/gpu-queue.sh
source /srv/qwen5090/lib/gateway-drain.sh
source "$P/glm_arms.sh"
TOUCHED=0
# Override the helper's queued-unit optimization: this brief requires a daily restore.
arms_cleanup(){
  local rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  set +e
  if [[ "$GLM_ARMS_LOCKED" == 1 && "$TOUCHED" == 1 ]]; then
    sudo -n docker logs glm53 > "$R/last-engine.log" 2>&1
    sudo -n docker rm -f glm53 > "$R/remove-final.log" 2>&1
    local daily
    daily=$(cat "$GLM_DAILY_FILE")
    if [[ -n "$daily" ]]; then
      note "restoring current daily: $daily"
      # shellcheck disable=SC2086
      boot "$R/exit-serve" "$GLM_DAILY_TOOLS" $daily FLAN_POWER=0
      restore_rc=$?
    else
      restore_rc=1
    fi
    if [[ "$restore_rc" == 0 ]]; then
      note "SERVING on :8029: restored ($daily)"
    else
      note "FATAL daily restore failed rc=$restore_rc (exit-serve.log)"
      rc=3
    fi
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  printf 'exit=%s restore_rc=%s results=%s\n' "$rc" "$restore_rc" "$R" > "$R/last.txt"
  exit "$rc"
}
trap arms_cleanup EXIT
trap 'note interrupted; exit 130' INT
trap 'note terminated; exit 143' TERM HUP
note "Expected 50-70 min excluding queue/drain; results $R; waiting for GPU lock"
gpu_lock || exit 2
flock -n 9 || { note "FATAL GPU lock fd9 missing"; exit 2; }
[[ -s "$GLM_DAILY_FILE" ]] || { note "FATAL daily env missing"; exit 2; }
arms_init
read -r -a DAILY_WORDS <<< "$DENV"
BASE= EXTRA=
for kv in "${DAILY_WORDS[@]}"; do
  case "$kv" in GLM_IMG=*) BASE=${kv#*=};; EXL3_EXTRA=*) EXTRA=${kv#*=};; esac
done
[[ "$BASE" == "$EXPECTED_BASE" ]] || { note "FATAL daily base differs: $BASE"; exit 2; }
[[ "$DENV" != *EXL3_MOE_CPU_SWAP_TEXT_CLOCK* && "$DENV" != *EXL3_MOE_CPU_SWAP_DEBUG* ]] \
  || { note "FATAL daily already overrides clock/debug; A would not be the reviewed baseline"; exit 2; }
printf '%s\n' "$DENV" > "$R/daily.env"
gateway_drain || { note "FATAL gateway drain failed"; exit 2; }
gateway_wait_idle 900 | tee -a "$R/summary.txt" || { note "FATAL gateway did not idle"; exit 2; }
sudo -n docker image inspect "$BASE" > "$R/base-image.json"
# All build inputs are local; never pull or use a build network.
timeout -k 30 900 sudo -n docker build --network=none --pull=false --build-arg "BASE=$BASE" \
  -f "$P/Dockerfile.textclock" -t "$IMG" "$P" > "$R/build.log" 2>&1 \
  || { note "FATAL overlay build failed (build.log)"; exit 2; }
note "built $IMG FROM $BASE with --network=none"

fresh_boot(){  # tag image text_clock debug; no probe warmup or c1 before c4 cold
  local tag=$1 image=$2 clock=$3 debug=$4 D=$R/$1
  local extra="$EXTRA"
  local overrides=("GLM_IMG=$image" FLAN_POWER=0)
  mkdir -p "$D"
  if [[ "$clock" == 1 ]]; then
    extra="${extra:+$extra;}EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1;EXL3_MOE_CPU_SWAP_DEBUG=$debug"
    overrides+=("EXL3_EXTRA=$extra")
  fi
  TOUCHED=1
  sudo -n docker rm -f glm53 > "$D/remove.log" 2>&1 || true
  sleep 3
  note "$tag fresh boot image=$image text_clock=$clock debug=${debug:-off}"
  boot "$D/boot" "$HERE" "${DAILY_WORDS[@]}" "${overrides[@]}" \
    || { note "FATAL $tag boot failed"; return 1; }
  sudo -n docker inspect glm53 > "$D/container.json"
  python3 - "$D/container.json" "$image" "$clock" "$debug" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))[0]['Config']; e=dict(x.split('=',1) for x in c['Env'])
# The GLM launcher passes EXL3_* as `/usr/bin/env [-u NAME]... NAME=VALUE... python3 main.py` arguments (Config.Cmd),
# not as docker -e (R968 first run 2026-10-10 failed here): merge them, honouring -u.
cmd=list(c.get('Cmd') or []); i=0
while i<len(cmd):
    if cmd[i]=='-u': e.pop(cmd[i+1],None); i+=2; continue
    if '=' not in cmd[i]: break
    k,v=cmd[i].split('=',1); e[k]=v; i+=1
assert c['Image']==sys.argv[2], c['Image']
assert e.get('EXL3_MOE_CPU_SWAP_TEXT_CLOCK','0')==sys.argv[3], e
assert bool(e.get('EXL3_MOE_CPU_SWAP_DEBUG',''))==bool(sys.argv[4]), e
assert e['EXL3_MOE_CPU_SWAP_MODE']=='exchange'
assert e.get('EXL3_MOE_CPU_SWAP_POLICY','histogram')=='histogram'
assert e.get('EXL3_MOE_CPU_SWAP_CADENCE','exact')=='exact'
PY
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
}
c4(){
  local tag=$1 phase=$2 runs=${3:-2} D=$R/$1
  timeout -k 15 2400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/$phase.jsonl" --phase decode --concurrency 4 --runs "$runs" --distinct > "$D/$phase.log" 2>&1
}
# Short DEBUG boot is wholly discarded before A1; one distinct c4 round (4096 tokens).
fresh_boot Bdebug "$IMG" 1 1
sudo -n docker logs glm53 > "$R/Bdebug/engine-before-c4.log" 2>&1
c4 Bdebug c4-debug 1
sudo -n docker logs glm53 > "$R/Bdebug/engine.log" 2>&1
python3 - "$R/Bdebug" <<'PY' | tee -a "$R/summary.txt"
from pathlib import Path
import re, sys
p=Path(sys.argv[1]); before=(p/'engine-before-c4.log').read_text(); after=(p/'engine.log').read_text()
assert after.startswith(before), 'container log prefix changed'
window=after[len(before):]; (p/'engine-c4-window.log').write_text(window)
owners=[s for s in before.splitlines() if '[SWAP-OWNER]' in s]
assert owners and 'component=text' in owners[-1], 'missing text startup owner'
sweeps=window.count(' -- exchange sweep:')
clock_sweeps=sum('[SWAP-CLOCK]' in s and 'event=sweep' in s for s in window.splitlines())
assert sweeps>0 and sweeps==clock_sweeps, f'c4 sweeps={sweeps} clock_sweeps={clock_sweeps}'
assert 'FATAL' not in after, 'FATAL in debug engine'
print('Bdebug '+owners[-1]); print(f'Bdebug c4 sweeps={sweeps} clock_sweeps={clock_sweeps}')
PY
arm(){
  local tag=$1 image=$2 clock=$3
  fresh_boot "$tag" "$image" "$clock" ''
  c4 "$tag" c4-cold
  timeout -k 15 3600 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$R/$tag/c1.jsonl" --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$R/$tag/c1.log" 2>&1
  c4 "$tag" c4-after
  fp "$tag"  # glm_arms greedy fixed salt, five kinds; check its logged exit too
  if grep -Fq "$tag fp rc=" "$R/summary.txt"; then
    note "FATAL $tag fingerprint command failed"; return 1
  fi
  [[ -s "$R/$tag/fp.jsonl" ]] || { note "FATAL $tag missing fingerprint"; return 1; }
  sudo -n docker logs glm53 > "$R/$tag/engine.log" 2>&1
  note "$tag measurements complete"
}
arm A1 "$BASE" 0
arm B1 "$IMG" 1
arm B2 "$IMG" 1
arm A2 "$BASE" 0
python3 "$P/textclock_summary.py" "$R" | tee -a "$R/summary.txt"
note "R968 complete; gate result above; daily restore follows"

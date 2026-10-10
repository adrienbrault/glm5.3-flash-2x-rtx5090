#!/usr/bin/env bash
# R969: c1 FIRST on the served textclock image; diagnostic B, then A1 B1 B2 A2 B3 A3.
# Authoring only: no automatic promotion or rollback.
# Install this script, glm_arms.sh and r969_summary.py at /srv/qwen5090/r969/.
# In this repository: this script, r969_summary.py and glm_arms.sh are in scripts/; the overlay is docker/textclock-r1/.
# sudo -n systemd-run --unit=r969-glm53-textclock-$(date +%H%M%S) --uid="$(id -un)" \
#   -p RuntimeMaxSec=43200 -p TimeoutStopSec=5400 -p KillMode=mixed \
#   --setenv=HOME="$HOME" /usr/bin/bash /srv/qwen5090/r969/r969-glm53-textclock-c1.sh
# Expected ~85-115 min excluding queue/drain: eight boots including debug/restore,
# six times c1/c2/c3/c4, plus diagnostic c1/c2. Uses the installed image; no build.
# No promotion. Always restore the current daily at exit, including failure/interruption.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UNIT=r969
MODEL=glm53-flash-exl3-2.05bpw-turboderp
EXPECTED_BASE=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1
IMG=${EXPECTED_BASE}_textclock1
R=/srv/qwen5090/results/$(date -u +%F)-r969-glm53-textclock-$(date -u +%H%M%S)
mkdir -p "$R"
HERE=$R/tools
PK=/srv/qwen5090/glm-daily-tools
export FLAN_POWER=0 GPU_QUEUE_NAME=r969-glm53-textclock-c1
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
      rm -f "$R/gate.json"  # fail closed if restoration failed after reporting
    fi
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  printf 'exit=%s restore_rc=%s results=%s\n' "$rc" "$restore_rc" "$R" > "$R/last.txt"
  exit "$rc"
}
trap arms_cleanup EXIT
trap 'note interrupted; exit 130' INT
trap 'note terminated; exit 143' TERM HUP
note "Expected 85-115 min excluding queue/drain; results $R; waiting for GPU lock"
gpu_lock || exit 2
flock -n 9 || { note "FATAL GPU lock fd9 missing"; exit 2; }
[[ -s "$GLM_DAILY_FILE" ]] || { note "FATAL daily env missing"; exit 2; }
arms_init
printf '%s\n' "$DENV" > "$R/daily.env"
# ref/glm-daily.env is a historical pre-promotion snapshot. Require the live served
# daily rather than silently converting that snapshot into B.
python3 "$P/r969_summary.py" --daily "$R/daily.env" \
  || { note "FATAL daily is not the reviewed textclock daily"; exit 2; }
read -r -a DAILY_WORDS <<< "$DENV"
EXTRA=
for kv in "${DAILY_WORDS[@]}"; do
  case "$kv" in EXL3_EXTRA=*) EXTRA=${kv#*=};; esac
done
# Strip only the clock/debug entries; preserve every other served extra.
ARM_EXTRA=
IFS=';' read -r -a EXTRA_WORDS <<< "$EXTRA"
for kv in "${EXTRA_WORDS[@]}"; do
  case "$kv" in EXL3_MOE_CPU_SWAP_TEXT_CLOCK=*|EXL3_MOE_CPU_SWAP_DEBUG=*|'') continue;; esac
  ARM_EXTRA="${ARM_EXTRA:+$ARM_EXTRA;}$kv"
done
gateway_drain || { note "FATAL gateway drain failed"; exit 2; }
gateway_wait_idle 900 | tee -a "$R/summary.txt" || { note "FATAL gateway did not idle"; exit 2; }
sudo -n docker image inspect "$IMG" > "$R/image.json"
: > "$R/boots.jsonl"
: > "$R/phases.jsonl"

fresh_boot(){  # tag clock debug; no probe warmup before c1
  local tag=$1 clock=$2 debug=$3 D=$R/$1
  local extra="${ARM_EXTRA:+$ARM_EXTRA;}EXL3_MOE_CPU_SWAP_TEXT_CLOCK=$clock;EXL3_MOE_CPU_SWAP_DEBUG=$debug"
  mkdir -p "$D"
  TOUCHED=1
  sudo -n docker rm -f glm53 > "$D/remove.log" 2>&1 || true
  sleep 3
  note "$tag fresh boot image=$IMG text_clock=$clock debug=${debug:-off}"
  boot "$D/boot" "$HERE" "${DAILY_WORDS[@]}" GLM_IMG="$IMG" FLAN_POWER=0 "EXL3_EXTRA=$extra" \
    || { note "FATAL $tag boot failed"; return 1; }
  sudo -n docker inspect glm53 > "$D/container.json"
  python3 "$P/r969_summary.py" --container "$D/container.json" "$clock" "$debug" \
    || { note "FATAL $tag container environment differs"; return 1; }
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  python3 - "$D/container.json" "$R/boots.jsonl" "$tag" <<'PYBOOT'
import json, sys
c=json.load(open(sys.argv[1]))[0]
with open(sys.argv[2], 'a') as f:
    f.write(json.dumps(dict(tag=sys.argv[3], container_id=c['Id']))+'\n')
PYBOOT
}
phase(){
  local tag=$1 phase=$2 D=$R/$1/$2
  mkdir -p "$D"
  if [[ "$phase" == c1 ]]; then
    timeout -k 15 3600 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
      --out "$D/samples.jsonl" --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/probe.log" 2>&1
  else
    timeout -k 15 2400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
      --out "$D/samples.jsonl" --phase decode --concurrency "${phase#c}" --runs 2 --distinct > "$D/probe.log" 2>&1
  fi
  # Completion record makes ordering/failure checks independent of mtimes.
  printf '{"tag":"%s","phase":"%s"}\n' "$tag" "$phase" >> "$R/phases.jsonl"
}
# This diagnostic boot has its own c1/c2 workloads/log windows and is discarded.
fresh_boot Bdebug 1 1
for p in c1 c2; do
  sudo -n docker logs glm53 > "$R/Bdebug/engine-before-$p.log" 2>&1
  phase Bdebug "$p"
  sudo -n docker logs glm53 > "$R/Bdebug/engine-after-$p.log" 2>&1
  python3 "$P/r969_summary.py" --debug-window "$R/Bdebug" "$p" | tee -a "$R/summary.txt"
done
sudo -n docker logs glm53 > "$R/Bdebug/engine.log" 2>&1
arm(){
  local tag=$1 clock=$2 p
  fresh_boot "$tag" "$clock" ''
  for p in c1 c2 c3 c4; do phase "$tag" "$p"; done
  sudo -n docker logs glm53 > "$R/$tag/engine.log" 2>&1
  note "$tag measurements complete"
}
arm A1 0
arm B1 1
arm B2 1
arm A2 0
arm B3 1
arm A3 0
# A valid non-KEEP decision still restores daily and preserves its distinct exit code.
summary_rc=0
python3 "$P/r969_summary.py" "$R" | tee -a "$R/summary.txt" || summary_rc=$?
note "R969 complete; summary rc=$summary_rc; daily restore follows; operator decides any rollback"
exit "$summary_rc"

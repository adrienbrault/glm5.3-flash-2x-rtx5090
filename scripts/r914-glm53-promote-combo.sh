#!/usr/bin/env bash
# R914 (2026-10-08): promote the combined daily (user: "apply all the good things together", "promote as new daily/base
# for measurements") and measure it. /srv/qwen5090/glm-daily.env (repo flan/r858/glm-daily.env):
#   MTP depth 1 + MTP_FAST + EXL3_MTP_MAX_BATCH=1 (MTP only with one active request; R902/R911) + VISION_OFFLOAD=1 (R901)
#   + the four MTP host-overhead flags (R902b: 29.82 vs 29.96/30.60 ms/step, inside boot spread, lossless, greedy-only
#   accept) at N=104 on the daily's dynamic exchange placement, image cheapswap-r3-agent-r2_mtpfast1_overhead-r2.
#   C0 combo without the four flags; CF the full daily; C1 = C0 again (sandwich). Each: c1 five kinds x2 + server step,
#   distinct c1/c2/c4 x2, vision; CF also the thinking/tool sanity. Leaves the daily (file) serving if nothing is queued.
#   sudo systemd-run --unit=r914-glm53-promote-combo --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r914-glm53-promote-combo.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r914
PK=/srv/qwen5090/r914
HERE=/srv/qwen5090/r914-tools
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r914-glm53-promote-combo-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r914-glm53-promote-combo
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
gpu_lock
note "GPU lock held; results $R; daily $DENV"
updyn(){  # tag, env...: the daily file's env, env after it wins
  local tag=$1; shift; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  # shellcheck disable=SC2086
  boot "$D/boot" "$HERE" $DENV "$@" \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|ABORT[^;]*|Error[^;]{0,120}' "$D"/boot*.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  note "$tag landed: $(grep -aoE '\[MTP-OVERHEAD\] landed[^"]*' "$D/engine-boot.log" | head -1 | cut -c1-160)"
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1
  note "$tag free MiB after warmup: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
}
measure(){  # tag
  local D=$R/$1
  c1spd "$1"
  note "$1 server step: $(python3 "$HERE/mtp_steps.py" "$D/engine.log" 2>&1 | tail -1 | cut -c1-200)"
  timeout -k 15 3000 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/dec.jsonl" \
    --phase decode --concurrency 1,2,4 --runs 2 --distinct > "$D/dec.log" 2>&1 || note "$1 decode rc=$?"
  note "$1 decode distinct: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$D/dec.jsonl" 2>/dev/null | tail -1)"
  timeout -k 15 900 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/vision.jsonl" \
    --phase vision > "$D/vision.log" 2>&1; note "$1 vision rc=$? $(tail -1 "$D/vision.log" | cut -c1-160)"
}
updyn C0 "EXL3_EXTRA=EXL3_MTP_MAX_BATCH=1" && measure C0
updyn CF && { measure CF
  timeout -k 15 900 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$R/CF/sanity.jsonl" \
    --phase sanity > "$R/CF/sanity.log" 2>&1; note "CF sanity rc=$? $(tail -1 "$R/CF/sanity.log" | cut -c1-160)"; }
updyn C1 "EXL3_EXTRA=EXL3_MTP_MAX_BATCH=1" && measure C1
exit 0

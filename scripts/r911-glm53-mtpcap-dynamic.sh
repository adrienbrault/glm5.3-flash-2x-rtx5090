#!/usr/bin/env bash
# R911 (2026-10-08): the serving candidate R902 pointed to. R902 (static, N=104): MTP-fast with EXL3_MTP_MAX_BATCH=1 (MTP only
# when one request is active) gave c1 60.4 vs plain 52.7 (+14.6 %) and c4 74.6 vs 74.5 (penalty gone). But static N=104
# is far below the served daily (dynamic exchange, N=96: c1 ~57-60, distinct c4 92-100), so the candidate must run on the
# daily's dynamic placement. VISION_OFFLOAD=1 (R901: frees 316 MiB GPU0, vision identical) makes room for the MTP layer.
#   D0 the daily exactly as served; M MTP-cap on the daily's dynamic placement + VISION_OFFLOAD=1 at the lowest N in
#   96..104 that boots; D1 the daily again (c1 noise sandwich). Each: c1 five kinds x2 (+ server MTP step for M), distinct
#   c1/c4 x2, three vision requests. Restores the daily at the end (serving MTP is the user's decision).
#   sudo systemd-run --unit=r911-glm53-mtpcap-dynamic --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r911-glm53-mtpcap-dynamic.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r911
PK=/srv/qwen5090/r911
HERE=/srv/qwen5090/r911-tools
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r911-glm53-mtpcap-dynamic-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r911-glm53-mtpcap-dynamic
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
gpu_lock
note "GPU lock held; results $R; daily $DENV"
updyn(){  # tag, EXL3_EXTRA, env...: the daily's own (dynamic exchange) placement, env after it wins
  local tag=$1 extra=$2; shift 2; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $* EXL3_EXTRA=[$extra]"
  # shellcheck disable=SC2086
  boot "$D/boot" "$HERE" $DENV "$@" ${extra:+EXL3_EXTRA="$extra"} \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|ABORT[^;]*|Error[^;]{0,120}' "$D"/boot*.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  note "$tag up: VRAM $(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tr '\n' ' ')"
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1
}
measure(){  # tag
  local D=$R/$1
  c1spd "$1"
  note "$1 server step: $(python3 "$HERE/mtp_steps.py" "$D/engine.log" 2>&1 | tail -1 | cut -c1-200)"
  timeout -k 15 2400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/dec.jsonl" \
    --phase decode --concurrency 1,4 --runs 2 --distinct > "$D/dec.log" 2>&1 || note "$1 decode rc=$?"
  note "$1 decode distinct: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$D/dec.jsonl" 2>/dev/null | tail -1)"
  timeout -k 15 900 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/vision.jsonl" \
    --phase vision > "$D/vision.log" 2>&1; note "$1 vision rc=$? $(tail -1 "$D/vision.log" | cut -c1-160)"
}
updyn D0 "" && measure D0
ok=""
for n in 96 98 100 102 104; do
  updyn "M-n$n" "EXL3_MTP_MAX_BATCH=1" GLM_IMG=$IMG DRAFT=1 DRAFT_N=1 MTP_FAST=1 VISION_OFFLOAD=1 OFFLOAD_N=$n && { ok=$n; break; }
done
[[ -n "$ok" ]] && measure "M-n$ok"
updyn D1 "" && measure D1
exit 0

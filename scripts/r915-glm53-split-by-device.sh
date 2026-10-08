#!/usr/bin/env bash
# R915 (2026-10-08): per-device CPU expert count. The layer split assigns whole layers and every layer used one N, so with
# MTP at N=104 GPU0 kept ~1.35 GB unused (GPU1, which holds the MTP layer, 365 MiB). Patch split-by-device-r1:
# EXL3_MOE_CPU_SPLIT_BY_DEVICE="N0,N1" sets N per CUDA device (Python only, image ..._overhead-r2_splitdev1).
#   D the daily (glm-daily.env); then N0 in 94/96/98/100 with N1=104: lowest that boots and keeps >= 450 MiB free on GPU0
#   (GPU1 >= 300, the daily itself has 365) after warmup -> c1 five kinds x2 + server step, distinct c1/c2/c4 x2, vision. Replaces R913 (GPU_SPLIT budget).
#   sudo systemd-run --unit=r915-glm53-split-by-device --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r915-glm53-split-by-device.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r915
PK=/srv/qwen5090/r915
HERE=/srv/qwen5090/r915-tools
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev1
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r915-glm53-split-by-device-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r915-glm53-split-by-device
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
sudo -n docker image inspect "$IMG" >/dev/null 2>&1 || sudo -n docker build -t "$IMG" "$PK/split-by-device-r1" > "$R/build.log" 2>&1 \
  || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-200)"; exit 2; }
gpu_lock
note "GPU lock held; results $R; daily $DENV"
DX=$(grep -oE 'EXL3_EXTRA=[^ ]+' <<<"$DENV" | head -1); DX=${DX#EXL3_EXTRA=}
updyn(){  # tag, env...: the daily's env, env after it wins; headroom check after warmup
  local tag=$1; shift; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  # shellcheck disable=SC2086
  boot "$D/boot" "$HERE" $DENV "$@" \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|ABORT[^;]*|Error[^;]{0,120}' "$D"/boot*.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1 \
    || { note "$tag warmup FAILED: $(tail -1 "$D/warmup.log" | cut -c1-160)"; return 1; }
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
  note "$tag free MiB after warmup: $free | CPU split per layer: $(grep -aoE 'CPU split experts[^:]*: [^ ]+ \[[0-9]+\.\.288\)' "$D/engine-boot.log" | grep -oE '\[[0-9]+' | sort | uniq -c | tr '\n' ' ')"
  # GPU0 must keep >= 450 MiB; GPU1 (unchanged at N=104, holds the MTP layer) runs at the daily's own ~365 MiB, so it
  # only has to stay >= 300. (First run rejected the daily itself on GPU1's 365 MiB.)
  local g0 g1; read -r g0 g1 <<<"$free"
  [[ "$tag" == D ]] || { ((g0 >= 450 && g1 >= 300)) || { note "$tag REJECT: headroom GPU0 $g0 / GPU1 $g1 MiB (need 450 / 300)"; return 1; }; }
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
updyn D && measure D
for n0 in 94 96 98 100; do
  updyn "S-n$n0" GLM_IMG=$IMG "EXL3_EXTRA=$DX;EXL3_MOE_CPU_SPLIT_BY_DEVICE=$n0,104" && { measure "S-n$n0"; break; }
done
exit 0

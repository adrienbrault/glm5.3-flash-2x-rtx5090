#!/usr/bin/env bash
# R870 (2026-10-07): GLM-5.3 decode vs context depth. R867's prefill check at 32k decoded at 22.7 tok/s, against ~48 at
# short context (same config, N=96 MTP off). Agent sessions sit at 30-100k tokens of context, so this curve matters more
# than another 2 % at depth 0. The CPU expert work does not depend on depth, so the loss is GPU-side (11 DSA attention
# layers with the indexer, or the KDA layers). One boot, N=96 MTP off, 256k: decode 256 tokens after cold prefill at
# 4k / 8k / 16k / 32k / 64k / 128k, then a second pass at 32k with the CPU and handoff profilers for attribution.
#   sudo systemd-run --unit=r870-glm53-depth --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r870-glm53-depth.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r869-tools
R=/srv/qwen5090/results/$(date +%F)-r870-glm53-depth-$(date +%H%M%S); mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
note(){ printf '%s [r870] %s\n' "$(date -Iseconds)" "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
MODEL=glm53-flash-exl3-2.05bpw-turboderp
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
COMMON=(PACK=A OFFLOAD_MODE=split OFFLOAD_N=96 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 SYSMEM_RC_MB=1024
        GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 DRAFT=0 CPU_THREADS=8)
export GPU_QUEUE_NAME=r870-glm53-depth
. /srv/qwen5090/lib/gpu-queue.sh
cleanup(){
  local rc=$?; trap - EXIT INT TERM HUP
  sudo -n docker logs glm53 > "$R/engine-last.log" 2>&1 || true
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  "${CLEAN[@]}" HOME="$HOME" "${COMMON[@]}" RUN_DIR="$R/serve" bash "$HERE/launch-glm53.sh" > "$R/serve.log" 2>&1 \
    && note "SERVING on :8029: N=96 MTP off, plain image" || note "serve FAILED (serve.log)"
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; echo "exit=$rc" > "$R/last.txt"; exit "$rc"
}
trap cleanup EXIT
trap 'note terminated; exit 143' TERM HUP INT
gpu_lock
note "GPU lock held; results $R"
depths(){  # tag, depths, then extra launcher env
  local tag=$1 ds=$2; shift 2
  local D=$R/$tag; mkdir -p "$D"
  sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
  "${CLEAN[@]}" HOME="$HOME" "${COMMON[@]}" RUN_DIR="$D" "$@" bash "$HERE/launch-glm53.sh" > "$D/boot.log" 2>&1 \
    || { note "$tag NO BOOT: $(tail -1 "$D/boot.log" | cut -c1-200)"; return 1; }
  timeout -k 15 5400 python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/depth.jsonl" \
    --phase c1 --runs 1 --kinds "" --depths "$ds" > "$D/depth.log" 2>&1 || note "$tag probe rc=$?"
  python3 - "$D/depth.jsonl" "$tag" <<'PY' | tee -a "$R/summary.txt"
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get('phase') == 'depth':
        print(f"{sys.argv[2]} {r['prompt_tokens']//1024}k: prefill {r['engine_prefill_tps']:.0f} t/s ({r['engine_prefill_s']:.1f} s), decode {r['decode_tps']:.1f} t/s")
PY
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1 || true
}
depths ladder 4096,8192,16384,32768,65536,131072
depths prof32k 32768 PROFILE=1
grep -aE "handoff prof|moe_cpu prof|split prof" "$R/prof32k/engine.log" | tail -12 > "$R/prof32k/prof-tail.txt"
exit 0

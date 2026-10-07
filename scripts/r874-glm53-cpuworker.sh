#!/usr/bin/env bash
# R874 (2026-10-07): CPU MoE worker microbenchmarks (codex cpu-worker-bw-r1, patches/exllamav3/cpuworker-r1). The worker
# reads ~47 GB/s during the GEMVs; a one-CCD 9800X3D tops out near 32 B x FCLK (~64 GB/s at 2000). This separates
# bandwidth from decode/issue limits (cold vs L3-hot data vs a pure-read ceiling), dissects the 47 us prep phase, and A/Bs
# the default-off flags (affinity, spin-only helpers, prefetch distance) with bit-identical outputs.
# CPU-only, but it holds the GPU lock so no decode measurement runs alongside it (and the GLM server is down meanwhile).
#   sudo systemd-run --unit=r874-glm53-cpuworker --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r874-glm53-cpuworker.sh
set -uo pipefail
: "${HOME:=/root}"
HERE=/srv/qwen5090/r874-tools
PK=/srv/qwen5090/cpuworker-r1
R=/srv/qwen5090/results/$(date +%F)-r874-glm53-cpuworker-$(date +%H%M%S); mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
note(){ printf '%s [r874] %s\n' "$(date -Iseconds)" "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
CLEAN=(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SERVE=(PACK=A OFFLOAD_MODE=split OFFLOAD_N=96 CACHE_TOKENS=262144 MAX_SEQ=262144 CHUNK=2048 MAX_BATCH=4 SYSMEM_RC_MB=1024
       GPU_SPLIT=31,31 CACHE_MODE=8,8 BOOT_TIMEOUT=1200 DRAFT=0 PLACEMENT=static SPLIT_STATS=$HERE/split-stats-broad-r869.json
       GLM_IMG=tabbyapi:r861-glm-agent-r2 AGENT=1)
export GPU_QUEUE_NAME=r874-glm53-cpuworker
. /srv/qwen5090/lib/gpu-queue.sh
cleanup(){
  local rc=$?; trap - EXIT INT TERM HUP
  if ! curl -fsS -m 5 http://127.0.0.1:8029/health >/dev/null 2>&1; then
    "${CLEAN[@]}" HOME="$HOME" "${SERVE[@]}" RUN_DIR="$R/serve" bash "$HERE/launch-glm53.sh" > "$R/serve.log" 2>&1 \
      && note "SERVING on :8029: static N=96 + agent overlay" || note "serve FAILED (serve.log)"
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; echo "exit=$rc" > "$R/last.txt"; exit "$rc"
}
trap cleanup EXIT
trap 'note terminated; exit 143' TERM HUP INT
gpu_lock
note "GPU lock held; results $R"
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 5
free -g > "$R/free.txt"
cd "$PK" || exit 1
(cd out && sha256sum -c SHA256SUMS) > "$R/sums.txt" 2>&1 || { note "packet checksum FAILED"; exit 1; }
python3 out/bench/probe_host.py --output "$R/host.json" > "$R/probe_host.log" 2>&1 || note "probe_host rc=$?"
sweep(){  # tag image extra-docker-args... -- sweep args
  local tag=$1 img=$2; shift 2; local dargs=(); while [[ $1 != -- ]]; do dargs+=("$1"); shift; done; shift
  timeout -k 30 3600 sudo -n docker run --rm --entrypoint python "${dargs[@]}" -v "$R:/results" "$img" \
    /opt/cpuworker/bench/sweep.py --output-dir "/results/$tag" "$@" > "$R/$tag.log" 2>&1
  local rc=$?; note "$tag rc=$rc"
  python3 out/bench/summarize.py "$R/$tag" > "$R/$tag.summary.txt" 2>&1 && sed 's/^/  /' "$R/$tag.summary.txt" | head -40 >> "$R/summary.txt"
}
BENCH=(-v "$PK/out/bench:/opt/cpuworker/bench:ro")
sweep base tabbyapi:r828-prompt-lookup-r3 "${BENCH[@]}" -- --variants baseline --threads 4,6,8 --layouts physical,smt --repeats 3
timeout -k 30 5400 sudo -n docker build --progress=plain -f out/r1/Dockerfile -t tabbyapi:cpuworker-r1 out > "$R/build.log" 2>&1
rc=$?; note "build cpuworker-r1 rc=$rc: $(tail -2 "$R/build.log" | tr '\n' ' ' | cut -c1-200)"
((rc == 0)) || exit 0
sweep r1 tabbyapi:cpuworker-r1 -- --variants baseline,affinity,spin,pf2,pf4,pf8 --threads 8 --layouts physical --repeats 3
sweep detail100 tabbyapi:cpuworker-r1 -- --threads 8 --layouts physical --variants baseline,spin --detail --gap-us 100 --repeats 3
sweep detailparked tabbyapi:cpuworker-r1 -- --threads 8 --layouts physical --variants baseline,spin --detail --gap-us 2000 --no-prime --repeats 3
sweep pages4k tabbyapi:cpuworker-r1 -- --threads 8 --layouts physical --variants baseline --pages 4k --repeats 3
sweep pagesthp tabbyapi:cpuworker-r1 -- --threads 8 --layouts physical --variants baseline --pages thp --repeats 3
exit 0

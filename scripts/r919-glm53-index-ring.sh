#!/usr/bin/env bash
# R919 (2026-10-08): codex-glm-index-ring-r2 = the DSA indexer ring rebased onto the daily (MTP depth 1, layer 45 in the
# ring, two-row verify/rewind) + a same-invocation SHADOW oracle (EXL3_DSA_INDEX_RING_SHADOW=1: keeps the paged k_idx
# plane, recomputes the pooled scorer inputs from both sources and compares bytes on served traffic). It replaces the
# cross-run exactness gate that R891b could not read (R894/R910b: two runs never match). Why: the ring frees ~1.37 GiB at
# 262k (2.75 at 524k); MTP-cap only boots at N=104 (R911), which costs ~9 % at c4. Image
# tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_ring2. Packet: patches/exllamav3/glm-index-ring-r2/out (OPERATOR.md).
#
# Arms (every boot = glm-daily.env + INDEX_RING=1 (launcher -> EXL3_DSA_INDEX_RING=1) + EXL3_EXTRA "<daily>;<flag>"):
#   build  after gpu_lock if missing (native rebuild, MAX_JOBS=4, torch-first landing, in-image CPU tests).
#   K      kernel tests in the image, glm53 stopped, both GPUs: test_kernels, test_graph_shadow (its deliberate-corruption
#          negative control prints one expected mismatch), test_dispatch_shadow; any FAIL stops the unit. <= 25 min.
#   SH     shadow validation <= 60 min: N=104, 262144 pool, ring + shadow; serve_workload.py --mode shadow (8k/16k/32k c1,
#          shared-prefix resumes, distinct c4, mixed prefill/decode, page-reuse churn, vision, agentic tool + resume);
#          graceful stop for the final [RING-SHADOW] summary; check_shadow.py. If N=104 does not boot with shadow's extra
#          planes, ONE labelled retry at N=108 (coverage only: the pool is never shrunk). Shadow speed is not a measurement.
#          PASS = zero mismatched bytes in every summary, every path counter > 0, 11 trunk indexer layers + MTP layer 45.
#          FAIL / not run -> no promotion arms.
#   P-nN   promotion, ring ON, shadow OFF, 262144 pool: N = 96, 97, ... 104 ascending; the first N that boots, warms up and
#          keeps >= 450 MiB free on GPU0 and >= 300 on GPU1 (the daily itself runs GPU1 at ~365) after warmup is the arm.
#          A booting N that fails warmup/headroom is recorded and the search continues. Lowest booting N and lowest
#          passing N are both noted.
#   P-n104-524k  ring ON, N=104, CACHE_TOKENS=524288 (MAX_SEQ stays at the daily's 262144: the pool grows, not the
#          per-request limit; boot.py raised both), measured even if its headroom fails (flagged).
#   Each promotion arm: c1 five kinds x2 + server step (mtp_steps), distinct c1/c2/c4 x2, vision, free VRAM per GPU after
#   warmup and after the matrix. check_promotion.py is not run (it needs serve_workload --mode promotion traffic,
#   65 min per arm); the matrix above is the daily's own (R914/R915: D distinct c1 63-66, c2 75.5, c4 84-89).
# Pass/fail: K all PASS; SH verdict passed; promotion arm boots at N < 104 with headroom and vision OK and c1/c4 not below
#   the R914/R915 daily band -> the operator writes INDEX_RING=1 GLM_IMG=<ring2 tag> OFFLOAD_N=<N> into glm-daily.env
#   (EXL3_DSA_INDEX_RING_SHADOW never). The 524k arm decides a separate pool promotion (user's call).
# PK=/srv/qwen5090/r919 must hold: glm_arms.sh launch-glm53.sh glm53_plan.py glm53_probe.py glm53_verify.py mtp_steps.py
#   (from flan/r858/) and ring2/ = the whole patches/exllamav3/glm-index-ring-r2/out/ directory (Docker context, tests/).
# Expected wall: queue + build (~30-60 min) + K ~10-25 + SH <= 60 + promotion <= 80 + restore ~3 = ~3-4 h.
#   sudo systemd-run --unit=r919-glm53-index-ring --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r919-glm53-index-ring.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r919
PK=/srv/qwen5090/r919
HERE=/srv/qwen5090/r919-tools   # the unit's OWN dir: arms_init does rm -rf "$HERE"
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_ring2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r919-glm53-index-ring-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r919-glm53-index-ring
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
cp "$PK/glm53_verify.py" "$HERE/" || { note "verify overlay FAILED"; exit 2; }
[[ -f "$PK/ring2/Dockerfile" && -f "$PK/ring2/tests/serve_workload.py" && -f "$PK/ring2/tests/check_shadow.py" ]] \
  || { note "packet missing in $PK/ring2"; exit 2; }
note "disk before build: $(df -h / | tail -1)"
gpu_lock
note "GPU lock held; results $R; daily $DENV"

if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
  note "building $IMG"
  timeout -k 30 5400 sudo -n docker build -f "$PK/ring2/Dockerfile" -t "$IMG" "$PK/ring2" > "$R/build.log" 2>&1 \
    || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 2; }
fi
note "image $IMG id $(sudo -n docker image inspect -f '{{.Id}}' "$IMG")"
DX=$(grep -oE 'EXL3_EXTRA=[^ ]+' <<<"$DENV" | head -1); DX=${DX#EXL3_EXTRA=}
# this image predates splitdev2 (uniform N): drop the daily's EXL3_MOE_CPU_SPLIT_BY_DEVICE so glm53_verify expects the arm's N
# (R919 run 2: SH-n108 booted at 108, verify read the daily's 100,104 -> NO BOOT)
DX=$(tr ';' '\n' <<< "$DX" | grep -v '^EXL3_MOE_CPU_SPLIT_BY_DEVICE=' | paste -sd';' -)
SLOT_START=$SECONDS

# --- slot helpers (WORK_END is set per phase) ---
cap(){ local l=$((WORK_END - SECONDS)); ((l < 10)) && l=10; ((l < $1)) && echo "$l" || echo "$1"; }
have(){ (( WORK_END - SECONDS >= $1 )); }
aboot(){  # dir env...: glm_arms boot (env -i, daily defaults) with the launcher's health wait capped by the phase
  local d=$1; shift; local t=$((WORK_END - SECONDS - 60))
  ((t >= 120)) || { note "no phase time left to boot $d"; return 1; }
  ((t > 900)) && t=900
  boot "$d" "$HERE" "$@" BOOT_TIMEOUT=$t
}
vram(){  # tag stage -> FREE0 FREE1
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$R/$1/vram-$2.csv"
  read -r FREE0 FREE1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
  note "$1 free MiB ($2): GPU0=$FREE0 GPU1=$FREE1"
}
STAGE=
armup(){  # tag env...: the daily env + overrides (later assignment wins); boot, warmup, VRAM after warmup
  local tag=$1; shift; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  STAGE=boot; note "$tag TRY: $*"
  # shellcheck disable=SC2086
  aboot "$D/boot" $DENV "$@" \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|OutOfMemory|ABORT[^;]{0,120}|NO BOOT[^;]{0,80}|Error[^;]{0,120}' "$D"/boot.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  # effective env = image Config.Env, then the launcher's `env -u K ... K=V ... python3 main.py` Args (later wins);
  # run 2 read Config.Env only (image defaults =0) and rejected a correct SH-n108 boot
  sudo -n docker inspect glm53 --format '{{range .Config.Env}}{{println .}}{{end}}{{range .Args}}{{println .}}{{end}}' > "$D/container-env.txt"
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  STAGE=warmup
  timeout -k 10 "$(cap 600)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1 \
    || { sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1; note "$tag WARMUP FAILED: $(tail -1 "$D/warmup.log" | cut -c1-160)"; return 1; }
  STAGE=up; vram "$tag" warmup
}
envcheck(){  # tag ring shadow: the container really got the flags
  local f=$R/$1/container-env.txt
  [[ "$(grep -E '^EXL3_DSA_INDEX_RING=' "$f" | tail -1)" == "EXL3_DSA_INDEX_RING=$2" && \
     "$(grep -E '^EXL3_DSA_INDEX_RING_SHADOW=' "$f" | tail -1)" == "EXL3_DSA_INDEX_RING_SHADOW=$3" ]] \
    || { note "$1 REJECT: container env lacks EXL3_DSA_INDEX_RING=$2 / _SHADOW=$3"; return 1; }
}
decsum(){ python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$1" 2>/dev/null | tail -1; }
measure(){  # tag: c1 five kinds x2 + server step, distinct c1/c2/c4 x2, vision, VRAM after the matrix
  local tag=$1 D=$R/$1 l0 rc=0
  l0=$(sudo -n docker logs glm53 2>&1 | wc -l)
  timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/c1.jsonl" \
    --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/c1.log" 2>&1 || note "$tag c1 rc=$?"
  sudo -n docker logs glm53 2>&1 | tail -n +"$((l0 + 1))" > "$D/engine-c1.log"
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee -a "$R/summary.txt"
  note "$tag server step: $(python3 "$HERE/mtp_steps.py" "$D/engine-c1.log" 10 2>&1 | tail -1 | cut -c1-200)"
  timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/dec.jsonl" \
    --phase decode --concurrency 1,2,4 --runs 2 --distinct > "$D/dec.log" 2>&1 || note "$tag decode rc=$?"
  note "$tag decode distinct: $(decsum "$D/dec.jsonl")"
  timeout -k 15 "$(cap 300)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/vision.jsonl" \
    --phase vision > "$D/vision.log" 2>&1 || rc=$?
  note "$tag vision rc=$rc $(tail -1 "$D/vision.log" | cut -c1-160)"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  vram "$tag" final
  grep -aqE 'RING-SHADOW-MISMATCH|illegal memory access|device-side assert|CUDA error' "$D/engine.log" && note "$tag REJECT: engine error in log" || true
}

# --- K: kernel tests on both GPUs, no model ---
WORK_END=$((SECONDS + 1500))
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
for _ in $(seq 12); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{b=1} END{exit b}' && break; sleep 5; done   # VRAM released
mkdir -p "$R/K"
for g in 0 1; do
  for t in test_kernels test_graph_shadow test_dispatch_shadow; do
    timeout -k 15 "$(cap 1200)" sudo -n docker run --rm --gpus all --ipc=host --entrypoint python "$IMG" \
      "/opt/index-ring/tests/$t.py" --device "cuda:$g" > "$R/K/$t-cuda$g.log" 2>&1
    rc=$?; note "K $t cuda:$g rc=$rc $(tail -1 "$R/K/$t-cuda$g.log" | cut -c1-120)"
    ((rc == 0)) || { note "VERDICT R919 kernel tests FAIL ($t cuda:$g): no serving arms"; exit 3; }
  done
done
note "VERDICT R919 kernel tests PASS on both GPUs"

# --- SH: shadow validation on served traffic, <= 60 min ---
WORK_END=$((SECONDS + 3600))
SH=
for n in 104 108; do
  tag=SH-n$n
  [[ $n == 104 ]] || note "$tag: labelled fallback for shadow COVERAGE only (more CPU experts frees VRAM; pool unchanged)"
  if armup "$tag" GLM_IMG="$IMG" INDEX_RING=1 OFFLOAD_N=$n "EXL3_EXTRA=$DX;EXL3_DSA_INDEX_RING_SHADOW=1" && envcheck "$tag" 1 1; then
    SH=$tag; break
  fi
done
if [[ -z "$SH" ]]; then
  note "VERDICT R919 shadow NOT RUN: no shadow boot (N=104, 108); memory in $R/SH-n*/boot/; no promotion arms"; exit 3
fi
MIN=$(( (WORK_END - SECONDS - 240) / 60 )); ((MIN > 50)) && MIN=50
note "$SH workload: serve_workload --mode shadow, $MIN min"
timeout -k 30 "$((MIN * 60 + 120))" python3 -u "$PK/ring2/tests/serve_workload.py" --mode shadow --minutes "$MIN" \
  --cache-tokens 262144 --model "$MODEL" --out "$R/$SH/traffic.jsonl" > "$R/$SH/workload.log" 2>&1
note "$SH workload rc=$? requests=$(grep -c . "$R/$SH/traffic.jsonl" 2>/dev/null) errors=$(grep -c '"error"' "$R/$SH/traffic.jsonl" 2>/dev/null)"
sleep 30   # idle summary
sudo -n docker logs glm53 > "$R/$SH/engine-live.log" 2>&1
sudo -n docker stop -t 90 glm53 >/dev/null 2>&1   # graceful: shutdown summary (docker rm -f would SIGKILL)
sudo -n docker logs glm53 > "$R/$SH/engine.log" 2>&1
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
note "$SH last summary: $(grep -a '\[RING-SHADOW\] calls=' "$R/$SH/engine.log" | tail -1 | cut -c1-220)"
python3 "$PK/ring2/tests/check_shadow.py" "$R/$SH/engine.log" --traffic "$R/$SH/traffic.jsonl" --out "$R/$SH/verdict.json" > "$R/$SH/check.log" 2>&1
SHRC=$?
note "$SH check_shadow rc=$SHRC: $(python3 -c 'import json,sys;v=json.load(open(sys.argv[1]));print("passed" if v["passed"] else "FAILED", "calls", v["calls"], "rows", v["compared_rows"], "errors", v["errors"][:4])' "$R/$SH/verdict.json" 2>&1 | cut -c1-300)"
((SHRC == 0)) || { note "VERDICT R919 shadow FAIL or incomplete coverage: no promotion arms (first mismatch context in $R/$SH/engine.log)"; exit 3; }
note "VERDICT R919 shadow PASS ($SH): zero mismatches, full served coverage"

# --- P: promotion, ring ON, shadow OFF ---
WORK_END=$((SECONDS + 4800))
LOWBOOT=; LOWPASS=
for n in 96 97 98 99 100 101 102 103 104; do
  tag=P-n$n
  if armup "$tag" GLM_IMG="$IMG" INDEX_RING=1 OFFLOAD_N=$n "EXL3_EXTRA=$DX;EXL3_DSA_INDEX_RING_SHADOW=0"; then
    LOWBOOT=${LOWBOOT:-$n}
    envcheck "$tag" 1 0 || continue
    if ((FREE0 >= 450 && FREE1 >= 300)); then LOWPASS=$n; measure "$tag"; break; fi
    note "$tag REJECT headroom GPU0 $FREE0 / GPU1 $FREE1 MiB (need 450 / 300); trying N+1"
  else
    [[ $STAGE == boot ]] || LOWBOOT=${LOWBOOT:-$n}
    have 300 || { note "P search stopped at N=$n: phase deadline"; break; }
  fi
done
note "VERDICT R919 promotion: lowest booting N=${LOWBOOT:-none} (262k, ring on); lowest N with headroom=${LOWPASS:-none}"
if have 900; then
  tag=P-n104-524k
  if armup "$tag" GLM_IMG="$IMG" INDEX_RING=1 OFFLOAD_N=104 CACHE_TOKENS=524288 "EXL3_EXTRA=$DX;EXL3_DSA_INDEX_RING_SHADOW=0" \
     && envcheck "$tag" 1 0; then
    ((FREE0 >= 450 && FREE1 >= 300)) || note "$tag headroom FAIL GPU0 $FREE0 / GPU1 $FREE1 MiB (need 450 / 300): measured, not promotable"
    measure "$tag"
    note "VERDICT R919 524k pool at N=104: boots; see $tag lines"
  else
    note "VERDICT R919 524k pool at N=104: NO BOOT/WARMUP (logs $R/$tag/boot)"
  fi
else
  note "VERDICT R919 524k arm UNTESTED: $(( (WORK_END - SECONDS) / 60 )) min left"
fi
note "unit used $(( (SECONDS - SLOT_START) / 60 )) min after the build; promotion into glm-daily.env is the operator's"
exit 0

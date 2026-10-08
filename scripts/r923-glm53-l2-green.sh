#!/usr/bin/env bash
# R923 (2026-10-08): codex-glm-sm120-l2-green-r1, L2 HALF ONLY. EXL3_SM120_L2=1 (default off) sets a CUDA L2
# access-policy window (persisting set-aside, default 8 MiB, hitRatio 1) per decode layer on the decode stream and on
# captured graph kernel nodes, pinning the layer's per-step re-read state (KDA slot state, DSA pooled planes, HC/router
# weights). Cache policy only: bitwise identical by construction. Init probes support per device and falls back if the
# GeForce driver gives a zero persisting limit or rejects stream/graph windows.
# The green-context half is OUT: it runs only inside the micro-batch pipeline, which failed on GPU today (R920: c4 -10 %).
# Packet: patches/exllamav3/glm-sm120-l2-green-r1/out (OPERATOR.md; BRIEF.md one level up).
#
# Image tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_l2, built FROM the CURRENT daily image (splitdev2)
# with patches/exllamav3/glm-sm120-l2-green-r1/Dockerfile.l2-splitdev2 (the packet's l2 stage, FROM line changed): the 7
# files l2.patch modifies are byte-identical in the overhead-r2 base and the splitdev2 source (checked 2026-10-08), so
# the arms boot the daily's per-device split 100/104, not a uniform N.
#
# Deviations from OPERATOR.md, on purpose:
#   - No site adapters (R923_STOP/RESTART/GPU/STEP/RESTORE/URL/MODEL env) and no run_operator.py: glm_arms.sh does lock,
#     boot (launcher, daily env) and queue-aware restore; glm53_probe.py measures; mtp_steps.py gives the server step from
#     the R828 counters over the c1 window. The receipt JSON is replaced by the launcher's resolved.json, container-env and
#     served-inspect.json in each arm's boot dir.
#   - serving_probe.py/compare_serving.py unused (192-token probes); the house matrix is c1 five kinds x2 + distinct.
#   - Cross-boot text comparison is skipped: greedy output is not reproducible across boots (R894), so it cannot judge a
#     cache-policy change; the bitwise gate is the in-image native chain test (G).
#   - ABA (OFF, ON, OFF) instead of one OFF/ON pair: the OPERATOR itself says boot noise exceeds the expected gain.
#
# Arms, one slot <= 60 min after the build (last 5 min reserved for the cleanup restore):
#   build  after gpu_lock if the image is missing (apply_image.py: base SHA + zero-fuzz + native rebuild, torch-first
#          /app landing assert; ~6 min, R920's native rebuild took 5.4 min). Not counted in the 60 min.
#   P      glm53 stopped; l2_green_probe on GPU0 and GPU1: persisting-L2 limit set/readback, stream window, captured
#          graph kernel-node window, re-read microbench (byte-equal). If either GPU lacks L2 support (l2_supported &&
#          stream_window && graph_window), VERDICT "unsupported" and STOP: an inactive flag is not a speed result.
#   G      gpu_tests.py --feature l2: synthetic HC->norm->KDA->router chain, off/on byte equality, eager vs captured,
#          per-device capture streams, c1 and c4 rows, both GPUs. Any failure stops the unit (no serving arms).
#   H      profile_l2.sh: ncu L2 sector hit rate off/on per GPU on the probe's re-read kernel (records UNAVAILABLE if ncu
#          or the counters are blocked; timings excluded from speed).
#   OFF1 / ON / OFF2   daily config (glm-daily.env) on the l2 image, EXL3_SM120_L2=0/1/0 via EXL3_EXTRA; each: boot,
#          warmup, VRAM after warmup, c1 five kinds x2 + server step, distinct c1/c2/c4 x2, vision, VRAM after the matrix.
#          ON also records SM120_L2_INIT (per-device support) / SM120_L2_SELECT / "SM120_L2 fallback" lines; a fallback
#          marks ON inactive. OFF2 runs only if >= 14 min remain.
# Verdict line: ON vs mean(OFF1, OFF2) for c1 score, distinct c1/c2/c4, server ms/token, with the OFF1/OFF2 spread
#   beside it. VRAM gate as the daily: GPU0 >= 450 MiB and GPU1 >= 300 MiB free after warmup. No promotion by this unit:
#   a winner goes into flan/r858/glm-daily.env (GLM_IMG=<this image>, EXL3_EXTRA += EXL3_SM120_L2=1) by the operator.
# PK=/srv/qwen5090/r923 must hold: glm_arms.sh launch-glm53.sh glm53_plan.py glm53_probe.py glm53_verify.py mtp_steps.py
#   (copied from flan/r858/) and l2/ = the whole patches/exllamav3/glm-sm120-l2-green-r1/out/ directory plus
#   ../Dockerfile.l2-splitdev2 (Docker context).
# Expected wall: queue + build ~6-8 min + P/G/H ~5-10 min + 3 arms ~13-15 min each (~50-55 min slot) + restore ~3 min;
#   L2 unsupported: build + ~5 min, then restore.
#   sudo systemd-run --unit=r923-glm53-l2-green --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r923-glm53-l2-green.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r923
PK=/srv/qwen5090/r923
HERE=/srv/qwen5090/r923-tools   # the unit's OWN dir: arms_init does rm -rf "$HERE"
BASE_IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_l2
MODEL=glm53-flash-exl3-2.05bpw-turboderp
GPU_CTR=r923-gpu   # name of every one-shot GPU test container (removed on exit)
R=/srv/qwen5090/results/$(date +%F)-r923-glm53-l2-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r923-glm53-l2
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
# a test container killed by timeout must not survive into the restore; then glm_arms' cleanup with the original rc
trap 'rc=$?; sudo -n docker rm -f "$GPU_CTR" >/dev/null 2>&1; (exit $rc); arms_cleanup' EXIT
cp "$PK/glm53_verify.py" "$HERE/" || { note "verify overlay FAILED"; exit 2; }
[[ -f "$PK/l2/Dockerfile.l2-splitdev2" && -f "$PK/l2/l2.patch" && -f "$PK/l2/apply_image.py" ]] || { note "packet missing in $PK/l2"; exit 2; }
(cd "$PK/l2" && sha256sum --quiet -c SHA256SUMS) > "$R/packet-sha256.log" 2>&1 || { note "packet checksum FAILED ($(head -2 "$R/packet-sha256.log" | tr '\n' ' '))"; exit 2; }
sudo -n docker image inspect "$BASE_IMG" >/dev/null 2>&1 || { note "base image $BASE_IMG missing"; exit 2; }
note "disk before build: $(df -h / | tail -1)"
gpu_lock
note "GPU lock held; results $R; daily $DENV"

# Build after the lock (a native rebuild must not overlap another unit's measurement); plain docker build, legacy builder.
if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
  note "building $IMG FROM $BASE_IMG"
  timeout -k 30 3600 sudo -n docker build -f "$PK/l2/Dockerfile.l2-splitdev2" -t "$IMG" "$PK/l2" > "$R/build.log" 2>&1 \
    || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 2; }
fi
note "image $IMG id $(sudo -n docker image inspect -f '{{.Id}}' "$IMG") native $(grep -aoE '"sha256": "[0-9a-f]{64}"' "$R/build.log" 2>/dev/null | tail -1)"

SLOT_START=$SECONDS
WORK_END=$((SECONDS + 3600 - 300))   # 60 min slot after the build, last 5 min reserved for the cleanup restore
DX=$(grep -oE 'EXL3_EXTRA=[^ ]+' <<<"$DENV" | head -1); DX=${DX#EXL3_EXTRA=}

# --- slot helpers (R920 conventions) ---
cap(){ local l=$((WORK_END - SECONDS)); ((l < 10)) && l=10; ((l < $1)) && echo "$l" || echo "$1"; }
have(){ (( WORK_END - SECONDS >= $1 )); }
aboot(){  # dir env...: glm_arms boot (env -i, daily defaults) with the launcher's health wait capped by the slot
  local d=$1; shift; local t=$((WORK_END - SECONDS - 60))
  ((t >= 120)) || { note "no slot time left to boot $d"; return 1; }
  ((t > 900)) && t=900
  boot "$d" "$HERE" "$@" BOOT_TIMEOUT=$t
}
vram(){  # tag stage -> FREE0 FREE1
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$R/$1/vram-$2.csv"
  read -r FREE0 FREE1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
  note "$1 free MiB ($2): GPU0=$FREE0 GPU1=$FREE1 (gate 450/300: $([[ "$FREE0" =~ ^[0-9]+$ && "$FREE1" =~ ^[0-9]+$ ]] && ((FREE0 >= 450 && FREE1 >= 300)) && echo pass || echo FAIL))"
}
armup(){  # tag env...: the daily env + overrides (later assignment wins); boot, warmup, VRAM after warmup
  local tag=$1; shift; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  # shellcheck disable=SC2086
  aboot "$D/boot" $DENV "$@" \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|OutOfMemory|ABORT[^;]{0,120}|NO BOOT[^;]{0,80}|Error[^;]{0,120}' "$D"/boot.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  sudo -n docker inspect glm53 --format '{{range .Config.Env}}{{println .}}{{end}}' > "$D/container-env.txt"
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  timeout -k 10 "$(cap 600)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1 \
    || { sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1; note "$tag WARMUP FAILED: $(tail -1 "$D/warmup.log" | cut -c1-160)"; return 1; }
  vram "$tag" warmup
}
decsum(){ python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$1" 2>/dev/null | tail -1; }
measure(){  # tag: c1 five kinds x2 + server step (c1 window of the engine log), distinct c1/c2/c4 x2, vision, VRAM
  local tag=$1 D=$R/$1 l0 rc=0
  l0=$(sudo -n docker logs glm53 2>&1 | wc -l)
  timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/c1.jsonl" \
    --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/c1.log" 2>&1 || note "$tag c1 rc=$?"
  sudo -n docker logs glm53 2>&1 | tail -n +"$((l0 + 1))" > "$D/engine-c1.log"
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee -a "$R/summary.txt"
  python3 "$HERE/mtp_steps.py" "$D/engine-c1.log" 10 > "$D/server-step.txt" 2>&1
  note "$tag server step: $(tail -1 "$D/server-step.txt" | cut -c1-200)"
  timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/dec.jsonl" \
    --phase decode --concurrency 1,2,4 --runs 2 --distinct > "$D/dec.log" 2>&1 || note "$tag decode rc=$?"
  note "$tag decode distinct: $(decsum "$D/dec.jsonl")"
  timeout -k 15 "$(cap 300)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/vision.jsonl" \
    --phase vision > "$D/vision.log" 2>&1 || rc=$?
  note "$tag vision rc=$rc $(tail -1 "$D/vision.log" | cut -c1-160)"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  vram "$tag" final
  grep -aqE 'Traceback|illegal memory access|device-side assert|CUDA error|worker abort|Active L2 policy requires' "$D/engine.log" \
    && note "$tag REJECT: engine error in log ($(grep -aoE 'illegal memory access|device-side assert|CUDA error[^;]{0,80}|Active L2 policy requires[^;]{0,60}|Traceback' "$D/engine.log" | head -1))" || true
}
l2evidence(){  # tag: per-device init support, selections, fallbacks, as logged by the engine
  local D=$R/$1
  grep -a 'SM120_L2_INIT' "$D/engine.log" > "$D/l2-init.log" 2>/dev/null
  grep -a 'SM120_L2_SELECT' "$D/engine.log" > "$D/l2-select.log" 2>/dev/null
  note "$1 L2 evidence: INIT $(cut -c1-200 "$D/l2-init.log" | tr '\n' ' ') | SELECT lines $(wc -l < "$D/l2-select.log") applied=false $(grep -c '"applied": false' "$D/l2-select.log") | fallback lines $(grep -ac 'SM120_L2 fallback' "$D/engine.log")"
}
idle_gpus(){  # also removes a test container that outlived its timeout (it would hold VRAM: phantom NO BOOT)
  sudo -n docker rm -f glm53 "$GPU_CTR" >/dev/null 2>&1 || true
  for _ in $(seq 12); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{b=1} END{exit b}' && break; sleep 5; done
}
gpurun(){  # cap log docker-run-args...: one-shot GPU container, bounded by the slot
  local c=$1 log=$2; shift 2
  sudo -n docker rm -f "$GPU_CTR" >/dev/null 2>&1 || true
  timeout -k 15 "$(cap "$c")" sudo -n docker run --rm --name "$GPU_CTR" "$@" > "$log" 2>&1
}

# --- P: capability probe on both GPUs, no model loaded ---
idle_gpus
mkdir -p "$R/P" "$R/G" "$R/H"; chmod 0777 "$R/G" "$R/H"
for dev in 0 1; do
  gpurun 120 "$R/P/probe-d$dev.log" --gpus all --entrypoint /opt/r923/l2_green_probe "$IMG" "$dev"
  note "P probe GPU$dev rc=$?"
done
PSUM=$(python3 - "$R/P" <<'PY'
import json, sys, pathlib
P = pathlib.Path(sys.argv[1]); ok = True; parts = []
for dev in (0, 1):
    rows = []
    for line in (P / f'probe-d{dev}.log').read_text(errors='replace').splitlines():
        if line.startswith('{'):
            try: rows.append(json.loads(line))
            except ValueError: pass
    lim = next((r for r in rows if 'max_persisting' in r), {})
    sup = next((r for r in rows if 'l2_supported' in r), {})
    mb = [r for r in rows if r.get('microbench')]
    s = bool(sup.get('l2_supported') and sup.get('stream_window') and sup.get('graph_window'))
    ok &= s
    on = [r for r in mb if r.get('mode') == 1]
    best = min(on, key=lambda r: r['graph_ms']) if on else None
    off = [r['graph_ms'] for r in mb if r.get('mode') == 0]
    parts.append(f"GPU{dev} supported={s} (l2 {sup.get('l2_supported')} stream {sup.get('stream_window')} graph {sup.get('graph_window')}) "
                 f"l2_bytes={lim.get('l2_bytes')} max_persisting={lim.get('max_persisting')} max_window={lim.get('max_window')} "
                 f"microbench off {min(off) if off else 'NA'} ms, best on {best['graph_ms'] if best else 'NA'} ms "
                 f"(budget {best['budget'] if best else 'NA'} hit {best['hit_ratio'] if best else 'NA'}) byte_equal all="
                 f"{all(r.get('byte_equal') for r in mb) if mb else 'NA'}")
print(('SUPPORTED ' if ok else 'UNSUPPORTED ') + ' | '.join(parts))
PY
)
note "P ${PSUM:0:700}"
if [[ "$PSUM" != SUPPORTED* ]]; then
  note "VERDICT R923 L2 UNSUPPORTED on this GeForce driver (see P): EXL3_SM120_L2=1 would fall back, no serving arms"
  exit 0
fi

# --- G: native chain byte-equality gate, both GPUs ---
gpurun 900 "$R/G/l2-native.log" --gpus all --ipc=host -v "$R/G":/results -w /app --entrypoint python "$IMG" \
  /opt/r923/gpu_tests.py --feature l2 --out /results/l2-native
GRC=$?
note "G native chain gate rc=$GRC: $(tail -2 "$R/G/l2-native.log" | tr '\n' ' ' | cut -c1-240)"
((GRC == 0)) || { note "VERDICT R923 native L2 gate FAIL (rc=$GRC): no serving arms"; exit 3; }

# --- H: hardware L2 hit rate (ncu), informative only ---
gpurun 240 "$R/H/profile.log" --gpus all -v "$R/H":/results --entrypoint bash "$IMG" /opt/r923/profile_l2.sh /results/l2-hits
note "H hit-rate rc=$? $(tr -s ' \n' ' ' 2>/dev/null < "$R/H/l2-hits/hit-rate-status.json" | cut -c1-300)"

# --- served ABA on the daily config ---
idle_gpus
armup OFF1 GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_SM120_L2=0" && measure OFF1
if armup ON GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_SM120_L2=1"; then
  measure ON
  l2evidence ON
fi
if have 840; then armup OFF2 GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_SM120_L2=0" && measure OFF2
else note "OFF2 UNTESTED: slot deadline (OFF drift unbounded)"; fi

python3 - "$R" <<'PY' | while read -r l; do note "$l"; done
import json, re, statistics as st, sys, pathlib
R = pathlib.Path(sys.argv[1])
def arm(tag):
    d, out = R / tag, {}
    try:
        by = {}
        for l in open(d / 'c1.jsonl'):
            r = json.loads(l)
            if r.get('phase') == 'c1-decode': by.setdefault(r['kind'], []).append(r['tps'])
        if len(by) == 5: out['c1 score'] = st.mean(st.median(v) for v in by.values())
    except Exception: pass
    try:
        for l in open(d / 'dec.jsonl'):
            r = json.loads(l)
            if r.get('phase') == 'decode-summary':
                out.update({f"distinct c{s['c']}": s['ss_agg_tps_median'] for s in r['summaries']})
    except Exception: pass
    try:
        m = re.search(r'ms/tok=([0-9.]+)', (d / 'server-step.txt').read_text())
        if m: out['server ms/tok'] = float(m.group(1))
    except Exception: pass
    return out
off = [a for a in (arm('OFF1'), arm('OFF2')) if a]; on = arm('ON')
inactive = False
try:
    init = [json.loads(l.split('SM120_L2_INIT', 1)[1]) for l in open(R / 'ON' / 'l2-init.log')]
    sup = {r.get('device'): r.get('supported') for r in init}
    inactive = not init or not all(sup.values())
    fb = sum('SM120_L2 fallback' in l for l in open(R / 'ON' / 'engine.log', errors='replace'))
    inactive |= fb > 0
    ev = f'ON init support {sup}, fallback lines {fb}'
except Exception as e:
    ev, inactive = f'ON support evidence missing ({e})', True
if not off or not on:
    print(f'VERDICT R923 L2 speed: incomplete arms (OFF n={len(off)}, ON {"ok" if on else "missing"}); {ev}')
    sys.exit()
parts = []
for k in ('c1 score', 'distinct c1', 'distinct c2', 'distinct c4', 'server ms/tok'):
    xs = [a[k] for a in off if k in a]
    if k not in on or not xs: parts.append(f'{k} NA'); continue
    m = st.mean(xs); spread = (max(xs) - min(xs)) / m if len(xs) > 1 else float('nan')
    parts.append(f'{k} {on[k]:.2f} vs {m:.2f} ({on[k] / m - 1:+.1%}, OFF spread {spread:.1%})')
print(f'VERDICT R923 L2 ON vs mean(OFF n={len(off)}): ' + '; '.join(parts) + f'; {ev}'
      + (' -> ON INACTIVE (fallback): not a feature result' if inactive else ''))
PY
note "slot used $(( (SECONDS - SLOT_START) / 60 )) min; no promotion requested by this unit (operator decides; sub-percent needs a repeat slot)"
exit 0

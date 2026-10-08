#!/usr/bin/env bash
# R920 (2026-10-08): codex-glm-mbpipe-r1, first GPU gate. Asynchronous two-stage micro-batch pipeline decode at c2-c4
# (EXL3_MB_PIPELINE=1): lanes A/B, GPU0 runs one lane's stage 1 while GPU1 runs the other's stage 2, each lane samples and
# re-enters on its own (no global round join), one 8-thread CPU worker serves both lanes in readiness order. c1, prefill,
# admission/finish/cancel and exchange sweeps drain to the normal path. Modelled c4 129 (110 if CPU service is 20 %
# heavier) vs ~96 distinct today. Image tabbyapi:cheapswap-r3-agent-r2_mtpfast1_mbpipe1 (FROM the daily image
# cheapswap-r3-agent-r2_mtpfast1_overhead-r2 despite its name; native worker change -> rebuild).
# Packet: patches/exllamav3/glm-mbpipe-r1/out (OPERATOR.md; BRIEF.md one level up).
#
# OPERATOR.md predates the daily promotion (it says N=96, no MTP). This unit uses the CURRENT daily (glm-daily.env: N=104,
# MTP depth 1 + MTP_FAST + EXL3_MTP_MAX_BATCH=1, vision offload, exchange). The served arms are therefore also the
# OPERATOR's "MTP-cap smoke" (MTP at c1, pipeline at c2-c4, drains on membership changes, one 43-layer worker).
# Arms, one slot <= 90 min after the build (5 min of it reserved for the restore); stop at the first failing gate:
#   build  after gpu_lock if missing (apply_image.py: sha256 + zero-fuzz + native rebuild, torch-first landing).
#   Q      one-process quality, glm53 stopped: gpu_gate.py --n 104 --placement frozen-exchange (daily map/arena, exchange
#          interval frozen so tier numerics hold still), --cache 262144 --split 31 31 --vision-offload, c4 first, 4k ctx,
#          128 tokens: greedy normal / normal-repeat / pipeline, then teacher-forced normal / repeat / pipeline / bracket
#          on the same reference IDs; ragged finish, cancel, long-prefill admission, prefix hit. Plain decode (no MTP).
#          Pass: mean KL <= max(1e-6, 3x floor), p99 <= max(1e-5, 3x floor), top-1 >= floor - 0.002, finite masks,
#          stage0_calls > 0, report status PASS. FAIL -> no serving arms (a quality failure forbids speed promotion).
#   F0     daily config on the mbpipe image, EXL3_MB_PIPELINE=0 (default path, control).
#   P1     same, EXL3_MB_PIPELINE=1 (flag set before worker spawn: switching a flag-0 worker is rejected by design);
#          + functional_http.py (ragged c2/c3/c4, disconnect, long prefill during decode, prefix repeat, vision "blue",
#          forced tool call). c1 lane entries (MBPIPE_* lines during the c1 phase) are a bug and are counted.
#   F0b    flag 0 again (drift bound).
#   Each served arm: c1 five kinds x2 + server step, distinct c1/c2/c4 x2, vision, free VRAM per GPU after warmup/matrix.
#   In lane mode R828 step counters can over-count rows (OPERATOR); the c2/c4 decision uses the probe's client-side
#   distinct stream rates (sum of per-stream decode rates), not mtp_steps' c4 rounds.
# Pass/fail (OPERATOR): Q PASS; P1 distinct c4 >= +10 % over mean(F0, F0b) with c1 score within -3 %; no c1 lane entry;
#   functional PASS; VRAM GPU0 >= 450 / GPU1 >= 300 MiB after the matrix. No promotion is requested by this packet; a
#   winner would go into glm-daily.env (GLM_IMG=..._mbpipe1, EXL3_EXTRA += EXL3_MB_PIPELINE=1) by the operator only.
#   c3, the exchange-boundary trace (--placement exchange --functional-only --trace) and nsys overlap come in slot 2.
# PK=/srv/qwen5090/r920 must hold: glm_arms.sh launch-glm53.sh glm53_plan.py glm53_probe.py glm53_verify.py mtp_steps.py
#   (from flan/r858/) and mbpipe/ = the whole patches/exllamav3/glm-mbpipe-r1/out/ directory (Docker context).
# Expected wall: queue + build (~30-60 min) + Q ~20-25 + 3 served arms ~12 each + functional ~5 = ~70-80 min slot + restore.
#   sudo systemd-run --unit=r920-glm53-mbpipe --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r920-glm53-mbpipe.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r920
PK=/srv/qwen5090/r920
HERE=/srv/qwen5090/r920-tools   # the unit's OWN dir: arms_init does rm -rf "$HERE"
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_mbpipe1
MODEL=glm53-flash-exl3-2.05bpw-turboderp
R=/srv/qwen5090/results/$(date +%F)-r920-glm53-mbpipe-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r920-glm53-mbpipe
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
cp "$PK/glm53_verify.py" "$HERE/" || { note "verify overlay FAILED"; exit 2; }
[[ -f "$PK/mbpipe/Dockerfile" && -f "$PK/mbpipe/functional_http.py" ]] || { note "packet missing in $PK/mbpipe"; exit 2; }
note "disk before build: $(df -h / | tail -1)"
gpu_lock
note "GPU lock held; results $R; daily $DENV"

if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
  note "building $IMG"
  timeout -k 30 5400 sudo -n docker build -f "$PK/mbpipe/Dockerfile" -t "$IMG" "$PK/mbpipe" > "$R/build.log" 2>&1 \
    || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 2; }
fi
note "image $IMG id $(sudo -n docker image inspect -f '{{.Id}}' "$IMG")"
SLOT_START=$SECONDS
WORK_END=$((SECONDS + 5400 - 300))   # 90 min slot, last 5 min reserved for the cleanup restore
DX=$(grep -oE 'EXL3_EXTRA=[^ ]+' <<<"$DENV" | head -1); DX=${DX#EXL3_EXTRA=}

# --- slot helpers ---
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
  note "$1 free MiB ($2): GPU0=$FREE0 GPU1=$FREE1"
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
measure(){  # tag: c1 five kinds x2 + server step, distinct c1/c2/c4 x2, vision, VRAM; counts MBPIPE lines per phase
  local tag=$1 D=$R/$1 l0 l1 rc=0
  l0=$(sudo -n docker logs glm53 2>&1 | wc -l)
  timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/c1.jsonl" \
    --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/c1.log" 2>&1 || note "$tag c1 rc=$?"
  sudo -n docker logs glm53 2>&1 | tail -n +"$((l0 + 1))" > "$D/engine-c1.log"
  l1=$((l0 + $(wc -l < "$D/engine-c1.log")))
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee -a "$R/summary.txt"
  note "$tag server step: $(python3 "$HERE/mtp_steps.py" "$D/engine-c1.log" 10 2>&1 | tail -1 | cut -c1-200)"
  note "$tag MBPIPE lines during c1: $(grep -ac 'MBPIPE_' "$D/engine-c1.log") (must be 0: c1 lane entry is a bug)"
  timeout -k 15 "$(cap 900)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/dec.jsonl" \
    --phase decode --concurrency 1,2,4 --runs 2 --distinct > "$D/dec.log" 2>&1 || note "$tag decode rc=$?"
  sudo -n docker logs glm53 2>&1 | tail -n +"$((l1 + 1))" > "$D/engine-dec.log"
  note "$tag decode distinct: $(decsum "$D/dec.jsonl") | MBPIPE in decode: LAYOUT $(grep -ac 'MBPIPE_LAYOUT' "$D/engine-dec.log") DRAIN $(grep -ac 'MBPIPE_DRAIN' "$D/engine-dec.log") MEMORY $(grep -ac 'MBPIPE_MEMORY' "$D/engine-dec.log")"
  timeout -k 15 "$(cap 300)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/vision.jsonl" \
    --phase vision > "$D/vision.log" 2>&1 || rc=$?
  note "$tag vision rc=$rc $(tail -1 "$D/vision.log" | cut -c1-160)"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  vram "$tag" final
  grep -aqE 'Traceback|illegal memory access|device-side assert|CUDA error|worker abort' "$D/engine.log" && note "$tag REJECT: engine error in log" || true
}

# --- Q: one-process quality gate (no serving container) ---
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
for _ in $(seq 12); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{b=1} END{exit b}' && break; sleep 5; done   # VRAM released
CKPT=/srv/qwen5090/models/$MODEL
[[ -f "$CKPT/.nvme-verified" ]] || CKPT=/storage/data/models/$MODEL
STATS=$HERE/split-stats-broad-r869.json   # the daily's SWAP_INIT_STATS file (copied from glm-daily-tools by arms_init)
[[ -f "$STATS" ]] || { note "stats file $STATS missing"; exit 2; }
mkdir -p "$R/Q"; chmod 0777 "$R/Q"
note "Q quality gate: N=104 frozen-exchange, c4, ctx 4096, 128 tokens, checkpoint $CKPT"
timeout -k 30 "$(cap 1800)" sudo -n docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 \
  -v "$CKPT":/model:ro -v "$STATS":/stats.json:ro -v "$R/Q":/results --entrypoint python "$IMG" \
  /opt/mbpipe/gpu_gate.py --model /model --stats /stats.json --n 104 --cache 262144 --split 31 31 \
  --placement frozen-exchange --vision-offload --concurrency 4 --ctx 4096 --tokens 128 --out /results/quality-c4 \
  > "$R/Q/quality-c4.log" 2>&1
QRC=$?
QSUM=$(python3 - "$R/Q/quality-c4/report.json" 2>&1 <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
q = r.get('quality', [{}])[-1]; t = r.get('tokens', [{}])[-1]
print(f"status={r.get('status')} passed={q.get('passed')} token_gate={q.get('token_gate')} limits={q.get('limits')} "
      f"normal_repeat_equal={t.get('normal_repeat_equal')} pipeline_equal={t.get('pipeline_equal')} "
      f"stage0_calls={(r.get('functional') or {}).get('stage0_calls')} prefix_pages={r.get('prefix_cached_pages')}")
PY
)
note "Q rc=$QRC ${QSUM:0:400}"
note "Q last arms: $(grep -a 'MBPIPE_GATE' "$R/Q/quality-c4.log" | tail -2 | cut -c1-300 | tr '\n' ' ')"
((QRC == 0)) || { note "VERDICT R920 quality gate FAIL: no serving arms ($(tail -2 "$R/Q/quality-c4.log" | tr '\n' ' ' | cut -c1-200))"; exit 3; }
note "VERDICT R920 quality gate PASS"

# --- served arms on the daily config ---
armup F0 GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_MB_PIPELINE=0" && measure F0
if armup P1 GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_MB_PIPELINE=1"; then
  note "P1 layout at boot: $(grep -a 'MBPIPE_LAYOUT' "$R/P1/engine-boot.log" | head -1 | cut -c1-200)"
  measure P1
  if have 300; then
    timeout -k 15 "$(cap 600)" python3 -u "$PK/mbpipe/functional_http.py" --url http://127.0.0.1:8029 --out "$R/P1/http-functional" \
      > "$R/P1/http-functional.log" 2>&1
    note "P1 functional rc=$? $(tail -1 "$R/P1/http-functional.log" | cut -c1-200)"
    sudo -n docker logs glm53 > "$R/P1/engine-functional.log" 2>&1
    note "P1 drains in log: $(grep -ac 'MBPIPE_DRAIN' "$R/P1/engine-functional.log")"
  else
    note "P1 functional UNTESTED: slot deadline"
  fi
fi
if have 780; then armup F0b GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_MB_PIPELINE=0" && measure F0b
else note "F0b UNTESTED: slot deadline (drift unbounded)"; fi
python3 - "$R" <<'PY' | while read -r l; do note "$l"; done
import json, statistics as st, sys, pathlib
R = pathlib.Path(sys.argv[1])
def arm(tag):
    out = {}
    try:
        by = {}
        for l in open(R / tag / 'c1.jsonl'):
            r = json.loads(l)
            if r.get('phase') == 'c1-decode': by.setdefault(r['kind'], []).append(r['tps'])
        out['score'] = st.mean(st.median(v) for v in by.values())
        for l in open(R / tag / 'dec.jsonl'):
            r = json.loads(l)
            if r.get('phase') == 'decode-summary':
                out.update({f"c{s['c']}": s['ss_agg_tps_median'] for s in r['summaries']})
    except Exception:
        pass
    return out
a = {t: arm(t) for t in ('F0', 'P1', 'F0b')}
ctl = [a[t] for t in ('F0', 'F0b') if 'c4' in a[t] and 'score' in a[t]]
p = a['P1']
if not ctl or 'c4' not in p or 'score' not in p:
    print(f'VERDICT R920 speed: incomplete arms {a}')
else:
    c4 = st.mean(x['c4'] for x in ctl); c2 = st.mean(x['c2'] for x in ctl); sc = st.mean(x['score'] for x in ctl)
    g4, g2, gs = p['c4'] / c4 - 1, p['c2'] / c2 - 1, p['score'] / sc - 1
    ok = g4 >= .10 and gs >= -.03
    print(f"VERDICT R920 speed: P1 distinct c4 {p['c4']:.1f} vs control {c4:.1f} ({g4:+.1%}), c2 {p['c2']:.1f} vs {c2:.1f} "
          f"({g2:+.1%}), c1 score {p['score']:.1f} vs {sc:.1f} ({gs:+.1%}); controls n={len(ctl)} -> "
          + ('meets the +10 % c4 / -3 % c1 gate' if ok else 'below the gate'))
PY
note "slot used $(( (SECONDS - SLOT_START) / 60 )) min; no promotion requested by this packet"
exit 0

#!/usr/bin/env bash
# R924 (2026-10-08): codex-glm-lowp-tc-r1. MXFP8 attention-PREFILL prototype, NUMERICS-CHANGING, default off.
# EXL3_LOWP_MXFP8=1 converts the selected K4 EXL3 linears (default regex: every layer's self_attn.o_proj) at load from the
# daily's rotated trellis reconstruction to E4M3 + UE8M0 scales per 32 K values (no BF16 source), and runs them through a
# separate sm_120a block-scaled MMA kernel (mma.sync kind::mxf8f6f4) for rows >= 64 (prefill only). Decode, routed/CPU
# experts, head, vision and MTP keep the daily arithmetic; the K4 trellis stays resident, so MXFP8 is EXTRA VRAM
# (capped at EXL3_LOWP_BUDGET_MIB=256 per GPU, "LOWP budget skip" lines beyond). Serving it is the user's call.
# Packet: patches/exllamav3/glm-lowp-tc-r1/out (OPERATOR.md, ANALYSIS.md; BRIEF.md one level up).
#
# Image tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_lowp1, built FROM the CURRENT daily image (splitdev2)
# with patches/exllamav3/glm-lowp-tc-r1/Dockerfile.lowp-splitdev2 (the packet's Dockerfile, FROM line changed): the 2 files
# lowp.patch modifies have the same SHA256 in the overhead-r2 base and the splitdev2 source (checked 2026-10-08), so the
# served arms keep the daily's per-device split 100/104.
#
# Deviations from OPERATOR.md / the packet's r924 unit and flan_driver.py, on purpose:
#   - flan_driver.py is not used: it asserts the live daily is ..._overhead-r2 (stale), needs glm53 RUNNING at lock time
#     (false when the previous unit of a chain skipped its restore), clones the env from docker inspect Config.Env (the
#     launcher's effective env is in Args: `/usr/bin/env -u ... KEY=VAL ... python3 main.py`; Config.Env is the raw image
#     env, so the clone would drop CPU threads/MTP/swap flags and keep flags the launcher unsets), restores with
#     `docker start glm53` (containers here are rm -f'd; glm_arms owns the queue-aware restore), and searches every N in
#     94..112 per arm (unbounded in a 60 min slot). No GLM_ARMS_SH / R924_QUEUE_BUSY_CMD / R924_TOKENS_JSON env: the
#     paths are hardcoded below.
#   - Quality = model_probe.py --phase quality (in-process off/on/off-repeat teacher forcing, static uniform N, chunks of
#     64 rows) in a one-shot container cloned from the OFF1 arm's served-inspect.json (Binds + Args), with the R893 fixed
#     token IDs (six kinds x 3,501 IDs: agentic chat code edit-diff html prose). N 104; only a capacity failure retries
#     106, then 108.
#   - Speed = the house served matrix through the launcher (real tabbyAPI, MTP, exchange), not model_probe's in-process
#     Generator speed phase; prefill 8k/32k cold = glm53_probe --depths (uuid-prefixed prompts, engine prompt_time,
#     cached_tokens == 0 enforced), two runs.
#   - Minimum N: ON tries the daily split 100,104, then 100,106, then 100,108 (OFFLOAD_N = the larger value), moving up only
#     on a capacity failure (NO BOOT with Insufficient VRAM / out of memory) or a failed VRAM gate. No downward bracket:
#     the daily N is the floor of interest. OFF is the daily split.
#
# Arms, one slot <= 60 min after the build (last 5 min reserved for the cleanup restore):
#   build  after gpu_lock if the image is missing (install_patch.py base/patched SHA + zero fuzz; rebuild_native.py:
#          daily extension sm_120 + exllamav3_lowp_ext sm_120a, MAX_JOBS=4; landing assert; CPU converter tests in the
#          build). Then a no-GPU import check of the serving import path (exllamav3.ext + exllamav3_lowp_ext). Not counted.
#   K      glm53 stopped; gpu_tests.py on BOTH GPUs (quantize/scales vs CPU reference, GEMM vs dequantized fp32, tails,
#          capture, wrong-device guard, EXL3 Hadamard/bias integration, off/decode equality). FAIL stops the unit.
#   OFF1   daily config on the lowp image, EXL3_LOWP_MXFP8=0: boot, warmup, VRAM, c1 five kinds x2 + server step,
#          distinct c1/c2/c4 x2, prefill 8k/32k cold x2, vision, VRAM. Its boot dir is the quality container's template.
#   Q      teacher-forced KL, lowp ON vs OFF in one process, 6 kinds x 3,500 positions, plus an OFF repeat (must be
#          byte-identical); thresholds mean <= 0.001, p99 <= 0.01, top-1 >= 0.995 (R893 swap floor 0.00048 / 0.015 /
#          0.9959 for scale). Lowp dispatch counters must advance in ON and stay fixed in OFF (asserted by the harness).
#   ON     EXL3_LOWP_MXFP8=1 at the lowest split of the ladder that boots and passes the VRAM gate; same matrix; LOWP
#          converted / budget-skip lines recorded.
#   OFF2   flag 0 again if >= 15 min remain (drift bound; usually cut by the slot).
# Verdict line: KL mean / p99 / top-1 (and control) NEXT TO ON vs OFF c1 score, distinct c1/c2/c4, server ms/tok, prefill
#   8k/32k, N per arm and free VRAM per GPU. Screening (packet): quality PASS, >= +2 % on one cold prefill length, no
#   distinct decode concurrency below -1 %. No promotion by this unit; numerics change, so serving is the USER's decision.
# PK=/srv/qwen5090/r924 must hold: glm_arms.sh launch-glm53.sh glm53_plan.py glm53_probe.py glm53_verify.py mtp_steps.py
#   (copied from flan/r858/) and lowp/ = the whole patches/exllamav3/glm-lowp-tc-r1/out/ directory plus
#   ../Dockerfile.lowp-splitdev2 (Docker context).
# Expected wall: queue + build ~8-12 min (two native builds) + K ~3-5 + OFF1 ~15 + Q ~10-15 + ON ~15 (+3 per extra rung)
#   = ~50-55 min slot (OFF2 normally UNTESTED) + restore ~3 min.
#   sudo systemd-run --unit=r924-glm53-lowp --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r924-glm53-lowp.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r924
PK=/srv/qwen5090/r924
HERE=/srv/qwen5090/r924-tools   # the unit's OWN dir: arms_init does rm -rf "$HERE"
BASE_IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_lowp1
MODEL=glm53-flash-exl3-2.05bpw-turboderp
IDS=/srv/qwen5090/results/2026-10-08-r893-glm53-cpu-skip-quality-050418/teacher-quality-r3/texts   # R893 fixed token IDs
GPU_CTR=r924-gpu   # name of every one-shot GPU container (kernel tests, quality); removed on exit
R=/srv/qwen5090/results/$(date +%F)-r924-glm53-lowp-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r924-glm53-lowp
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
# a test container killed by timeout must not survive into the restore; then glm_arms' cleanup with the original rc
trap 'rc=$?; sudo -n docker rm -f "$GPU_CTR" >/dev/null 2>&1; rm -rf "$R"/Q/quality-n*/reference; (exit $rc); arms_cleanup' EXIT
cp "$PK/glm53_verify.py" "$HERE/" || { note "verify overlay FAILED"; exit 2; }
for f in Dockerfile.lowp-splitdev2 lowp.patch install_patch.py rebuild_native.py model_probe.py gpu_tests.py SHA256SUMS.patched; do
  [[ -f "$PK/lowp/$f" ]] || { note "packet missing $PK/lowp/$f"; exit 2; }
done
sudo -n docker image inspect "$BASE_IMG" >/dev/null 2>&1 || { note "base image $BASE_IMG missing"; exit 2; }
# The fixed IDs as model_probe wants them: {"sequences": {kind: [ids]}}, six kinds, >= 3,501 IDs each.
python3 - "$IDS" "$R/r893-ids.json" <<'PY' || { note "R893 fixed token IDs unusable in $IDS"; exit 2; }
import json, sys, pathlib, hashlib
src = sorted(pathlib.Path(sys.argv[1]).glob('*.tokens.json'))
seq = {p.name[:-len('.tokens.json')]: json.loads(p.read_text()) for p in src}
assert len(seq) == 6 and all(len(v) >= 3501 and all(isinstance(t, int) for t in v) for v in seq.values()), {k: len(v) for k, v in seq.items()}
pathlib.Path(sys.argv[2]).write_text(json.dumps({'sequences': seq}) + '\n')
print('ids', {k: len(v) for k, v in seq.items()}, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in src})
PY
note "disk before build: $(df -h / | tail -1)"
gpu_lock
note "GPU lock held; results $R; daily $DENV"

# Build after the lock (a native rebuild must not overlap another unit's measurement); plain docker build, legacy builder.
if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
  note "building $IMG FROM $BASE_IMG"
  timeout -k 30 3600 sudo -n docker build -f "$PK/lowp/Dockerfile.lowp-splitdev2" -t "$IMG" "$PK/lowp" > "$R/build.log" 2>&1 \
    || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 2; }
fi
note "image $IMG id $(sudo -n docker image inspect -f '{{.Id}}' "$IMG"); $(grep -a 'LANDING PASS' "$R/build.log" 2>/dev/null | tail -1 | cut -c1-200)"
# The serving path imports through exllamav3.ext (find_spec, then import): check that it lands on the rebuilt extensions.
timeout -k 10 120 sudo -n docker run --rm -w /app --entrypoint python3 "$IMG" -c \
  'import torch; from exllamav3.ext import exllamav3_ext as e; import exllamav3_lowp_ext as lp; print("IMPORTS", e.__file__, lp.__file__, lp.lowp_revision, lp.architecture)' \
  > "$R/import-check.log" 2>&1
note "import check rc=$? $(grep -a IMPORTS "$R/import-check.log" | cut -c1-200)"
grep -aq 'IMPORTS .*/opt/lowp-native/mxfp8/' "$R/import-check.log" || { note "VERDICT R924 lowp extension not importable on the serving path: no GPU arms"; exit 3; }

SLOT_START=$SECONDS
WORK_END=$((SECONDS + 3600 - 300))   # 60 min slot after the build, last 5 min reserved for the cleanup restore
DX=$(grep -oE 'EXL3_EXTRA=[^ ]+' <<<"$DENV" | head -1); DX=${DX#EXL3_EXTRA=}
CAPRE='Insufficient VRAM|out of memory|OutOfMemory'

# --- slot helpers (R920 conventions) ---
cap(){ local l=$((WORK_END - SECONDS)); ((l < 10)) && l=10; ((l < $1)) && echo "$l" || echo "$1"; }
have(){ (( WORK_END - SECONDS >= $1 )); }
aboot(){  # dir env...: glm_arms boot (env -i, daily defaults) with the launcher's health wait capped by the slot
  local d=$1; shift; local t=$((WORK_END - SECONDS - 60))
  ((t >= 120)) || { note "no slot time left to boot $d"; return 1; }
  ((t > 900)) && t=900
  boot "$d" "$HERE" "$@" BOOT_TIMEOUT=$t
}
vram(){  # tag stage -> FREE0 FREE1, VRAM_OK
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$R/$1/vram-$2.csv"
  read -r FREE0 FREE1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')
  VRAM_OK=0; [[ "$FREE0" =~ ^[0-9]+$ && "$FREE1" =~ ^[0-9]+$ ]] && ((FREE0 >= 450 && FREE1 >= 300)) && VRAM_OK=1
  note "$1 free MiB ($2): GPU0=$FREE0 GPU1=$FREE1 (gate 450/300: $( ((VRAM_OK)) && echo pass || echo FAIL))"
}
armup(){  # tag env...: the daily env + overrides (later assignment wins); boot, warmup, VRAM after warmup
  local tag=$1; shift; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  note "$tag TRY: $*"
  # shellcheck disable=SC2086
  aboot "$D/boot" $DENV "$@" \
    || { note "$tag NO BOOT ($(grep -ahoE "$CAPRE|ABORT[^;]{0,120}|NO BOOT[^;]{0,80}|Error[^;]{0,120}" "$D"/boot.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
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
measure(){  # tag: c1 five kinds x2 + server step, distinct c1/c2/c4 x2, prefill 8k/32k cold x2, vision, VRAM
  local tag=$1 D=$R/$1 l0 rc=0 i
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
  for i in 1 2; do  # own dir per run: the probe names its event files by tag (depth-8192), a second run would overwrite
    mkdir -p "$D/prefill-r$i"
    timeout -k 15 "$(cap 420)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
      --out "$D/prefill-r$i/prefill.jsonl" --phase c1 --runs 1 --kinds '' --depths 8192,32768 > "$D/prefill-r$i.log" 2>&1 \
      || note "$tag prefill r$i rc=$?"
  done
  note "$tag prefill cold: $(python3 -c 'import json,sys
for p in sys.argv[1:]:
  try:
    for l in open(p):
      r=json.loads(l)
      if r.get("phase")=="depth": print("%dk %.0f tok/s" % (r["target"]//1024, r["engine_prefill_tps"]), end="; ")
  except OSError: pass' "$D"/prefill-r1/prefill.jsonl "$D"/prefill-r2/prefill.jsonl 2>/dev/null)"
  timeout -k 15 "$(cap 300)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" --out "$D/vision.jsonl" \
    --phase vision > "$D/vision.log" 2>&1 || rc=$?
  note "$tag vision rc=$rc $(tail -1 "$D/vision.log" | cut -c1-160)"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  vram "$tag" final
  grep -aqE 'Traceback|illegal memory access|device-side assert|CUDA error|worker abort' "$D/engine.log" \
    && note "$tag REJECT: engine error in log ($(grep -aoE 'illegal memory access|device-side assert|CUDA error[^;]{0,80}|Traceback' "$D/engine.log" | head -1))" || true
}
lowpevidence(){  # tag: converted linears and bytes per device, budget skips (load-time; dispatch counters are in Q only)
  local D=$R/$1
  note "$1 LOWP: converted $(grep -ac 'LOWP converted' "$D/engine-boot.log"), budget skips $(grep -ac 'LOWP budget skip' "$D/engine-boot.log"), per-device total bytes $(grep -aoE 'total=[0-9]+ device=cuda:[0-9]+' "$D/engine-boot.log" | awk '{split($1,a,"=");split($2,b,"=");t[b[2]]=a[2]} END{for(k in t) printf "%s=%s ", k, t[k]}')"
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

# --- K: kernel tests on both GPUs, no model loaded ---
idle_gpus
mkdir -p "$R/K" "$R/Q"; chmod 0777 "$R/K" "$R/Q"
gpurun 480 "$R/K/gpu-tests.log" --gpus all --ipc=host -v "$R/K":/results -w /app -e EXL3_LOWP_MXFP8=1 \
  --entrypoint python3 "$IMG" /opt/lowp-tools/gpu_tests.py --output /results/kernels
KRC=$?
note "K kernel tests rc=$KRC $(cat "$R/K/kernels/PASS.json" 2>/dev/null) $(tail -1 "$R/K/gpu-tests.log" | cut -c1-200)"
((KRC == 0)) || { note "VERDICT R924 kernel tests FAIL (rc=$KRC): no KL, no speed"; exit 3; }

# --- OFF1: daily config on the lowp image, flag off (speed control + the template for the quality container) ---
armup OFF1 GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_LOWP_MXFP8=0" \
  || { note "VERDICT R924 the lowp image does not serve the daily config with the flag off"; exit 3; }
measure OFF1
SNAP=$R/OFF1/boot/served-inspect.json

# --- Q: teacher-forced KL in one process (one-shot container cloned from OFF1: same binds, same env args) ---
idle_gpus
QOK=1
python3 - "$SNAP" "$R/Q/binds.args" "$R/Q/env.args" <<'PY' || { note "Q SKIPPED: cannot clone $SNAP"; QOK=0; }
import json, sys
d = json.load(open(sys.argv[1]))[0]
assert d['Path'] == '/usr/bin/env' and 'python3' in d['Args'], 'launcher layout changed'
binds = d['HostConfig']['Binds'] or []
assert any(':/app/config.yml' in b for b in binds) and any(':/models/' in b for b in binds), binds
out = []
for b in binds: out += ['-v', b]
open(sys.argv[2], 'w').write('\0'.join(out) + '\0')
args, env, i = d['Args'], [], 0
while args[i] != 'python3':
    if args[i] == '-u': env += args[i:i + 2]; i += 2; continue
    # model_probe asserts a uniform split; the per-device override is dropped for the static quality load
    if not args[i].startswith(('EXL3_MOE_CPU_SPLIT_BY_DEVICE=', 'EXL3_LOWP_MXFP8=')): env.append(args[i])
    i += 1
assert any(a.startswith('EXL3_MOE_CPU_THREADS=') for a in env), env
open(sys.argv[3], 'w').write('\0'.join(env) + '\0')
PY
QBINDS=(); QENV=(); QN=
((QOK)) && { mapfile -d '' QBINDS < "$R/Q/binds.args"; mapfile -d '' QENV < "$R/Q/env.args"; }
QEND=$((SECONDS + 1260))   # Q gets <= 21 min in all, so the ON arm keeps ~15 min of the slot
for n in 104 106 108; do
  ((QOK)) || break
  QCAP=$((QEND - SECONDS))
  ((QCAP >= 240)) && have 300 || { note "Q N=$n UNTESTED: Q budget / slot deadline"; break; }
  note "Q quality N=$n (static uniform, off/on/off-repeat x 6 kinds x 3500, R893 IDs), cap ${QCAP}s"
  gpurun "$QCAP" "$R/Q/quality-n$n.log" --gpus all --ipc=host --shm-size=16g "${QBINDS[@]}" -v "$R":/r924 \
    -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache -w /app --entrypoint /usr/bin/env "$IMG" \
    "${QENV[@]}" EXL3_LOWP_MXFP8=1 PYTHONPATH=/app \
    python3 /opt/lowp-tools/model_probe.py --phase quality --arm lowp --n "$n" --output "/r924/Q/quality-n$n" \
    --tokens-json /r924/r893-ids.json
  QRC=$?
  rm -rf "$R/Q/quality-n$n/reference"   # spooled reference logits (GBs); the root fs must keep >= 15 % free (k3s)
  if ((QRC == 0)); then QN=$n; break; fi
  grep -aqE "$CAPRE" "$R/Q/quality-n$n.log" || { note "Q N=$n FAILED rc=$QRC (not capacity): $(tail -2 "$R/Q/quality-n$n.log" | tr '\n' ' ' | cut -c1-240)"; break; }
  note "Q N=$n capacity failure; next N"
done
python3 - "$R/Q" <<'PY' | while read -r l; do note "$l"; done
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob('quality-n*/quality.json')):
    q = json.loads(p.read_text()); c = q.get('candidate') or {}; k = q.get('control') or {}
    print(f"Q {p.parent.name}: status={q.get('status')} converted={q.get('converted')} KL mean={c.get('mean_kl')} "
          f"p99={c.get('p99_kl')} top1={c.get('top1')} | control KL mean={k.get('mean_kl')} top1={k.get('top1')} "
          f"stable={q.get('stable_control')} synthetic={(q.get('provenance') or {}).get('synthetic')}")
    for kind, v in (q.get('kinds') or {}).items():
        cc = v.get('candidate') or {}
        print(f"Q   {kind}: KL mean={cc.get('mean_kl', float('nan')):.5f} p99={cc.get('p99_kl', float('nan')):.4f} "
              f"top1={cc.get('top1', float('nan')):.4f} control byte-identical={v.get('control_byte_identical')}")
PY

# --- ON: flag on at the lowest split that boots and passes the VRAM gate ---
idle_gpus
ONTAG=
for split in 100,104 100,106 100,108; do
  n1=${split#*,}; tag=ON-n${split/,/-}
  have 600 || { note "$tag UNTESTED: slot deadline"; break; }
  if armup "$tag" GLM_IMG="$IMG" OFFLOAD_N="$n1" "EXL3_EXTRA=$DX;EXL3_LOWP_MXFP8=1;EXL3_MOE_CPU_SPLIT_BY_DEVICE=$split"; then
    lowpevidence "$tag"
    if ((VRAM_OK)); then ONTAG=$tag; measure "$tag"; break; fi
    note "$tag boots but fails the VRAM gate; next split"
  else
    grep -aqE "$CAPRE" "$R/$tag"/boot.log "$R/$tag"/boot/*.log 2>/dev/null || { note "$tag failed for a non-capacity reason: stop the ladder"; break; }
  fi
done

if have 900; then armup OFF2 GLM_IMG="$IMG" "EXL3_EXTRA=$DX;EXL3_LOWP_MXFP8=0" && measure OFF2
else note "OFF2 UNTESTED: slot deadline (OFF drift unbounded)"; fi

python3 - "$R" "$ONTAG" "$QN" <<'PY' | while read -r l; do note "$l"; done
import csv, json, re, statistics as st, sys, pathlib
R, ontag, qn = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def arm(tag):
    d, out = R / tag, {}
    if not tag or not d.is_dir(): return out
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
                out.update({f"c{s['c']}": s['ss_agg_tps_median'] for s in r['summaries']})
    except Exception: pass
    try:
        m = re.search(r'ms/tok=([0-9.]+)', (d / 'server-step.txt').read_text())
        if m: out['ms/tok'] = float(m.group(1))
    except Exception: pass
    pf = {}
    for p in d.glob('prefill-r*/prefill.jsonl'):
        for l in open(p):
            r = json.loads(l)
            if r.get('phase') == 'depth': pf.setdefault(r['target'], []).append(r['engine_prefill_tps'])
    for t, v in pf.items(): out[f'prefill{t // 1024}k'] = st.median(v)
    try:
        rows = list(csv.reader(open(d / 'vram-final.csv')))[1:]
        out['free'] = '/'.join(r[2].strip().split()[0] for r in rows)
    except Exception: pass
    return out
q = {}
if qn:
    try: q = json.loads((R / 'Q' / f'quality-n{qn}' / 'quality.json').read_text())
    except Exception: q = {}
c, k = q.get('candidate') or {}, q.get('control') or {}
f = lambda x, n: f'{x:.{n}f}' if isinstance(x, (int, float)) else 'NA'
kl = (f"quality={q.get('status', 'NA')} KL_mean={f(c.get('mean_kl'), 5)} KL_p99={f(c.get('p99_kl'), 4)} "
      f"top1={f(c.get('top1'), 4)} (control KL {f(k.get('mean_kl'), 5)}, stable={q.get('stable_control')}, N={qn or 'NA'}; "
      f"gate 0.001/0.01/0.995, R893 floor 0.00048/0.015/0.9959)")
offs = [a for a in (arm('OFF1'), arm('OFF2')) if a]; on = arm(ontag)
parts, gains = [], {}
for key in ('c1 score', 'c1', 'c2', 'c4', 'ms/tok', 'prefill8k', 'prefill32k'):
    xs = [a[key] for a in offs if key in a]
    if key not in on or not xs: parts.append(f'{key} NA'); continue
    m = st.mean(xs); g = on[key] / m - 1; gains[key] = g
    nd = 2 if key == 'ms/tok' else 1
    parts.append(f'{key} {on[key]:.{nd}f} vs {m:.{nd}f} ({g:+.1%})')
n_on = ontag.split('-n')[-1].replace('-', ',') if ontag else 'none'
free = f"free MiB GPU0/GPU1 ON {on.get('free', 'NA')} vs OFF1 {offs[0].get('free', 'NA') if offs else 'NA'}"
screen = 'NA'
if q.get('status') and all(x in gains for x in ('c1', 'c2', 'c4', 'prefill8k', 'prefill32k')):
    ok = (q['status'] == 'PASS' and max(gains['prefill8k'], gains['prefill32k']) >= .02
          and min(gains['c1'], gains['c2'], gains['c4']) >= -.01)
    screen = 'candidate for the USER (numerics change)' if ok else 'below the screen'
print(f'VERDICT R924 lowp MXFP8 o_proj prefill: {kl} | speed ON(N {n_on}) vs OFF(N 100,104, n={len(offs)}): '
      + '; '.join(parts) + f' | {free} | screen: {screen}')
PY
note "slot used $(( (SECONDS - SLOT_START) / 60 )) min; no promotion: numerics-changing, serving is the user's call"
exit 0

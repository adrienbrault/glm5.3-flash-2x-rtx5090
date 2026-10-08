#!/usr/bin/env bash
# R927 (2026-10-08): DFlash2-only with the drafter KV bounded to its 2,048-token window (codex glm-dflash-swa-r1,
# EXL3_DRAFT_WINDOW_CACHE=1): Q8 drafter KV 106 MiB on GPU (+2.7 GB unpinned host backing for exact prefix reuse) instead
# of 2.7 GB. R921b showed the full-context drafter cache was what did not fit. Drafter budget 1.0 GB on GPU0, main GPU0
# 30.8; uniform N searched 104..120 (dflash1 parent, no per-device split); daily control B-D in the same slot.
# Check the engine log for the allocation line (4 slots x 10 pages, GPU KV 106.25 MiB) before trusting any arm.
#   sudo systemd-run --unit=r927-glm53-dflash-window --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r927-glm53-dflash-window.sh
# --- derived from R921b ---
# R921b (2026-10-08): R921's DFlash-only arm reserved DRAFT_GPU_SPLIT=6,0 for a 610 MB drafter (5 sliding-window layers,
# window 2048, 8 KV heads; Q8 cache <= 2.6 GB even if allocated for the full 262k) on top of the daily's GPU_SPLIT 31.8,31,
# so GPU0 was oversubscribed: "Insufficient VRAM in split". Here the drafter gets 1.5 GB on GPU0 and the main model's GPU0
# budget drops to 30.3. N searched upward 104,108,112,116,120 (dflash1 image: uniform N, no per-device split); first N
# with >= 450 / 300 MiB free is measured, then the daily (B-D) in the same slot. Reuses R921's packet in /srv/qwen5090/r921.
#   sudo systemd-run --unit=r921b-glm53-dflash-only --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r921b-glm53-dflash-only.sh
# --- R921 header follows ---
# R921 (2026-10-08): codex-glm-dflash-adaptive-r1. DFlash2 drafter (incoai GLM-5.3-Flash-DFlash2, EXL3 3.00 bpw, 610 MiB,
# /srv/qwen5090/models/glm53-flash-dflash2-exl3-3.00bpw-r0b0tlab) next to the MTP layer, with a cost-aware per-step choice
# among plain / MTP-1 / DFlash prefix 1..5 (user: "use adaptive verify, not always verify 6 rows"), lossless (target
# verifies), no speculation with >= 2 active requests (today's MTP-cap rule). Image
# tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_dflash1 (Python-only overlay on the daily image). Packet:
# patches/exllamav3/glm-dflash-adaptive-r1/out (OPERATOR.md, GPU-TESTS.md; BRIEF.md one level up, incl. the addendum).
#
# Deviations from OPERATOR.md, on purpose:
#   - tools/calibration_slot.sh and compare_slot.sh are not used: they hardcode BOOT_TIMEOUT=180 / 360 s per arm (one boot
#     is ~2.3 min plus the drafter), take no slot deadline, rely on the operator exporting the daily's CACHE_TOKENS/
#     MAX_SEQ/CHUNK/GPU_SPLIT, and compare_slot runs c4 with R858_ALWAYS_THINK=1 (not comparable with R914/R915). This unit
#     drives the same arms with glm_arms boot (daily defaults: cache 262144, chunk 2048, batch 4) and keeps the packet's
#     exact flag strings, probe calls (--salt dflash-cal-r1), mtp_steps.py and calibrate.py/summarize.py.
#   - apply_launcher.py is NOT run against glm-daily-tools: the patched launch-glm53.sh/glm53_plan.py (DRAFT_MODEL,
#     DRAFT_GPU_SPLIT, DFLASH_COST_TABLE) are copied over this unit's own $HERE copies after their sha256 'before'
#     hashes are checked. The daily restore keeps using the unpatched daily tools.
#   - glm53_verify.py (repo flan/r858, copied into $HERE) accepts layers 3..44 when DRAFT_MODEL is set without
#     EXL3_DRAFTER_CHOICE=1: draft_mode model loads no MTP layer 45, and the old check would fail a healthy DFlash-only boot.
#   - The DFlash-only arm sets MTP_FAST=0 (no MTP layer to join the trunk worker); everything else is the daily.
#
# Arms (base = glm-daily.env; drafter env = DRAFT=1 DRAFT_MODEL=<drafter> DRAFT_N=5 DRAFT_GPU_SPLIT=6,0 on GPU0):
#   build  after gpu_lock if missing (fast: Python overlay + import landing).
#   C  (addendum, <= 45 min) DFlash2-only, MTP off, stock draft_mode model (no EXL3_DFLASH_* flags): N=104 first (if it
#      does not boot, nothing lower can); if it boots with >= 450 MiB free on EACH GPU after warmup, N=96..103 ascending for
#      the lowest that does. The chosen arm (or N=104 if it only boots there) gets c1 five kinds x2 + server step,
#      distinct c1/c2/c4 x2, vision, VRAM; drafter cache/scratch lines from the engine log. Compared with B-D (MTP daily).
#   A  (<= 45 min) calibration, coexist profile (MTP + DFlash both loaded, K=5 history in every arm), fixed arms
#      EXL3_DFLASH_FIXED=dflash:5 (first = the feasibility boot), plain, mtp, dflash:1..4, each a fresh container, c1 five
#      kinds x1 fixed salt, EXL3_DFLASH_TRACE=1, mtp_steps per arm, then calibrate.py -> A/costs.json (all 7 or no table).
#      Feasibility: N=104; if it does not boot, labelled pilots N=112, 120 (DESIGN expects K=5 + 262k Q8 draft KV not to
#      fit at 104); the table's placement id carries the N.
#   B  (<= 45 min) B-D the real daily; B-A adaptive + choice (EXL3_DFLASH_ADAPTIVE=1;EXL3_DRAFTER_CHOICE=1;
#      EXL3_DFLASH_TRACE=0;EXL3_DFLASH_PLACEMENT=<id>, cost table mounted) at the calibrated N, only with a complete table;
#      B-DN the daily config at the calibrated N (matched placement), only if that N != 104. Each: c1 five kinds x2 +
#      server step, distinct c4 x2 (log snapshot for summarize.py), then distinct c1/c2 x2, vision, VRAM.
# Pass/fail: promotable only if B-A beats B-D on c1 server ms/token with distinct c4 within B-D's repeat spread, at N=104
#   (or at a higher N that still beats the real daily end-to-end), vision OK, headroom GPU0 >= 450 / GPU1 >= 300; plus the
#   GPU-TESTS.md exactness gates (static-placement plain A/A2 identity, observer IDs) that this unit does NOT run. The
#   operator (not this unit) writes winners into glm-daily.env. C answers "does DFlash2 cost less VRAM than MTP" (MTP costs
#   ~2 GiB: N=96 without, 104 with).
# PK=/srv/qwen5090/r921 must hold: glm_arms.sh launch-glm53.sh glm53_plan.py glm53_probe.py glm53_verify.py mtp_steps.py
#   (from flan/r858/, glm53_verify.py with the DRAFT_MODEL change) and dflash/ = the whole
#   patches/exllamav3/glm-dflash-adaptive-r1/out/ directory (Docker context, launcher/, tools/).
# Expected wall: queue + build (~3 min) + C <= 45 + A <= 45 + B <= 45 + restore ~3 = ~2.5 h.
#   sudo systemd-run --unit=r921-glm53-dflash-adaptive --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r921-glm53-dflash-adaptive.sh
set -uo pipefail
: "${HOME:=/root}"
UNIT=r927
PK=/srv/qwen5090/r921
HERE=/srv/qwen5090/r927-tools   # the unit's OWN dir: arms_init does rm -rf "$HERE"
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_dflash1_swa3   # codex FIX1: lookup counters 0..7 in source (run 1) + contiguous selector slice (run 2)
PARENT=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_dflash1
SWA=/srv/qwen5090/r927/swa
MODEL=glm53-flash-exl3-2.05bpw-turboderp
DRAFTER=/srv/qwen5090/models/glm53-flash-dflash2-exl3-3.00bpw-r0b0tlab
R=/srv/qwen5090/results/$(date +%F)-r927-glm53-dflash-window-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
export GPU_QUEUE_NAME=r927-glm53-dflash-window
. /srv/qwen5090/lib/gpu-queue.sh
. "$PK/glm_arms.sh"
arms_init
cp "$PK/glm53_verify.py" "$HERE/" || { note "verify overlay FAILED"; exit 2; }
grep -q 'EXL3_DRAFTER_CHOICE=1' "$HERE/glm53_verify.py" || { note "glm53_verify.py lacks the DRAFT_MODEL change"; exit 2; }
[[ -f "$PK/dflash/Dockerfile" && -f "$PK/dflash/tools/calibrate.py" && -f "$DRAFTER/config.json" ]] \
  || { note "packet in $PK/dflash or drafter $DRAFTER missing"; exit 2; }
python3 - "$HERE" "$PK/dflash" <<'PY' || { note "patched launcher install FAILED"; exit 2; }
import hashlib, json, pathlib, shutil, sys
here, pk = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
m = json.loads((pk / 'launcher-SHA256.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for rel, h in m.items():
    assert sha(here / rel) == h['before'], f'{rel}: unit copy differs from the packet base (launcher changed since the round)'
    assert sha(pk / 'launcher' / rel) == h['after'], f'{rel}: packet launcher differs from its manifest'
for rel in m:
    shutil.copyfile(pk / 'launcher' / rel, here / rel)
print('patched launcher/plan installed in', here)
PY
bash -n "$HERE/launch-glm53.sh" || { note "patched launcher syntax FAILED"; exit 2; }
gpu_lock
note "GPU lock held; results $R; daily $DENV"

if ! sudo -n docker image inspect "$IMG" >/dev/null 2>&1; then
  note "building $IMG"
  sudo -n docker image inspect "$PARENT" >/dev/null 2>&1 || { note "parent $PARENT missing (built by R921)"; exit 2; }
  timeout -k 30 1800 sudo -n docker build -f "$SWA/Dockerfile" -t "$IMG" "$SWA" > "$R/build.log" 2>&1 \
    || { note "build FAILED: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 2; }
fi
note "image $IMG id $(sudo -n docker image inspect -f '{{.Id}}' "$IMG")"
DX=$(grep -oE 'EXL3_EXTRA=[^ ]+' <<<"$DENV" | head -1); DX=${DX#EXL3_EXTRA=}
# dflash1 has no per-device split: drop the daily's EXL3_MOE_CPU_SPLIT_BY_DEVICE so glm53_verify expects the arm's uniform N
DX=$(tr ';' '\n' <<< "$DX" | grep -v '^EXL3_MOE_CPU_SPLIT_BY_DEVICE=' | paste -sd';' -)
DRAFTENV=(GLM_IMG="$IMG" DRAFT=1 DRAFT_MODEL="$DRAFTER" DRAFT_N=5 DRAFT_GPU_SPLIT=1.0,0)
UNIT_START=$SECONDS

# --- slot helpers (WORK_END is set per slot) ---
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
STAGE=
armup(){  # tag env...: the daily env + overrides (later assignment wins); boot, warmup, VRAM after warmup
  local tag=$1; shift; local D=$R/$tag
  mkdir -p "$D"; sudo -n docker rm -f glm53 >/dev/null 2>&1 || true; sleep 3
  STAGE=boot; note "$tag TRY: $*"
  # shellcheck disable=SC2086
  aboot "$D/boot" $DENV "$@" \
    || { note "$tag NO BOOT ($(grep -ahoE 'Insufficient VRAM|out of memory|OutOfMemory|ABORT[^;]{0,120}|NO BOOT[^;]{0,80}|Error[^;]{0,120}' "$D"/boot.log "$D"/boot/*.log 2>/dev/null | head -1))"; return 1; }
  sudo -n docker inspect glm53 --format '{{range .Config.Env}}{{println .}}{{end}}' > "$D/container-env.txt"
  sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1
  STAGE=warmup
  timeout -k 10 "$(cap 600)" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" \
    --out "$D/warmup.jsonl" --phase warmup > "$D/warmup.log" 2>&1 \
    || { sudo -n docker logs glm53 > "$D/engine-boot.log" 2>&1; note "$tag WARMUP FAILED: $(tail -1 "$D/warmup.log" | cut -c1-160)"; return 1; }
  STAGE=up; vram "$tag" warmup
}
decsum(){ python3 -c 'import json,sys
for l in open(sys.argv[1]):
  r=json.loads(l)
  if r.get("phase")=="decode-summary": print(" ".join("c%s %.1f" % (s["c"], s["ss_agg_tps_median"]) for s in r["summaries"]))' "$1" 2>/dev/null | tail -1; }
probe(){ timeout -k 15 "$(cap "$1")" python3 -u "$HERE/glm53_probe.py" --url http://127.0.0.1:8029 --model "$MODEL" "${@:2}"; }
measure(){  # tag: c1 five kinds x2 + server step; distinct c4 x2 (snapshot for summarize.py); distinct c1/c2 x2; vision; VRAM
  local tag=$1 D=$R/$1 l0 rc=0
  l0=$(sudo -n docker logs glm53 2>&1 | wc -l)
  probe 900 --out "$D/c1.jsonl" --phase c1 --runs 2 --kinds code,prose,chat,html,edit > "$D/c1.log" 2>&1 || note "$tag c1 rc=$?"
  sudo -n docker logs glm53 2>&1 | tail -n +"$((l0 + 1))" > "$D/engine-c1.log"
  python3 "$HERE/r860_score.py" "$D/c1.jsonl" "$tag" | tee -a "$R/summary.txt"
  note "$tag server step: $(python3 "$HERE/mtp_steps.py" "$D/engine-c1.log" 10 2>&1 | tail -1 | cut -c1-200)"
  probe 600 --out "$D/dec4.jsonl" --phase decode --concurrency 4 --runs 2 --distinct > "$D/dec4.log" 2>&1 || note "$tag c4 rc=$?"
  sudo -n docker logs glm53 2>&1 | tail -n +"$((l0 + 1))" > "$D/engine-summ.log"   # c1 x10 then c4 groups of 4 only
  probe 600 --out "$D/dec12.jsonl" --phase decode --concurrency 1,2 --runs 2 --distinct > "$D/dec12.log" 2>&1 || note "$tag c1/c2 rc=$?"
  note "$tag decode distinct: $(decsum "$D/dec12.jsonl") $(decsum "$D/dec4.jsonl")"
  probe 300 --out "$D/vision.jsonl" --phase vision > "$D/vision.log" 2>&1 || rc=$?
  note "$tag vision rc=$rc $(tail -1 "$D/vision.log" | cut -c1-160)"
  sudo -n docker logs glm53 > "$D/engine.log" 2>&1
  vram "$tag" final
  grep -aqE 'Traceback|illegal memory access|device-side assert|CUDA error' "$D/engine.log" && note "$tag REJECT: engine error in log" || true
}
draftlines(){  # tag: what the engine says about the drafter's weights/cache/scratch
  grep -aiE 'draft|dflash' "$R/$1/engine-boot.log" | head -60 > "$R/$1/draft-lines.txt"
  note "$1 drafter lines: $(grep -aoiE '(draft|dflash).{0,100}[0-9.]+ ?(MiB|GiB|MB|GB|bytes)' "$R/$1/draft-lines.txt" | head -3 | tr '\n' ' ' | cut -c1-300)"
}

# --- C: DFlash2-only (MTP off), drafter 1.5 GB on GPU0, lowest N in 104..120 that fits ---
WORK_END=$((SECONDS + 2700))
CENV=("${DRAFTENV[@]}" MTP_FAST=0 GPU_SPLIT=30.8,31 "EXL3_EXTRA=$DX;EXL3_DRAFT_WINDOW_CACHE=1")
CPICK=
for n in 104 108 112 116 120; do
  have 900 || { note "C search stopped before N=$n: slot deadline"; break; }
  if armup "C-n$n" "${CENV[@]}" OFFLOAD_N=$n; then
    draftlines "C-n$n"
    if ((FREE0 >= 450 && FREE1 >= 300)); then CPICK=$n; break; fi
    note "C-n$n boots but GPU0 $FREE0 / GPU1 $FREE1 MiB free (< 450/300)"
  fi
done
if [[ -n $CPICK ]]; then
  note "VERDICT R927 C DFlash-only: lowest N with headroom = $CPICK"
  measure "C-n$CPICK"
else
  note "VERDICT R927 C DFlash-only: no N in 104..120 boots with headroom (see C-n*/boot/)"
fi
# --- B-D: the daily in the same slot ---
WORK_END=$((SECONDS + 1200))
if have 780; then armup B-D && measure B-D; else note "B-D UNTESTED: slot deadline"; fi
note "C (DFlash-only at N=${CPICK:-none}) vs B-D (MTP daily 100/104): compare c1 score, server ms/tok, distinct c1/c2/c4 above"
sudo -n docker rm -f glm53 >/dev/null 2>&1 || true
note "unit used $(( (SECONDS - UNIT_START) / 60 )) min after the build"
exit 0

#!/usr/bin/env bash
# On flan with the existing GPU lock held. Sequential boots, same static N=96,
# MTP off, R877 config/placement. Restores the existing glm53 container on exit.
set -euo pipefail
R4_PACKAGE=$(cd "$(dirname "$0")" && pwd)
R4_RESULTS=${1:?Usage: operator-run.sh /absolute/path/to/new-results}
mkdir -p "$R4_RESULTS"
R4_RESULTS=$(cd "$R4_RESULTS" && pwd)
R4_MODEL=glm53-flash-exl3-2.05bpw-turboderp
R4_A2_IMAGE=${R4_A2_IMAGE:-tabbyapi:r861-glm-agent-r2}
R4_CONFIG="$R4_PACKAGE/operator/config.yml"
R4_SPLIT=/srv/qwen5090/r877-tools/split-stats-broad-r869.json
R4_CACHE=/srv/qwen5090/.exl3cache-r858-glm53
R4_MODEL_DIR=/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp
for R4_PATH in "$R4_CONFIG" "$R4_SPLIT" "$R4_CACHE" "$R4_MODEL_DIR"; do test -e "$R4_PATH"; done
docker image inspect "$R4_A2_IMAGE" tabbyapi:r861-glm-agent-r4 > "$R4_RESULTS/images.json"
docker inspect glm53 > "$R4_RESULTS/original-container.json"
test "$(docker inspect --format '{{.State.Running}}' glm53)" = true
if docker container inspect glm53-r4-probe >/dev/null 2>&1; then
    echo 'Container glm53-r4-probe already exists; choose a clean probe name before running.' >&2; exit 2
fi
R4_ORIGINAL_STOPPED=0
r4_restore() {
    if [[ -n ${R4_DEST:-} ]] && docker container inspect glm53-r4-probe >/dev/null 2>&1; then
        docker logs glm53-r4-probe > "$R4_DEST/engine.log" 2>&1 || true
    fi
    docker rm -f glm53-r4-probe >/dev/null 2>&1 || true
    if [[ $R4_ORIGINAL_STOPPED == 1 ]]; then docker start glm53 >/dev/null; fi
}
trap r4_restore EXIT
R4_ORIGINAL_STOPPED=1
docker stop glm53 >/dev/null
R4_MATRIX_RC=0
for R4_ARM in r2 r4; do
    R4_DEST="$R4_RESULTS/$R4_ARM"
    R4_IMAGE=tabbyapi:r861-glm-agent-r4
    if [[ $R4_ARM == r2 ]]; then R4_IMAGE="$R4_A2_IMAGE"; fi
    mkdir -p "$R4_DEST"
    cp "$R4_CONFIG" "$R4_DEST/config.yml"
    docker run -d --name glm53-r4-probe --gpus all --ipc=host --shm-size=16g \
        --security-opt label=disable -p 127.0.0.1:8029:8029 \
        -v "$R4_CONFIG:/app/config.yml:ro" -v "$R4_SPLIT:/app/split-stats.json:ro" \
        -v "$R4_CACHE:/exl3-cache" -v "$R4_MODEL_DIR:/models/$R4_MODEL:ro" \
        --entrypoint /usr/bin/env "$R4_IMAGE" \
        -u EXL3_MOE_COOP_V2 -u EXL3_SHARED_EXPERT_OVERLAP -u EXL3_DECODE_OVERLAP \
        -u EXL3_DECODE_FUSE -u EXL3_MOE_PREFILL_E3 -u EXL3_EMBED_GPU_PRUNED \
        -u EXL3_CACHE_TRACE -u EXL3_PREFILL_WHOLE_PROMPT -u EXL3_PREFILL_RESUMABLE \
        -u EXL3_PROMPT_LOOKUP EXL3_MOE_CPU_THREADS=8 EXL3_MOE_PINNED_ARENA=0 \
        PYTHONUNBUFFERED=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT_STATS=/app/split-stats.json \
        TABBY_STREAM_TOOLCALLS=1 TABBY_SSE_KEEPALIVE_S=5 TABBY_GLM_TOOL_FIXES=1 \
        TABBY_GLM_FORCING=0 TABBY_GLM_TAG_TRACE=1 \
        python3 main.py --host 0.0.0.0 --port 8029 --disable-auth true > "$R4_DEST/container-id.txt"
    docker inspect glm53-r4-probe > "$R4_DEST/inspect.json"
    python3 - "$R4_MODEL" <<'PY'
import json,sys,time,urllib.request
for _ in range(900):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8029/v1/models',timeout=3) as r:
            if r.status==200 and any(m.get('id')==sys.argv[1] for m in json.load(r).get('data',[])):break
    except OSError:pass
    time.sleep(2)
else:raise SystemExit('Server was not ready within 30 minutes')
PY
    # Exact required matrix: 4 cases x zero/default x 2 repeats = 16 per image.
    set +e
    python3 "$R4_PACKAGE/glm53_toolcheck_r4.py" --url http://127.0.0.1:8029/v1 \
        --model "$R4_MODEL" --out "$R4_DEST/toolcheck" \
        --cases hard think-only close-only 1000-line --modes stream \
        --temperatures zero default --repeats 2 --max-tokens 12000 --long-max-tokens 40000 \
        | tee "$R4_DEST/toolcheck.log"
    R4_RC=${PIPESTATUS[0]}
    set -e
    if [[ $R4_ARM == r4 ]]; then R4_MATRIX_RC=$R4_RC; fi
    printf '%s\n' "$R4_RC" > "$R4_DEST/toolcheck.exit-code"
    if [[ $R4_ARM == r2 ]]; then
        # Original R877 prompt, plain response: exposes eos_reason / stop_str.
        set +e
        python3 "$R4_PACKAGE/glm53_toolcheck.py" --url http://127.0.0.1:8029/v1 \
            --model "$R4_MODEL" --out "$R4_DEST/original-plain-probe" \
            --cases hard think-only close-only --modes plain --temperatures zero --repeats 2 \
            | tee "$R4_DEST/original-plain-probe.log"
        printf '%s\n' "${PIPESTATUS[0]}" > "$R4_DEST/original-plain-probe.exit-code"
        set -e
    else
        docker cp "$R4_PACKAGE/tests/probe_literals.py" glm53-r4-probe:/tmp/glm-probe-literals.py
        docker exec glm53-r4-probe python3 /tmp/glm-probe-literals.py \
            "/models/$R4_MODEL" --app /app --out /tmp/glm-literals.json \
            > "$R4_DEST/literal-probe.log"
        docker cp glm53-r4-probe:/tmp/glm-literals.json "$R4_DEST/literal-probe.json"
    fi
    docker logs glm53-r4-probe > "$R4_DEST/engine.log" 2>&1
    docker rm -f glm53-r4-probe >/dev/null
done
exit "$R4_MATRIX_RC"

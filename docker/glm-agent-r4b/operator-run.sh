#!/usr/bin/env bash
# On flan with the existing GPU lock held. Sequential boots, same static N=96,
# MTP off, R880 config/placement. Restores the existing glm53 container on exit.
set -euo pipefail
R4B_PACKAGE=$(cd "$(dirname "$0")" && pwd)
R4B_RESULTS=${1:?Usage: operator-run.sh /absolute/path/to/new-results}
test ! -e "$R4B_RESULTS"
mkdir -p "$R4B_RESULTS"
R4B_RESULTS=$(cd "$R4B_RESULTS" && pwd)
R4B_MODEL=glm53-flash-exl3-2.05bpw-turboderp
R4B_ARMS=${R4B_ARMS:-"r4 r4b"}
R4B_LITERAL_MODE=${R4B_LITERAL_MODE:-runs}
R4B_CACHE_VERIFY=${R4B_CACHE_VERIFY:-0}
R4B_CONFIG="$R4B_PACKAGE/operator/config.yml"
R4B_SPLIT=/srv/qwen5090/r880-tools/split-stats-broad-r869.json
R4B_CACHE=/srv/qwen5090/.exl3cache-r858-glm53
R4B_MODEL_DIR=/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp
for R4B_PATH in "$R4B_CONFIG" "$R4B_SPLIT" "$R4B_CACHE" "$R4B_MODEL_DIR"; do test -e "$R4B_PATH"; done
docker image inspect tabbyapi:r861-glm-agent-r4 tabbyapi:r861-glm-agent-r4b > "$R4B_RESULTS/images.json"
docker inspect glm53 > "$R4B_RESULTS/original-container.json"
test "$(docker inspect --format '{{.State.Running}}' glm53)" = true
if docker container inspect glm53-r4b-probe >/dev/null 2>&1; then
    echo 'Container glm53-r4b-probe already exists; choose a clean probe name before running.' >&2; exit 2
fi
R4B_ORIGINAL_STOPPED=0
r4b_restore() {
    if [[ -n ${R4B_DEST:-} ]] && docker container inspect glm53-r4b-probe >/dev/null 2>&1; then
        docker logs glm53-r4b-probe > "$R4B_DEST/engine.log" 2>&1 || true
    fi
    docker rm -f glm53-r4b-probe >/dev/null 2>&1 || true
    if [[ $R4B_ORIGINAL_STOPPED == 1 ]]; then docker start glm53 >/dev/null; fi
}
trap r4b_restore EXIT
R4B_ORIGINAL_STOPPED=1
docker stop glm53 >/dev/null
R4B_MATRIX_RC=0
for R4B_ARM in $R4B_ARMS; do
    R4B_DEST="$R4B_RESULTS/$R4B_ARM"
    R4B_IMAGE=tabbyapi:r861-glm-agent-r4b
    if [[ $R4B_ARM == r4 ]]; then R4B_IMAGE=tabbyapi:r861-glm-agent-r4; fi
    [[ $R4B_ARM == r4 || $R4B_ARM == r4b ]] || exit 2
    mkdir -p "$R4B_DEST"
    cp "$R4B_CONFIG" "$R4B_DEST/config.yml"
    docker run -d --name glm53-r4b-probe --gpus all --ipc=host --shm-size=16g \
        --security-opt label=disable -p 127.0.0.1:8029:8029 \
        -v "$R4B_CONFIG:/app/config.yml:ro" -v "$R4B_SPLIT:/app/split-stats.json:ro" \
        -v "$R4B_CACHE:/exl3-cache" -v "$R4B_MODEL_DIR:/models/$R4B_MODEL:ro" \
        --entrypoint /usr/bin/env "$R4B_IMAGE" \
        -u EXL3_MOE_COOP_V2 -u EXL3_SHARED_EXPERT_OVERLAP -u EXL3_DECODE_OVERLAP \
        -u EXL3_DECODE_FUSE -u EXL3_MOE_PREFILL_E3 -u EXL3_EMBED_GPU_PRUNED \
        -u EXL3_PREFILL_WHOLE_PROMPT -u EXL3_PREFILL_RESUMABLE \
        -u EXL3_PROMPT_LOOKUP EXL3_MOE_CPU_THREADS=8 EXL3_MOE_PINNED_ARENA=0 \
        PYTHONUNBUFFERED=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SPLIT_STATS=/app/split-stats.json \
        TABBY_STREAM_TOOLCALLS=1 TABBY_SSE_KEEPALIVE_S=5 TABBY_GLM_TOOL_FIXES=1 \
        TABBY_GLM_FORCING=0 TABBY_GLM_TAG_TRACE=1 EXL3_CACHE_TRACE=1 \
        TABBY_GLM_LITERAL_ENCODING="$R4B_LITERAL_MODE" TABBY_GLM_CACHE_VERIFY="$R4B_CACHE_VERIFY" \
        python3 main.py --host 0.0.0.0 --port 8029 --disable-auth true > "$R4B_DEST/container-id.txt"
    docker inspect glm53-r4b-probe > "$R4B_DEST/inspect.json"
    python3 - "$R4B_MODEL" <<'PY'
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
    python3 "$R4B_PACKAGE/glm53_toolcheck_r4.py" --url http://127.0.0.1:8029/v1 \
        --model "$R4B_MODEL" --out "$R4B_DEST/toolcheck" \
        --cases hard think-only close-only 1000-line --modes stream \
        --temperatures zero default --repeats 2 --max-tokens 12000 --long-max-tokens 40000 \
        | tee "$R4B_DEST/toolcheck.log"
    R4B_RC=${PIPESTATUS[0]}
    set -e
    if [[ $R4B_ARM == r4b ]]; then R4B_MATRIX_RC=$R4B_RC; fi
    printf '%s\n' "$R4B_RC" > "$R4B_DEST/toolcheck.exit-code"
    # Three fresh requests in sequence at each temperature exercise repeated prompts.
    set +e
    python3 "$R4B_PACKAGE/glm53_toolcheck_r4.py" --url http://127.0.0.1:8029/v1 \
        --model "$R4B_MODEL" --out "$R4B_DEST/cache-repeat" \
        --cases 1000-line --modes stream --temperatures zero default --repeats 3 \
        --max-tokens 12000 --long-max-tokens 40000 | tee "$R4B_DEST/cache-repeat.log"
    R4B_REPEAT_RC=${PIPESTATUS[0]}
    printf '%s\n' "$R4B_REPEAT_RC" > "$R4B_DEST/cache-repeat.exit-code"
    python3 "$R4B_PACKAGE/tests/replay_vision.py" --url http://127.0.0.1:8029/v1 \
        --out "$R4B_DEST/vision" | tee "$R4B_DEST/vision.log"
    R4B_VISION_RC=${PIPESTATUS[0]}
    printf '%s\n' "$R4B_VISION_RC" > "$R4B_DEST/vision.exit-code"
    set -e
    if [[ $R4B_ARM == r4b ]]; then
        if [[ $R4B_REPEAT_RC != 0 || $R4B_VISION_RC != 0 ]]; then R4B_MATRIX_RC=1; fi
        docker exec glm53-r4b-probe python3 /opt/r4b/tests/probe_prompt_cpu.py \
            "/models/$R4B_MODEL" --app /app --out /tmp/glm-prompt-cpu.json \
            > "$R4B_DEST/prompt-cpu.log" || R4B_MATRIX_RC=1
        docker cp glm53-r4b-probe:/tmp/glm-prompt-cpu.json "$R4B_DEST/prompt-cpu.json"
    fi
    docker logs glm53-r4b-probe > "$R4B_DEST/engine.log" 2>&1
    docker rm -f glm53-r4b-probe >/dev/null
done
exit "$R4B_MATRIX_RC"

#!/usr/bin/env bash
# R881 (2026-10-07): agent overlay r4b vs r4. R880: r4 ended all 16 cases in valid write_file calls with tags intact (r2: 0/16,
# turns lost on the model's <|user|>), but doubled spaces around </fake> (12 cases) and 1000-line r02 mismatches kept it at
# 2/16. r4b (codex, patches/tabbyapi/glm-agent-r4b): TABBY_GLM_LITERAL_ENCODING=runs (contiguous ordinary BPE around literal
# tags), TABBY_GLM_CACHE_VERIFY option, deeper tag trace. Runs codex's operator-run.sh unchanged under the GPU lock:
# r4 then r4b, same R880 static config, 16-case matrix + 3-repeat 1000-line per arm, 4 vision requests, CPU prompt/cache probe;
# it stops the serving glm53 container and `docker start`s it again at exit. No promotion here.
#   sudo systemd-run --unit=r881-glm53-agent-r4b --property=RuntimeMaxSec=43200 bash /srv/qwen5090/r881-glm53-agent-r4b.sh
set -uo pipefail
: "${HOME:=/root}"
PK=/srv/qwen5090/glm-agent-r4b-build
R=/srv/qwen5090/results/$(date +%F)-r881-glm53-agent-r4b-$(date +%H%M%S)
mkdir -p "$R"
exec > >(tee -a "$R/transcript.log") 2>&1
log(){ printf '%s [r881] %s\n' "$(date -Iseconds)" "$*"; }
note(){ log "$*"; printf '%s\n' "$*" >> "$R/summary.txt"; }
export GPU_QUEUE_NAME=r881-glm53-agent-r4b
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
note "GPU lock held; results $R"
for i in $(seq 90); do [ "$(sudo -n docker inspect --format '{{.State.Running}}' glm53 2>/dev/null)" = true ] && break; sleep 20; done
( cd "$PK/r4b" && bash build.sh ) > "$R/build.log" 2>&1
BUILD_RC=$?; note "build r4b rc=$BUILD_RC: $(tail -1 "$R/build.log" | cut -c1-120)"
((BUILD_RC == 0)) || exit 1
bash "$PK/r4b/operator-run.sh" "$R/run" > "$R/operator-run.log" 2>&1
note "operator-run rc=$?"
for arm in r4 r4b; do
  for f in "$R/run/$arm/toolcheck.log" "$R/run/$arm/cache-repeat.log"; do
    [ -f "$f" ] && note "$arm $(basename "$f" .log): $(grep -a '^RESULT' "$f" | python3 -c 'import json,sys
rs=[json.loads(l[7:]) for l in sys.stdin]; print(sum(r.get("passed") is True for r in rs), "of", len(rs), "passed; failed:", " ".join(r["tag"] for r in rs if not r.get("passed")))' | cut -c1-300)"
  done
done
note "glm53 running after: $(sudo -n docker inspect --format '{{.State.Running}} {{.Config.Image}}' glm53 2>/dev/null)"
exit 0

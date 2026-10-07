#!/usr/bin/env bash
set -euo pipefail
R4_TEST_DIR=$(cd "$(dirname "$0")" && pwd)
R4_PYTHON=${R4_PYTHON:-python3}
: "${R861_APP:?Set R861_APP to patched app}"
: "${R861_BASE:?Set R861_BASE to pristine app}"
export TOOLFIX_APP="$R861_APP" TOOLFIX_BASE="$R861_BASE"
export TOOLFIX_TEMPLATE="$R4_TEST_DIR/../glm53-chat_template.jinja"
unset TABBY_STREAM_TOOLCALLS TABBY_SSE_KEEPALIVE_S TABBY_GLM_TOOL_FIXES TABBY_GLM_FORCING TABBY_GLM_TAG_TRACE
"$R4_PYTHON" -c 'import tokenizers'
"$R4_PYTHON" -B "$R4_TEST_DIR/landing.py"
"$R4_PYTHON" -B "$R4_TEST_DIR/landing_fixes.py"
"$R4_PYTHON" -B "$R4_TEST_DIR/landing_r4.py"
for R4_SUITE in test_overlay test_review test_fixes test_r2 test_r3 test_toolcheck test_r4; do
    "$R4_PYTHON" -B "$R4_TEST_DIR/$R4_SUITE.py"
done
if [[ ${R4_CPU_ONLY:-0} == 1 ]]; then
    echo 'NOT EXECUTED: test_sse_runtime, test_sse_send_disconnect, test_lifecycle (sse-starlette dependency unavailable locally)'
else
    for R4_SUITE in test_sse_runtime test_sse_send_disconnect test_lifecycle; do
        "$R4_PYTHON" -B "$R4_TEST_DIR/$R4_SUITE.py"
    done
fi
